#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <chrono>
#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include "network_schedule.h" 

#define MAX_DDR_BUFFER_SIZE (32 * 1024 * 1024) 

struct FpgaOutput {
    int w, h, valid_channels;
    int y_zero;
    float y_scale;
    std::vector<float> data_float;
};

// 🚀 VARIABLES MONOLÍTICAS
cl_platform_id platform; cl_device_id device; cl_context context;
cl_command_queue queue_engine;
cl_program program; cl_kernel kernel_engine;
cl_mem d_weights, d_bias, d_schedule_data, d_ram_pool;
std::vector<ConvLayerDesc> active_schedule;
int total_layers;

#define CHECK_ERR(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << "🔴 ERROR FATAL: " << msg << " (Código: " << err << ")" << std::endl; \
        exit(1); \
    }

std::vector<unsigned char> load_binary_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) throw std::runtime_error("❌ No se pudo abrir " + filename);
    size_t size = file.tellg(); file.seekg(0, std::ios::beg);
    std::vector<unsigned char> buffer(size); file.read((char*)buffer.data(), size);
    return buffer;
}

void pad_weights_for_simd(std::vector<unsigned char>& w_data) {
    if (active_schedule[0].in_c % 4 == 0) return;
    int orig_in_c = active_schedule[0].in_c; 
    int pad_in_c = ((orig_in_c + 3) / 4) * 4; 
    int out_c = active_schedule[0].out_c;
    int w_zero = active_schedule[0].w_zero;
    
    int old_l0_size = out_c * 3 * 3 * orig_in_c;
    int new_l0_size = out_c * 3 * 3 * pad_in_c;
    std::vector<unsigned char> new_w; new_w.reserve(w_data.size() + (new_l0_size - old_l0_size));
    
    int old_idx = active_schedule[0].w_offset;
    for (int oc = 0; oc < out_c; ++oc) {
        for (int k = 0; k < 9; ++k) {
            for (int ic = 0; ic < pad_in_c; ++ic) {
                if (ic < orig_in_c) new_w.push_back(w_data[old_idx++]);
                else new_w.push_back(w_zero); 
            }
        }
    }
    for (size_t i = old_l0_size; i < w_data.size(); ++i) new_w.push_back(w_data[i]);
    int offset_diff = new_l0_size - old_l0_size;
    active_schedule[0].in_c = pad_in_c; 
    for (size_t i = 1; i < active_schedule.size(); ++i) active_schedule[i].w_offset += offset_diff;
    w_data = std::move(new_w);
}

void init_opencl(unsigned char* w_data, int* b_data, size_t w_size, size_t b_size) {
    cl_int err; cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    
    bool fpga_found = false;
    for(auto p : platforms) {
        char name[128];
        clGetPlatformInfo(p, CL_PLATFORM_NAME, 128, name, NULL);
        if (std::string(name).find("FPGA") != std::string::npos || std::string(name).find("Emulation") != std::string::npos) {
            platform = p; 
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
            fpga_found = true; 
            std::cout << "✅ Plataforma: " << name << std::endl;
            break;
        }
    }
    if(!fpga_found) { std::cerr << "❌ No se encontró plataforma Intel FPGA"; exit(1); }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue_engine = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    
    //  ATENCIÓN: Pon el nombre de tu .aocx final aquí
    std::vector<unsigned char> binary = load_binary_file("engineCNNV7.aocx"); 
    size_t binary_size = binary.size(); const unsigned char* binary_data = binary.data();
    program = clCreateProgramWithBinary(context, 1, &device, &binary_size, &binary_data, NULL, &err);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    kernel_engine = clCreateKernel(program, "cnn_engine", &err);

    total_layers = active_schedule.size();
    std::vector<int> flat_schedule(total_layers * 16, 0); 
    
    for(int i = 0; i < total_layers; i++) {
        flat_schedule[i*16 + 0] = active_schedule[i].in_offset;
        flat_schedule[i*16 + 1] = active_schedule[i].out_offset;
        flat_schedule[i*16 + 2] = active_schedule[i].w_offset;
        flat_schedule[i*16 + 3] = active_schedule[i].b_offset / 4; 
        flat_schedule[i*16 + 4] = active_schedule[i].w;
        flat_schedule[i*16 + 5] = active_schedule[i].h;
        flat_schedule[i*16 + 6] = active_schedule[i].in_c;
        flat_schedule[i*16 + 7] = active_schedule[i].out_c;
        flat_schedule[i*16 + 8] = active_schedule[i].stride;
        flat_schedule[i*16 + 9] = active_schedule[i].pad;
        flat_schedule[i*16 + 10] = active_schedule[i].x_zero;
        flat_schedule[i*16 + 11] = active_schedule[i].w_zero;
        flat_schedule[i*16 + 12] = active_schedule[i].y_zero;
        flat_schedule[i*16 + 13] = active_schedule[i].M_multiplier;
        flat_schedule[i*16 + 14] = active_schedule[i].M_shift;
        flat_schedule[i*16 + 15] = active_schedule[i].buf_out; 
    }

    d_schedule_data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, flat_schedule.size() * sizeof(int), flat_schedule.data(), &err);
    d_weights       = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_size, w_data, &err);
    d_bias          = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b_size, b_data, &err);
    d_ram_pool      = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, MAX_DDR_BUFFER_SIZE, NULL, &err);
}

double run_inference_monolithic() {
    cl_int err;
    clSetKernelArg(kernel_engine, 0, sizeof(cl_mem), &d_ram_pool);
    clSetKernelArg(kernel_engine, 1, sizeof(cl_mem), &d_weights);
    clSetKernelArg(kernel_engine, 2, sizeof(cl_mem), &d_bias);
    clSetKernelArg(kernel_engine, 3, sizeof(cl_mem), &d_schedule_data);
    clSetKernelArg(kernel_engine, 4, sizeof(int), &total_layers);

    cl_event event_engine;
    err = clEnqueueTask(queue_engine, kernel_engine, 0, NULL, &event_engine); 
    err = clFinish(queue_engine); 

    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event_engine, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event_engine, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    
    double pure_hardware_ms = (time_end - time_start) / 1000000.0;
    clReleaseEvent(event_engine);
    return pure_hardware_ms;
}

void custom_NMSBoxes(const std::vector<cv::Rect>& bboxes, const std::vector<float>& scores, float score_threshold, float nms_threshold, std::vector<int>& indices) {
    indices.clear(); std::vector<std::pair<float, int>> score_index_vec;
    for (size_t i = 0; i < bboxes.size(); i++) if (scores[i] > score_threshold) score_index_vec.push_back({scores[i], (int)i});
    std::sort(score_index_vec.begin(), score_index_vec.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
    std::vector<bool> suppressed(bboxes.size(), false);
    for (size_t i = 0; i < score_index_vec.size(); i++) {
        int idx = score_index_vec[i].second;
        if (suppressed[idx]) continue;
        indices.push_back(idx);
        for (size_t j = i + 1; j < score_index_vec.size(); j++) {
            int next_idx = score_index_vec[j].second;
            if (suppressed[next_idx]) continue;
            cv::Rect intersection = bboxes[idx] & bboxes[next_idx];
            float inter_area = intersection.area();
            float union_area = bboxes[idx].area() + bboxes[next_idx].area() - inter_area;
            if ((inter_area / union_area) > nms_threshold) suppressed[next_idx] = true;
        }
    }
}

void post_process_and_draw(cv::Mat& image, unsigned char* ram_pool_ptr) {
    std::vector<FpgaOutput> all_outputs;
    for (int i = 0; i < active_schedule.size(); i++) {
        ConvLayerDesc layer = active_schedule[i];
        if (layer.buf_out == 2) {
            int out_w = (layer.w + 2 * layer.pad - 3) / layer.stride + 1;
            int out_h = (layer.h + 2 * layer.pad - 3) / layer.stride + 1;
            FpgaOutput f_out; f_out.w = out_w; f_out.h = out_h; f_out.valid_channels = layer.out_c_orig;
            f_out.y_zero = layer.y_zero; f_out.y_scale = layer.y_scale;
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    for (int c = 0; c < f_out.valid_channels; ++c) {
                        int index = layer.out_offset + (y * out_w * layer.out_c) + (x * layer.out_c) + c;
                        f_out.data_float.push_back(((int)ram_pool_ptr[index] - layer.y_zero) * layer.y_scale);
                    }
                }
            }
            all_outputs.push_back(f_out);
        }
    }
    std::vector<FpgaOutput> confs, locs;
    for(auto& o : all_outputs) { if (o.valid_channels <= 6) confs.push_back(o); else locs.push_back(o); }
    std::vector<cv::Rect> bboxes; std::vector<float> scores;
    float SCORE_THRESHOLD = 0.50f; float NMS_THRESHOLD = 0.40f;
    std::vector<std::vector<int>> sizes = {{10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}};
    std::vector<int> steps = {8, 16, 32, 64};

    for (size_t i = 0; i < confs.size() && i < locs.size(); i++) {
        int f_w = confs[i].w; int f_h = confs[i].h; int n_anchors = sizes[i].size();
        for (int y = 0; y < f_h; ++y) {
            for (int x = 0; x < f_w; ++x) {
                for (int a = 0; a < n_anchors; ++a) {
                    int p_idx = y * f_w + x; int c_idx = (p_idx * n_anchors * 2) + (a * 2);
                    float c_bg = confs[i].data_float[c_idx]; float c_face = confs[i].data_float[c_idx + 1]; 
                    float max_c = std::max(c_bg, c_face);
                    float prob = std::exp(c_face - max_c) / (std::exp(c_bg - max_c) + std::exp(c_face - max_c));
                    if (prob > SCORE_THRESHOLD) {
                        int l_idx = (p_idx * n_anchors * 4) + (a * 4);
                        float dx = locs[i].data_float[l_idx + 0]; float dy = locs[i].data_float[l_idx + 1];
                        float dw = locs[i].data_float[l_idx + 2]; float dh = locs[i].data_float[l_idx + 3];
                        float p_cx = (x + 0.5f) * steps[i] / 320.0f; float p_cy = (y + 0.5f) * steps[i] / 240.0f;
                        float p_w  = (float)sizes[i][a] / 320.0f; float p_h  = (float)sizes[i][a] / 240.0f;
                        float r_cx = p_cx + dx * 0.1f * p_w; float r_cy = p_cy + dy * 0.1f * p_h;
                        float r_w  = p_w * std::exp(dw * 0.2f); float r_h  = p_h * std::exp(dh * 0.2f);
                        int x_min = std::max(0, (int)((r_cx - r_w/2) * image.cols));
                        int y_min = std::max(0, (int)((r_cy - r_h/2) * image.rows));
                        int x_max = std::min(image.cols, (int)((r_cx + r_w/2) * image.cols));
                        int y_max = std::min(image.rows, (int)((r_cy + r_h/2) * image.rows));
                        int final_w = x_max - x_min; int final_h = y_max - y_min;
                        if (final_w > 0 && final_h > 0) {
                            bboxes.push_back(cv::Rect(x_min, y_min, final_w, final_h));
                            scores.push_back(prob);
                        }
                    }
                }
            }
        }
    }
    std::vector<int> indices; custom_NMSBoxes(bboxes, scores, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int idx : indices) {
        cv::rectangle(image, bboxes[idx], cv::Scalar(0, 255, 0), 2);
        std::string label = cv::format("%.1f%%", scores[idx] * 100);
        cv::putText(image, label, cv::Point(bboxes[idx].x, bboxes[idx].y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
}

int main() {
    std::cout << " MODO MONOLÍTICO EN VIVO INICIADO" << std::endl;
    active_schedule.assign(network_schedule, network_schedule + TOTAL_LAYERS);
    
    std::vector<unsigned char> rw = load_binary_file("weights.bin");
    std::vector<unsigned char> rb_bytes = load_binary_file("bias.bin");
    std::vector<int> rb(rb_bytes.size() / sizeof(int));
    std::memcpy(rb.data(), rb_bytes.data(), rb_bytes.size());
    
    pad_weights_for_simd(rw);
    rw.resize(rw.size() + 500000, active_schedule[0].w_zero);
    
    init_opencl(rw.data(), rb.data(), rw.size(), rb_bytes.size());

    // ABRIR LA CÁMARA (El 0 suele ser la webcam por defecto o la cámara USB)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { 
        std::cerr << "❌ Error al abrir la cámara" << std::endl; 
        return -1; 
    }
    
    cv::Mat frame, resized_img, rgb_img;
    unsigned char x_z = active_schedule[0].x_zero;
    int first_in_c = active_schedule[0].in_c;
    cl_int err;

    std::cout << "🎥 Cámara lista. Pulsa 'ESC' para salir." << std::endl;

    // Variables para calcular los FPS
    auto time_start_fps = std::chrono::steady_clock::now();
    int frame_count = 0;
    float current_fps = 0.0f;

    // EL BUCLE PRINCIPAL DE VÍDEO
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << " Frame vacío recibido. Saliendo..." << std::endl;
            break;
        }

        cv::resize(frame, resized_img, cv::Size(320, 240)); 
        cv::cvtColor(resized_img, rgb_img, cv::COLOR_BGR2RGB);

        // Mapeamos para escribir la nueva imagen
        unsigned char* host_pool_ptr = (unsigned char*)clEnqueueMapBuffer(
            queue_engine, d_ram_pool, CL_TRUE, CL_MAP_WRITE, 0, MAX_DDR_BUFFER_SIZE, 0, NULL, NULL, &err);

        for (int y = 0; y < 240; ++y) {
            for (int x = 0; x < 320; ++x) {
                cv::Vec3b p = rgb_img.at<cv::Vec3b>(y, x);
                int base = (y * 320 + x) * first_in_c;
                host_pool_ptr[base + 0] = p[0]; host_pool_ptr[base + 1] = p[1]; host_pool_ptr[base + 2] = p[2]; 
                if (first_in_c > 3) host_pool_ptr[base + 3] = x_z;
                
                for (int c = 4; c < first_in_c; ++c) {
                    host_pool_ptr[base + c] = x_z;
                }
            }
        }
        clEnqueueUnmapMemObject(queue_engine, d_ram_pool, host_pool_ptr, 0, NULL, NULL);

        // LA INFERENCIA MASIVA EN LA FPGA
        double hw_ms = run_inference_monolithic();

        // Mapeamos para extraer las Bounding Boxes
        host_pool_ptr = (unsigned char*)clEnqueueMapBuffer(
            queue_engine, d_ram_pool, CL_TRUE, CL_MAP_READ, 0, MAX_DDR_BUFFER_SIZE, 0, NULL, NULL, &err);

        // Dibujamos sobre el 'frame' original de alta resolución
        post_process_and_draw(frame, host_pool_ptr);
        clEnqueueUnmapMemObject(queue_engine, d_ram_pool, host_pool_ptr, 0, NULL, NULL);

        // CALCULAR Y DIBUJAR ESTADÍSTICAS EN PANTALLA
        frame_count++;
        auto time_now = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = time_now - time_start_fps;
        
        if (elapsed.count() >= 1.0f) { // Actualizar FPS cada segundo
            current_fps = frame_count / elapsed.count();
            frame_count = 0;
            time_start_fps = time_now;
        }

        std::string stats_hw = cv::format("FPGA Tiempo Puro: %.2f ms", hw_ms);
        std::string stats_fps = cv::format("FPS Totales: %.1f", current_fps);
        
        cv::putText(frame, stats_hw, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, stats_fps, cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

        // 5. MOSTRAR EL RESULTADO
        cv::imshow("Acelerador IA - FPGA", frame);

        // Salir si se pulsa la tecla ESC
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // Limpieza
    cap.release();
    cv::destroyAllWindows();
    std::cout << "👋 Programa finalizado correctamente." << std::endl;
    
    return 0;
}