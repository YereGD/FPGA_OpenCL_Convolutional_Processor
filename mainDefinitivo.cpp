#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/acl/acl_provider_factory.h>
#include "network_schedule.h" 

#define MAX_DDR_BUFFER_SIZE (32 * 1024 * 1024) 

// --- VARIABLES OPENCL ---
cl_platform_id platform; cl_device_id device; cl_context context;
cl_command_queue queue_engine;  
cl_command_queue queue_mem;     
cl_program program; cl_kernel kernel_engine;
cl_mem d_weights, d_bias, d_schedule_data;
cl_mem d_ram_pool[2]; 
std::vector<ConvLayerDesc> active_schedule;
int total_layers;

// --- BUZONES MULTIHILO ---
struct { 
    cv::Mat frame_bgr; 
    std::vector<unsigned char> fpga_buffer; 
    bool is_new = false; 
} mail_cam;
std::mutex mtx_cam;

struct {
    cv::Mat frame;
    bool is_new = false;
} mail_display;
std::mutex mtx_display;

#define CHECK_ERR(err, msg) if (err != CL_SUCCESS) { std::cerr << "?? ERROR: " << msg << std::endl; exit(1); }

std::vector<unsigned char> load_binary_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) throw std::runtime_error("? Error abriendo " + filename);
    size_t size = file.tellg(); file.seekg(0, std::ios::beg);
    std::vector<unsigned char> buffer(size); file.read((char*)buffer.data(), size);
    return buffer;
}

void init_opencl(unsigned char* w_data, int* b_data, size_t w_size, size_t b_size) {
    cl_int err; cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    for(auto p : platforms) {
        char name[128]; clGetPlatformInfo(p, CL_PLATFORM_NAME, 128, name, NULL);
        if (std::string(name).find("FPGA") != std::string::npos || std::string(name).find("Emulation") != std::string::npos) {
            platform = p; clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL); break;
        }
    }
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue_engine = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    queue_mem = clCreateCommandQueue(context, device, 0, &err); 
    
    std::vector<unsigned char> binary = load_binary_file("engineCNNV8.aocx"); 
    size_t binary_size = binary.size(); const unsigned char* binary_data = binary.data();
    program = clCreateProgramWithBinary(context, 1, &device, &binary_size, &binary_data, NULL, &err);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    kernel_engine = clCreateKernel(program, "cnn_engine", &err);

    total_layers = active_schedule.size();
    std::vector<int> flat_schedule(total_layers * 16, 0); 
    for(int i = 0; i < total_layers; i++) {
        flat_schedule[i*16 + 0] = active_schedule[i].in_offset; flat_schedule[i*16 + 1] = active_schedule[i].out_offset;
        flat_schedule[i*16 + 2] = active_schedule[i].w_offset; flat_schedule[i*16 + 3] = active_schedule[i].b_offset / 4; 
        flat_schedule[i*16 + 4] = active_schedule[i].w; flat_schedule[i*16 + 5] = active_schedule[i].h;
        flat_schedule[i*16 + 6] = active_schedule[i].in_c; flat_schedule[i*16 + 7] = active_schedule[i].out_c;
        flat_schedule[i*16 + 8] = active_schedule[i].stride; flat_schedule[i*16 + 9] = active_schedule[i].pad;
        flat_schedule[i*16 + 10] = active_schedule[i].x_zero; flat_schedule[i*16 + 11] = active_schedule[i].w_zero;
        flat_schedule[i*16 + 12] = active_schedule[i].y_zero; flat_schedule[i*16 + 13] = active_schedule[i].M_multiplier;
        flat_schedule[i*16 + 14] = active_schedule[i].M_shift; flat_schedule[i*16 + 15] = active_schedule[i].buf_out; 
    }
    d_schedule_data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, flat_schedule.size() * sizeof(int), flat_schedule.data(), &err);
    d_weights       = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_size, w_data, &err);
    d_bias          = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b_size, b_data, &err);
    d_ram_pool[0]   = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, MAX_DDR_BUFFER_SIZE, NULL, &err);
    d_ram_pool[1]   = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, MAX_DDR_BUFFER_SIZE, NULL, &err);
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

void post_process_onnx(cv::Mat& image, float* scores_ptr, float* boxes_ptr) {
    // ?? UMBRAL REDUCIDO para hacer el tracking más robusto al movimiento
    float SCORE_THRESHOLD = 0.38f; 
    float NMS_THRESHOLD = 0.40f;
    std::vector<std::vector<int>> sizes = {{10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}};
    std::vector<int> steps = {8, 16, 32, 64};
    std::vector<std::pair<int, int>> fmaps = {{40, 30}, {20, 15}, {10, 8}, {5, 4}};

    std::vector<cv::Rect> bboxes; 
    std::vector<float> final_scores;
    int global_idx = 0;

    for (size_t i = 0; i < 4; i++) {
        int f_w = fmaps[i].first; int f_h = fmaps[i].second; int n_anchors = sizes[i].size();
        for (int y = 0; y < f_h; ++y) {
            for (int x = 0; x < f_w; ++x) {
                for (int a = 0; a < n_anchors; ++a) {
                    float prob = scores_ptr[global_idx * 2 + 1]; 
                    if (prob > SCORE_THRESHOLD) {
                        float dx = boxes_ptr[global_idx * 4 + 0], dy = boxes_ptr[global_idx * 4 + 1];
                        float dw = boxes_ptr[global_idx * 4 + 2], dh = boxes_ptr[global_idx * 4 + 3];

                        float p_cx = (x + 0.5f) * steps[i] / 320.0f, p_cy = (y + 0.5f) * steps[i] / 240.0f;
                        float p_w  = (float)sizes[i][a] / 320.0f, p_h  = (float)sizes[i][a] / 240.0f;

                        float r_cx = p_cx + dx * 0.1f * p_w, r_cy = p_cy + dy * 0.1f * p_h;
                        float r_w  = p_w * std::exp(dw * 0.2f), r_h  = p_h * std::exp(dh * 0.2f);

                        int x_min = std::max(0, (int)((r_cx - r_w/2) * image.cols));
                        int y_min = std::max(0, (int)((r_cy - r_h/2) * image.rows));
                        int x_max = std::min(image.cols, (int)((r_cx + r_w/2) * image.cols));
                        int y_max = std::min(image.rows, (int)((r_cy + r_h/2) * image.rows));
                        int final_w = x_max - x_min, final_h = y_max - y_min;

                        if (final_w > 0 && final_h > 0) {
                            bboxes.push_back(cv::Rect(x_min, y_min, final_w, final_h));
                            final_scores.push_back(prob);
                        }
                    }
                    global_idx++;
                }
            }
        }
    }
    std::vector<int> indices; custom_NMSBoxes(bboxes, final_scores, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int idx : indices) {
        cv::rectangle(image, bboxes[idx], cv::Scalar(0, 255, 0), 2);
        std::string label = cv::format("%.1f%%", final_scores[idx] * 100);
        cv::putText(image, label, cv::Point(bboxes[idx].x, bboxes[idx].y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
}

int main() {
    std::cout << "?? MODO ACORAZADO TOTAL V2: SINCRONIZACIÓN PERFECTA" << std::endl;
    
    int capas_fpga = 14; 
    active_schedule.assign(network_schedule, network_schedule + capas_fpga);
    
    std::vector<unsigned char> rw = load_binary_file("weights.bin");
    std::vector<unsigned char> rb_bytes = load_binary_file("bias.bin");
    std::vector<int> rb(rb_bytes.size() / sizeof(int));
    std::memcpy(rb.data(), rb_bytes.data(), rb_bytes.size());
    init_opencl(rw.data(), rb.data(), rw.size(), rb_bytes.size());

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_ARM");
    Ort::SessionOptions session_options; 
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(1); 
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ACL(session_options, 1));
    Ort::Session session(env, "modelo_cola_cpu.onnx", session_options);
    
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    std::vector<const char*> input_node_names = {input_name_ptr.get()};
    
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> output_names_ptrs;
    std::vector<const char*> output_node_names;
    for (size_t i = 0; i < num_output_nodes; i++) {
        output_names_ptrs.push_back(session.GetOutputNameAllocated(i, allocator));
        output_node_names.push_back(output_names_ptrs.back().get());
    }
    
    std::vector<int64_t> input_shape = {1, 64, 30, 40}; 
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<uint8_t> onnx_input_nchw(64 * 30 * 40, 0); 

    unsigned char x_z = active_schedule[0].x_zero;
    int first_in_c = active_schedule[0].in_c;

    std::atomic<bool> sistema_corriendo(true);

    // --- HILO 1: CÁMARA OBRERA ---
    std::thread hilo_camara([&]() {
        cv::VideoCapture cap(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 320); cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240); cap.set(cv::CAP_PROP_FPS, 30); 
        cv::Mat frame_temp, resized_img, rgb_img;
        std::vector<unsigned char> local_buf(320 * 240 * first_in_c);

        while (sistema_corriendo) {
            cap >> frame_temp; 
            if (frame_temp.empty()) continue;

            bool maestro_ocupado = false;
            {
                std::lock_guard<std::mutex> lock(mtx_cam);
                maestro_ocupado = mail_cam.is_new;
            }
            if (maestro_ocupado) continue; 

            cv::resize(frame_temp, resized_img, cv::Size(320, 240)); 
            cv::cvtColor(resized_img, rgb_img, cv::COLOR_BGR2RGB);

            for (int y = 0; y < 240; ++y) {
                cv::Vec3b* ptr_fila = rgb_img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < 320; ++x) {
                    int base = (y * 320 + x) * first_in_c;
                    local_buf[base + 0] = ptr_fila[x][0]; 
                    local_buf[base + 1] = ptr_fila[x][1]; 
                    local_buf[base + 2] = ptr_fila[x][2]; 
                    if (first_in_c > 3) local_buf[base + 3] = x_z;
                    for (int c = 4; c < first_in_c; ++c) local_buf[base + c] = x_z;
                }
            }

            {
                std::lock_guard<std::mutex> lock(mtx_cam);
                mail_cam.frame_bgr = resized_img.clone(); 
                std::swap(mail_cam.fpga_buffer, local_buf);
                mail_cam.is_new = true;
            }
            if (local_buf.size() != 320 * 240 * first_in_c) local_buf.resize(320 * 240 * first_in_c);
        }
        cap.release();
    });

    // --- HILO 2: PANTALLA ---
    std::thread hilo_display([&]() {
        cv::Mat frame_local;
        while (sistema_corriendo) {
            bool has_frame = false;
            {
                std::lock_guard<std::mutex> lock(mtx_display);
                if (mail_display.is_new) {
                    frame_local = mail_display.frame.clone();
                    mail_display.is_new = false;
                    has_frame = true;
                }
            }
            
            if (has_frame && !frame_local.empty()) {
                cv::imshow("Frankenstein Ping-Pong", frame_local);
                int key = cv::waitKey(1); 
                if (key == 27) sistema_corriendo = false;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(5)); 
            }
        }
        cv::destroyAllWindows();
    });

    cl_int err;
    std::cout << "? Esperando a la cámara obrera..." << std::endl;
    cv::Mat frames[2];
    int curr = 0;

    while (sistema_corriendo) {
        {
            std::lock_guard<std::mutex> lock(mtx_cam);
            if (mail_cam.is_new) { 
                frames[curr] = mail_cam.frame_bgr.clone(); 
                mail_cam.is_new = false; 
                break; 
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    if (!sistema_corriendo) return 0; 

    // --- WARMUP ---
    unsigned char* host_pool_ptr = (unsigned char*)clEnqueueMapBuffer(queue_mem, d_ram_pool[curr], CL_TRUE, CL_MAP_WRITE, 0, MAX_DDR_BUFFER_SIZE, 0, NULL, NULL, &err);
    std::memcpy(host_pool_ptr, mail_cam.fpga_buffer.data(), mail_cam.fpga_buffer.size());
    clEnqueueUnmapMemObject(queue_mem, d_ram_pool[curr], host_pool_ptr, 0, NULL, NULL);
    clFlush(queue_mem);

    bool is_first = true;
    double ms_fpga = 0.0;
    auto time_start_fps = std::chrono::steady_clock::now();
    int frame_count = 0; float current_fps = 0.0f;

    std::cout << "? Profiler Acorazado Total V2 en marcha." << std::endl;

    // --- BUCLE PRINCIPAL ---
    while (sistema_corriendo) {
        clSetKernelArg(kernel_engine, 0, sizeof(cl_mem), &d_ram_pool[curr]);
        clSetKernelArg(kernel_engine, 1, sizeof(cl_mem), &d_weights);
        clSetKernelArg(kernel_engine, 2, sizeof(cl_mem), &d_bias);
        clSetKernelArg(kernel_engine, 3, sizeof(cl_mem), &d_schedule_data);
        clSetKernelArg(kernel_engine, 4, sizeof(int), &total_layers);

        cl_event fpga_event;
        clEnqueueTask(queue_engine, kernel_engine, 0, NULL, &fpga_event);
        clFlush(queue_engine); 

        int prev = 1 - curr; 
        int next = 1 - curr;

        double ms_onnx = 0, ms_post = 0, ms_prep_next = 0, ms_cpu_wait = 0;

        if (!is_first && !frames[prev].empty()) {
            auto t_onnx_start = std::chrono::steady_clock::now();
            unsigned char* out_ptr = (unsigned char*)clEnqueueMapBuffer(queue_mem, d_ram_pool[prev], CL_TRUE, CL_MAP_READ, 0, MAX_DDR_BUFFER_SIZE, 0, NULL, NULL, &err);
            
            int h = 30, w = 40, in_c = 64, out_offset = active_schedule[13].out_offset; 
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    for (int c = 0; c < in_c; ++c) {
                        onnx_input_nchw[c * 1200 + y * w + x] = (uint8_t)out_ptr[out_offset + (y * w * in_c) + (x * in_c) + c];
                    }
                }
            }
            clEnqueueUnmapMemObject(queue_mem, d_ram_pool[prev], out_ptr, 0, NULL, NULL);

            Ort::Value input_tensor = Ort::Value::CreateTensor<uint8_t>(memory_info, onnx_input_nchw.data(), onnx_input_nchw.size(), input_shape.data(), input_shape.size());
            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), output_node_names.size()); 
            auto t_onnx_end = std::chrono::steady_clock::now();
            ms_onnx = std::chrono::duration<double, std::milli>(t_onnx_end - t_onnx_start).count();

            auto t_post_start = std::chrono::steady_clock::now();
            float *scores_ptr = nullptr, *boxes_ptr = nullptr;
            for (size_t i = 0; i < output_tensors.size(); i++) {
                auto shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
                if (shape.size() == 3 && shape[2] == 2) scores_ptr = output_tensors[i].GetTensorMutableData<float>();
                if (shape.size() == 3 && shape[2] == 4) boxes_ptr = output_tensors[i].GetTensorMutableData<float>();
            }
            if (scores_ptr && boxes_ptr) post_process_onnx(frames[prev], scores_ptr, boxes_ptr);
            auto t_post_end = std::chrono::steady_clock::now();
            ms_post = std::chrono::duration<double, std::milli>(t_post_end - t_post_start).count();

            frame_count++;
            auto time_now = std::chrono::steady_clock::now();
            std::chrono::duration<float> elapsed = time_now - time_start_fps;
            if (elapsed.count() >= 1.0f) {
                current_fps = frame_count / elapsed.count();
                frame_count = 0; time_start_fps = time_now;
            }
            cv::putText(frames[prev], cv::format("FPS: %.1f", current_fps), cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

            {
                std::lock_guard<std::mutex> lock(mtx_display);
                mail_display.frame = frames[prev].clone();
                mail_display.is_new = true;
            }
        }

        // ?? LA MAGIA ANTIBASURA: ESPERA INTELIGENTE
        auto t_prep_start = std::chrono::steady_clock::now();
        bool got_new_frame = false;
        std::vector<unsigned char> local_host_buf;
        
        while (!got_new_frame && sistema_corriendo) {
            {
                std::lock_guard<std::mutex> lock(mtx_cam);
                if (mail_cam.is_new) { 
                    cv::swap(frames[next], mail_cam.frame_bgr); 
                    std::swap(local_host_buf, mail_cam.fpga_buffer); 
                    mail_cam.is_new = false; 
                    got_new_frame = true;
                }
            }
            // Si la cámara va con retraso (por ej. bajada de luz), esperamos pacientemente 2ms.
            // Esto garantiza que la FPGA JAMÁS procese memoria vieja o corrupta.
            if (!got_new_frame) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        
        if (got_new_frame && !local_host_buf.empty()) {
            unsigned char* in_ptr = (unsigned char*)clEnqueueMapBuffer(queue_mem, d_ram_pool[next], CL_TRUE, CL_MAP_WRITE, 0, MAX_DDR_BUFFER_SIZE, 0, NULL, NULL, &err);
            std::memcpy(in_ptr, local_host_buf.data(), local_host_buf.size()); 
            clEnqueueUnmapMemObject(queue_mem, d_ram_pool[next], in_ptr, 0, NULL, NULL);
            clFlush(queue_mem); 
        }
        auto t_prep_end = std::chrono::steady_clock::now();
        ms_prep_next = std::chrono::duration<double, std::milli>(t_prep_end - t_prep_start).count();

        // ESPERAR A FPGA
        auto t_wait_start = std::chrono::steady_clock::now();
        clWaitForEvents(1, &fpga_event);
        auto t_wait_end = std::chrono::steady_clock::now();
        ms_cpu_wait = std::chrono::duration<double, std::milli>(t_wait_end - t_wait_start).count();
        
        cl_ulong t_start, t_end;
        clGetEventProfilingInfo(fpga_event, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
        clGetEventProfilingInfo(fpga_event, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
        ms_fpga = (t_end - t_start) / 1000000.0;
        clReleaseEvent(fpga_event);

        if (!is_first && !frames[next].empty()) {
            cv::putText(frames[next], cv::format("FPGA T.Puro: %.2f ms", ms_fpga), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
            cv::putText(frames[next], cv::format("CPU T.Puro: %.2f ms", ms_onnx), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

        // IMPRIMIR PROFILING (Terminal)
        static int print_counter = 0;
        if (!is_first && print_counter++ % 10 == 0) {
            double ms_cpu_total = ms_onnx + ms_post + ms_prep_next;
            std::cout << "\n=== PROFILER ACORAZADO TOTAL V2 ===" << std::endl;
            std::cout << "[Hardware] FPGA T.Puro  : " << ms_fpga << " ms" << std::endl;
            std::cout << "[Software] ONNX CPU     : " << ms_onnx << " ms" << std::endl;
            std::cout << "[Software] Prep Next Mem: " << ms_prep_next << " ms" << std::endl;
            std::cout << "------------------------" << std::endl;
            std::cout << "CPU Total Ocupada       : " << ms_cpu_total << " ms" << std::endl;
            std::cout << "CPU Ociosa (Esperando)  : " << ms_cpu_wait << " ms" << std::endl;
        }

        is_first = false;
        curr = next; 
    }

    hilo_camara.join();
    hilo_display.join();
    return 0;
}