#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/acl/acl_provider_factory.h>


// Estructura para almacenar los puntos de anclaje de la red
struct AnchorPrior {
    float cx, cy, w, h;
};

// --- FUNCIÓN DE SUPRESIÓN DE NO MÁXIMOS (NMS) EXACTAMENTE IGUAL A TU FPGA ---
void custom_NMSBoxes(const std::vector<cv::Rect>& bboxes, const std::vector<float>& scores, float score_threshold, float nms_threshold, std::vector<int>& indices) {
    indices.clear(); 
    std::vector<std::pair<float, int>> score_index_vec;
    for (size_t i = 0; i < bboxes.size(); i++) {
        if (scores[i] > score_threshold) score_index_vec.push_back({scores[i], (int)i});
    }
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <ruta_al_modelo.onnx>" << std::endl;
        return -1;
    }
    const char* model_path = argv[1];

    std::cout << "==================================================" << std::endl;
    std::cout << " BENCHMARK COMPLETO: DECODIFICACIÓN EN CPU ARM    " << std::endl;
    std::cout << "==================================================" << std::endl;

    // 1. INICIALIZAR ENTORNO ONNX (1 Hilo para mantener comparativa justa)
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_ARM");
    Ort::SessionOptions session_options; 
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(1); 
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ACL(session_options, 1));
    Ort::Session session(env, model_path, session_options);

    const char* input_names[] = {"input_quantized"};   
    const char* output_names[] = {"scores", "boxes"};   

    std::vector<int64_t> input_shape = {1, 3, 240, 320}; 
    size_t input_tensor_size = 3 * 240 * 320;
    std::vector<uint8_t> input_tensor_values(input_tensor_size);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // --- GENERACIÓN DE ANCHOR PRIORS (4420 en total de tus mapas de características) ---
    std::vector<AnchorPrior> priors;
    priors.reserve(4420);
    
    std::vector<std::vector<int>> sizes = {{10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}};
    std::vector<int> steps = {8, 16, 32, 64};
    std::vector<std::pair<int, int>> grids = {{30, 40}, {15, 20}, {8, 10}, {4, 5}};

    for (size_t i = 0; i < grids.size(); i++) {
        int f_h = grids[i].first;
        int f_w = grids[i].second;
        int n_anchors = sizes[i].size();
        for (int y = 0; y < f_h; ++y) {
            for (int x = 0; x < f_w; ++x) {
                for (int a = 0; a < n_anchors; ++a) {
                    AnchorPrior p;
                    p.cx = (x + 0.5f) * steps[i] / 320.0f;
                    p.cy = (y + 0.5f) * steps[i] / 240.0f;
                    p.w  = (float)sizes[i][a] / 320.0f;
                    p.h  = (float)sizes[i][a] / 240.0f;
                    priors.push_back(p);
                }
            }
        }
    }

    // 2. INICIALIZAR LA CÁMARA WEB
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { std::cerr << "Error al abrir la cámara." << std::endl; return -1; }
    
    cv::Mat frame, resized_img, rgb_img;
    double avg_inference_ms = 0.0;
    int frame_count = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Preprocesado
        cv::resize(frame, resized_img, cv::Size(320, 240)); 
        cv::cvtColor(resized_img, rgb_img, cv::COLOR_BGR2RGB);

        int idx = 0;
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < 240; ++y) {
                for (int x = 0; x < 320; ++x) {
                    input_tensor_values[idx++] = rgb_img.at<cv::Vec3b>(y, x)[c];
                }
            }
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor<uint8_t>(
            memory_info, input_tensor_values.data(), input_tensor_size,
            input_shape.data(), input_shape.size());

        // Inferencia
        auto t_start = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 2);
        auto t_end = std::chrono::high_resolution_clock::now();
        
        double current_inference_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // Punteros a los tensores decodificados por ONNX
        float* raw_scores = output_tensors[0].GetTensorMutableData<float>(); // 4420 x 2 (Softmax ya aplicado)
        float* raw_boxes  = output_tensors[1].GetTensorMutableData<float>(); // 4420 x 4 (Deltas dx, dy, dw, dh)

        // --- C. DECODIFICACIÓN DE LAS 4420 CAJAS DEL MODELO ---
        std::vector<cv::Rect> bboxes;
        std::vector<float> scores;

        for (int i = 0; i < 4420; ++i) {
            float prob = raw_scores[i * 2 + 1]; // Canal 1: Probabilidad de cara (Softmax del grafo)

            if (prob > 0.50f) { // Score Threshold
                float dx = raw_boxes[i * 4 + 0];
                float dy = raw_boxes[i * 4 + 1];
                float dw = raw_boxes[i * 4 + 2];
                float dh = raw_boxes[i * 4 + 3];

                AnchorPrior p = priors[i];
                
                // Aplicar la transformación de los cuadros delimitadores
                float r_cx = p.cx + dx * 0.1f * p.w;
                float r_cy = p.cy + dy * 0.1f * p.h;
                float r_w  = p.w * std::exp(dw * 0.2f);
                float r_h  = p.h * std::exp(dh * 0.2f);

                // Convertir coordenadas relativas (0.0 a 1.0) a píxeles de la cámara original
                int x_min = std::max(0, (int)((r_cx - r_w / 2.0f) * frame.cols));
                int y_min = std::max(0, (int)((r_cy - r_h / 2.0f) * frame.rows));
                int x_max = std::min(frame.cols, (int)((r_cx + r_w / 2.0f) * frame.cols));
                int y_max = std::min(frame.rows, (int)((r_cy + r_h / 2.0f) * frame.rows));
                
                int final_w = x_max - x_min;
                int final_h = y_max - y_min;

                if (final_w > 0 && final_h > 0) {
                    bboxes.push_back(cv::Rect(x_min, y_min, final_w, final_h));
                    scores.push_back(prob);
                }
            }
        }

        // --- D. APLICAR FILTRADO NMS Y DIBUJAR ---
        std::vector<int> indices;
        custom_NMSBoxes(bboxes, scores, 0.50f, 0.40f, indices);

        for (int idx_box : indices) {
            cv::rectangle(frame, bboxes[idx_box], cv::Scalar(0, 255, 0), 2);
            std::string label = cv::format("%.1f%%", scores[idx_box] * 100);
            cv::putText(frame, label, cv::Point(bboxes[idx_box].x, bboxes[idx_box].y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        // Métricas
        if (frame_count == 0) avg_inference_ms = current_inference_ms;
        else avg_inference_ms = 0.9 * avg_inference_ms + 0.1 * current_inference_ms;
        frame_count++;

        cv::putText(frame, cv::format("Latencia Inferencia: %.1f ms", current_inference_ms), cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, cv::format("Latencia Media:     %.1f ms", avg_inference_ms),     cv::Point(15, 65), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(frame, cv::format("FPS Red (CPU):      %.1f FPS", 1000.0 / avg_inference_ms), cv::Point(15, 100), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Benchmark CPU - Inferencia de Red INT8", frame);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}