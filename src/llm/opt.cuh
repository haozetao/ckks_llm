#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <fstream> 
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "../ckks/include/TimeUtils.cuh"

class opt_model {
public:
    __half *embed_tokens_weight = nullptr;
    __half *embed_positions_weight = nullptr;

    __half *k_proj_weight[32];
    __half *k_proj_bias[32];
    __half *v_proj_weight[32];
    __half *v_proj_bias[32];
    __half *q_proj_weight[32];
    __half *q_proj_bias[32];
    __half *out_proj_weight[32];
    __half *out_proj_bias[32];
    __half *attn_layer_norm_weight[32];
    __half *attn_layer_norm_bias[32];
    __half *fc1_weight[32];
    __half *fc1_bias[32];
    __half *fc2_weight[32];
    __half *fc2_bias[32];
    __half *layers_final_layer_norm_weight[32];
    __half *layers_final_layer_norm_bias[32];

    __half *final_layer_norm_weight = nullptr;
    __half *final_layer_norm_bias = nullptr;
    
    int num_layers = 32;
    int embed_tokens_weight_row = 50272;
    int embed_tokens_weight_column = 4096;
    int embed_positions_weight_row = 2050;
    int embed_positions_weight_column = 4096;

    int kqv_proj_row = 4096;
    int kqv_proj_column = 4096;
    int attn_layer_norm_column = 4096;
    int fc1_weight_row = 16384;
    int fc1_weight_column = 4096;
    int fc2_weight_row = 4096;
    int fc2_weight_column = 16384;
    int layer_final_layer_norm_column = 4096;

    int final_layer_norm_weight_column = 4096;

    opt_model(std::string model_catalog) {

        // 加载权重的辅助lambda
        auto mmap_load_tensor = [](const std::string& file, __half*& weight, size_t size) {
            int fd = open(file.c_str(), O_RDONLY);
            if (fd < 0) {
                throw std::runtime_error("Failed to open " + file);
            }
            off_t file_size = lseek(fd, 0, SEEK_END);
            if (file_size < (off_t)size) {
                close(fd);
                throw std::runtime_error("File too small: " + file);
            }
            void* file_map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (file_map == MAP_FAILED) {
                close(fd);
                throw std::runtime_error("Failed to mmap " + file);
            }
            cudaMalloc(&weight, size);
            if (weight == nullptr) {
                munmap(file_map, size);
                close(fd);
                throw std::runtime_error("Failed to allocate memory for " + file);
            }
            cudaMemcpy(weight, file_map, size, cudaMemcpyHostToDevice);
            munmap(file_map, size);
            close(fd);
        };

        std::string target_file;
        target_file = model_catalog + "decoder.embed_tokens.weight";
        size_t read_size = embed_tokens_weight_row * embed_tokens_weight_column * sizeof(__half);
        mmap_load_tensor(target_file, embed_tokens_weight, read_size);
        
        target_file = model_catalog + "decoder.embed_positions.weight";
        read_size = embed_positions_weight_row * embed_positions_weight_column * sizeof(__half);
        load_tensor(target_file, embed_positions_weight, read_size);
        
        target_file = model_catalog + "decoder.final_layer_norm.weight";
        read_size = final_layer_norm_weight_column * sizeof(__half);
        load_tensor(target_file, final_layer_norm_weight, read_size);
        
        target_file = model_catalog + "decoder.final_layer_norm.bias";
        read_size = final_layer_norm_weight_column * sizeof(__half);
        load_tensor(target_file, final_layer_norm_bias, read_size);

        // load layer weights
        for (int i = 0; i < 32; i++){
            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            read_size = kqv_proj_row * kqv_proj_column * sizeof(__half);
            mmap_load_tensor(target_file, k_proj_weight[i], read_size);
            
            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            read_size = kqv_proj_row * kqv_proj_column * sizeof(__half);
            load_tensor(target_file, k_proj_bias[i], read_size);
            
            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            read_size = kqv_proj_row * kqv_proj_column * sizeof(__half);
            mmap_load_tensor(target_file, v_proj_weight[i], read_size);
            
            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            read_size = kqv_proj_row * kqv_proj_column * sizeof(__half);
            load_tensor(target_file, v_proj_weight[i], read_size);

            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            read_size = kqv_proj_row * kqv_proj_column * sizeof(__half);
            mmap_load_tensor(target_file, q_proj_weight[i], read_size);
            
            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            read_size = kqv_proj_row * kqv_proj_column * sizeof(__half);
            load_tensor(target_file, q_proj_bias[i], read_size);
            
            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".self_attn.out_proj.weight";
            read_size = kqv_proj_row * kqv_proj_column * sizeof(__half);
            mmap_load_tensor(target_file, out_proj_weight[i], read_size);
            
            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".self_attn.out_proj.weight";
            read_size = kqv_proj_row * kqv_proj_column * sizeof(__half);
            load_tensor(target_file, out_proj_bias[i], read_size);

            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".self_attn_layer_norm.weight";
            read_size = attn_layer_norm_column * sizeof(__half);
            load_tensor(target_file, attn_layer_norm_weight[i], read_size);
            
            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".self_attn_layer_norm.bias";
            read_size = attn_layer_norm_column * sizeof(__half);
            load_tensor(target_file, attn_layer_norm_bias[i], read_size);
            
            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".fc1.weight";
            read_size = fc1_weight_column + fc1_weight_row * sizeof(__half);
            mmap_load_tensor(target_file, fc1_weight[i], read_size);
            
            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".fc1.bias";
            read_size = fc1_weight_column * sizeof(__half);
            load_tensor(target_file, fc1_bias[i], read_size);

            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".fc2.weight";
            read_size = fc2_weight_column * fc2_weight_row * sizeof(__half);
            mmap_load_tensor(target_file, fc2_weight[i], read_size);
            
            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".fc2.bias";
            read_size = fc2_weight_column * sizeof(__half);
            load_tensor(target_file, fc2_bias[i], read_size);

            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".final_layer_norm.weight";
            read_size = layer_final_layer_norm_column * sizeof(__half);
            load_tensor(target_file, layers_final_layer_norm_weight[i], read_size);
            
            target_file = model_catalog + "decoder.layers." + std::to_string(i) + ".final_layer_norm.bias";
            read_size = layer_final_layer_norm_column * sizeof(__half);
            load_tensor(target_file, layers_final_layer_norm_bias[i], read_size);
        }

        
        __half h_data[10];
        cudaMemcpy(h_data, k_proj_weight[31], sizeof(__half) * 10, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 10; ++i) {
            float f_val = __half2float(h_data[i]);
            printf("Element[%d] = %f\n", i, f_val);
        }
        /**/

    }
    void load_tensor(std::string target_file, __half*& weight, size_t read_size){
        cudaMalloc(&weight, read_size);
        if (weight == nullptr) {
            throw std::runtime_error("Failed to allocate memory for " + target_file);
        }
        char* buffer = new char[read_size];
        std::ifstream fin(target_file, std::ios::binary);
        if (fin.is_open()) {
            fin.read(buffer, read_size);
        }
        else {
            delete[] buffer;
            throw std::runtime_error("Failed to open " + target_file);
        }
        /*
        std::cout << "First 10 bytes of the buffer: ";
        for(int i = 0; i < 10 && i < read_size; ++i) { // 确保不会超出读取大小
            printf("%02x ", static_cast<unsigned char>(buffer[i]));
        }
        std::cout << std::endl;
        */
        fin.close();
        cudaMemcpy(weight, buffer, read_size, cudaMemcpyHostToDevice);
        delete[] buffer;
    }

    ~opt_model() {
        // free all allocated memory
        if (embed_tokens_weight) cudaFree(embed_tokens_weight);
        if (embed_positions_weight) cudaFree(embed_positions_weight);
        if (final_layer_norm_weight) cudaFree(final_layer_norm_weight);
        if (final_layer_norm_bias) cudaFree(final_layer_norm_bias);

        // free layer weights
        for (int i = 0; i < num_layers; ++i) {
            if (k_proj_weight[i]) cudaFree(k_proj_weight[i]);
            if (k_proj_bias[i]) cudaFree(k_proj_bias[i]);
            if (v_proj_weight[i]) cudaFree(v_proj_weight[i]);
            if (v_proj_bias[i]) cudaFree(v_proj_bias[i]);
            if (q_proj_weight[i]) cudaFree(q_proj_weight[i]);
            if (q_proj_bias[i]) cudaFree(q_proj_bias[i]);
            if (out_proj_weight[i]) cudaFree(out_proj_weight[i]);
            if (out_proj_bias[i]) cudaFree(out_proj_bias[i]);
            if (attn_layer_norm_weight[i]) cudaFree(attn_layer_norm_weight[i]);
            if (attn_layer_norm_bias[i]) cudaFree(attn_layer_norm_bias[i]);
            if (fc1_weight[i]) cudaFree(fc1_weight[i]);
            if (fc1_bias[i]) cudaFree(fc1_bias[i]);
            if (fc2_weight[i]) cudaFree(fc2_weight[i]);
            if (fc2_bias[i]) cudaFree(fc2_bias[i]);
            if (layers_final_layer_norm_weight[i]) cudaFree(layers_final_layer_norm_weight[i]);
            if (layers_final_layer_norm_bias[i]) cudaFree(layers_final_layer_norm_bias[i]);
        }
    }

};