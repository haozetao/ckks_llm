#include <cuda_runtime.h>
#include <string>
#include <fstream> 
#include <cuda_bf16.h>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>


class llama_model {
public:
    __nv_bfloat16 *embed_tokens_weight = nullptr;

    __nv_bfloat16 *self_attn_k_proj_weight[32];
    __nv_bfloat16 *self_attn_o_proj_weight[32];
    __nv_bfloat16 *self_attn_q_proj_weight[32];
    __nv_bfloat16 *self_attn_v_proj_weight[32];
    __nv_bfloat16 *mlp_down_proj_weight[32];
    __nv_bfloat16 *mlp_gate_proj_weight[32];
    __nv_bfloat16 *mlp_up_proj_weight[32];
    __nv_bfloat16 *input_layernorm_weight[32];
    __nv_bfloat16 *post_attention_layernorm_weight[32];

    __nv_bfloat16 *lm_head_weight = nullptr;
    __nv_bfloat16 *model_norm_weight = nullptr;
    
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

    llama_model(std::string model_catalog) {
        for (int part = 1; part <= 4; ++part) {
            std::string fname = model_catalog + "model-0000" + std::to_string(part) + "-of-00004.safetensors";
            int fd = open(fname.c_str(), O_RDONLY);
            if (fd < 0) {
                std::cout << "Failed to open model file: " << fname << std::endl;
                continue;
            }

            // 读取 header_size
            uint64_t header_size = 0;
            ssize_t n = pread(fd, &header_size, sizeof(header_size), 0);
            if (n != sizeof(header_size)) {
                std::cout << "Failed to read header size from: " << fname << std::endl;
                close(fd);
                continue;
            }

            // 读取 header
            std::string header(header_size, '\0');
            n = pread(fd, &header[0], header_size, sizeof(uint64_t));
            if (n != static_cast<ssize_t>(header_size)) {
                std::cout << "Failed to read header from: " << fname << std::endl;
                close(fd);
                continue;
            }

            // 获取文件大小
            off_t file_size = lseek(fd, 0, SEEK_END);

            // mmap整个文件
            char* file_map = (char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (file_map == MAP_FAILED) {
                std::cout << "Failed to mmap file: " << fname << std::endl;
                close(fd);
                continue;
            }

            std::string targetKey = "embed_tokens.weight";
            int64_t offset1, offset2;
            size_t read_size;
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, embed_tokens_weight, sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = "lm_head.weight";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, lm_head_weight, sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = "model.norm.weight";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, model_norm_weight, sizeof(uint64_t) + header_size + offset1, read_size);
            }

            for (int i = 0; i < num_layers; ++i) {
                std::string layer_prefix = "model.layers." + std::to_string(i) + ".";

                targetKey = layer_prefix + "self_attn.k_proj.weight";
                if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                    read_size = offset2 - offset1;
                    load_tensor(file_map, self_attn_k_proj_weight[i], sizeof(uint64_t) + header_size + offset1, read_size);
                }

                targetKey = layer_prefix + "self_attn.o_proj.weight";
                if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                    read_size = offset2 - offset1;
                    load_tensor(file_map, self_attn_o_proj_weight[i], sizeof(uint64_t) + header_size + offset1, read_size);
                }

                targetKey = layer_prefix + "self_attn.q_proj.weight";
                if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                    read_size = offset2 - offset1;
                    load_tensor(file_map, self_attn_q_proj_weight[i], sizeof(uint64_t) + header_size + offset1, read_size);
                }

                targetKey = layer_prefix + "self_attn.v_proj.weight";
                if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                    read_size = offset2 - offset1;
                    load_tensor(file_map, self_attn_v_proj_weight[i], sizeof(uint64_t) + header_size + offset1, read_size);
                }

                targetKey = layer_prefix + "mlp.down_proj.weight";
                if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                    read_size = offset2 - offset1;
                    load_tensor(file_map, mlp_down_proj_weight[i], sizeof(uint64_t) + header_size + offset1, read_size);
                }

                targetKey = layer_prefix + "mlp.gate_proj.weight";
                if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                    read_size = offset2 - offset1;
                    load_tensor(file_map, mlp_gate_proj_weight[i], sizeof(uint64_t) + header_size + offset1, read_size);
                }

                targetKey = layer_prefix + "mlp.up_proj.weight";
                if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                    read_size = offset2 - offset1;
                    load_tensor(file_map, mlp_up_proj_weight[i], sizeof(uint64_t) + header_size + offset1, read_size);
                }

                targetKey = layer_prefix + "input_layernorm.weight";
                if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                    read_size = offset2 - offset1;
                    load_tensor(file_map, input_layernorm_weight[i], sizeof(uint64_t) + header_size + offset1, read_size);
                }

                targetKey = layer_prefix + "post_attention_layernorm.weight";
                if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                    read_size = offset2 - offset1;
                    load_tensor(file_map, post_attention_layernorm_weight[i], sizeof(uint64_t) + header_size + offset1, read_size);
                }
            }

            munmap(file_map, file_size);
            close(fd);
        }


        // Print the first 10 elements
        __nv_bfloat16 h_data[10];
        cudaMemcpy(h_data, self_attn_k_proj_weight[31], sizeof(__nv_bfloat16) * 10, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 10; ++i) {
            float f_val = __bfloat162float(h_data[i]);
            printf("Element[%d] = %e\n", i, f_val);
        }
        /**/
        
        // Print the first 10 elements of embed_tokens_weight in hex format
        /*
        __nv_bfloat16 h_data[10];
        cudaMemcpy(h_data, embed_tokens_weight, sizeof(__nv_bfloat16) * 10, cudaMemcpyDeviceToHost);
        unsigned char* byte_ptr = reinterpret_cast<unsigned char*>(h_data);
        for (int i = 0; i < 10; ++i) {
            printf("Element[%d] = %02x %02x\n", i, byte_ptr[i * 2], byte_ptr[i * 2 + 1]);
        }
        */

    }
    void load_tensor(char* file_map, __nv_bfloat16*& weight, size_t offset, size_t read_size) {
        cudaMalloc(&weight, read_size);
        cudaDeviceSynchronize();
        if (weight == nullptr) {
            throw std::runtime_error("Failed to allocate memory for file");
        }
        // copy data from file_map to weight
        cudaMemcpy(weight, file_map + offset, read_size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    std::string trim(const std::string& s) {
        size_t first = s.find_first_not_of(" \t\n\r\"");
        if (first == std::string::npos) return "";
        size_t last = s.find_last_not_of(" \t\n\r\"");
        return s.substr(first, last - first + 1);
    }

    // get two offsets from a string input
    bool parseKeyFromString(const std::string& input, const std::string& targetKey,
                        int64_t& offset1Out, int64_t& offset2Out) {
        // search for the target key
        size_t keyPos = input.find(targetKey);
        if (keyPos == std::string::npos) return false;
        // printf("find targetKey: %s, keyPos: %zu\n", targetKey.c_str(), keyPos);

        // get data_offsets
        std::string offset_string = "\"data_offsets\":[";
        size_t dataOffsetsPos = input.find(offset_string, keyPos);
        if (dataOffsetsPos == std::string::npos) return false;
        dataOffsetsPos += offset_string.length(); // pass "data_offsets":[

        size_t commaPos = input.find(",", dataOffsetsPos);
        size_t closePos = input.find("]", dataOffsetsPos);
        if (commaPos == std::string::npos || closePos == std::string::npos) return false;

        std::string val1 = trim(input.substr(dataOffsetsPos, commaPos - dataOffsetsPos));
        std::string val2 = trim(input.substr(commaPos + 1, closePos - commaPos - 1));
        // printf("val1: %s, val2: %s\n", val1.c_str(), val2.c_str());

        try {
            offset1Out = std::stoll(val1);
            offset2Out = std::stoll(val2);
            return true;
        } catch (...) {
            return false;
        }
    }

    ~llama_model() {
        
    }

};