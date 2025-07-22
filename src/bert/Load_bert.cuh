#include <cuda_runtime.h>
#include <string>
#include <fstream> 
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>


class Bert_model_weights {
public:
    float *embeddings_LayerNorm_beta = nullptr; // [768]
    float *embeddings_LayerNorm_gamma = nullptr; // [768]
    float *embeddings_position_embeddings_weight = nullptr; // [512,768]
    float *embeddings_token_type_embeddings_weight = nullptr; // [2,768]
    float *embeddings_word_embeddings_weight = nullptr; // [30522,768]
    
    // layer
    float *attention_output_LayerNorm_beta[12];
    float *attention_output_LayerNorm_gamma[12];
    float *attention_output_dense_bias[12];
    float *attention_output_dense_weight[12];
    float *attention_self_key_bias[12];
    float *attention_self_key_weight[12];
    float *attention_self_query_bias[12];
    float *attention_self_query_weight[12];
    float *attention_self_value_bias[12];
    float *attention_self_value_weight[12];
    float *intermediate_dense_bias[12];
    float *intermediate_dense_weight[12];
    float *output_LayerNorm_beta[12];
    float *output_LayerNorm_gamma[12];
    float *output_dense_bias[12];
    float *output_dense_weight[12];

    float *pooler_dense_bias = nullptr;
    float *pooler_dense_weight = nullptr;
    float *cls_predictions_bias = nullptr;
    float *cls_predictions_transform_LayerNorm_beta = nullptr;
    float *cls_predictions_transform_LayerNorm_gamma = nullptr;
    float *cls_predictions_transform_dense_bias = nullptr;
    float *cls_predictions_transform_dense_weight = nullptr;
    float *cls_seq_relationship_bias = nullptr;
    float *cls_seq_relationship_weight = nullptr;

    int num_layers = 12;

    Bert_model_weights(std::string model_catalog) {
        std::string fname = model_catalog + "model.safetensors";
        int fd = open(fname.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cout << "Failed to open model file: " << fname << std::endl;
            return;
        }

        // 读取 header_size
        uint64_t header_size = 0;
        ssize_t n = pread(fd, &header_size, sizeof(header_size), 0);
        if (n != sizeof(header_size)) {
            std::cout << "Failed to read header size from: " << fname << std::endl;
            close(fd);
            return;
        }

        // 读取 header
        std::string header(header_size, '\0');
        n = pread(fd, &header[0], header_size, sizeof(uint64_t));
        if (n != static_cast<ssize_t>(header_size)) {
            std::cout << "Failed to read header from: " << fname << std::endl;
            close(fd);
            return;
        }

        // 获取文件大小
        off_t file_size = lseek(fd, 0, SEEK_END);

        // mmap整个文件
        char* file_map = (char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (file_map == MAP_FAILED) {
            std::cout << "Failed to mmap file: " << fname << std::endl;
            close(fd);
            return;
        }

        std::string targetKey = "bert.embeddings.LayerNorm.beta";
        int64_t offset1, offset2;
        size_t read_size;
        if (parseKeyFromString(header, targetKey, offset1, offset2)) {
            read_size = offset2 - offset1;
            load_tensor(file_map, embeddings_LayerNorm_beta, sizeof(uint64_t) + header_size + offset1, read_size);
        }

        targetKey = "bert.embeddings.LayerNorm.gamma";
        if (parseKeyFromString(header, targetKey, offset1, offset2)) {
            read_size = offset2 - offset1;
            load_tensor(file_map, embeddings_LayerNorm_gamma, sizeof(uint64_t) + header_size + offset1, read_size);
        }

        targetKey = "bert.embeddings.position_embeddings.weight";
        if (parseKeyFromString(header, targetKey, offset1, offset2)) {
            read_size = offset2 - offset1;
            load_tensor(file_map, embeddings_position_embeddings_weight, sizeof(uint64_t) + header_size + offset1, read_size);
        }

        targetKey = "bert.embeddings.token_type_embeddings.weight";
        if (parseKeyFromString(header, targetKey, offset1, offset2)) {
            read_size = offset2 - offset1;
            load_tensor(file_map, embeddings_token_type_embeddings_weight, sizeof(uint64_t) + header_size + offset1, read_size);
        }

        targetKey = "bert.embeddings.word_embeddings.weight";
        if (parseKeyFromString(header, targetKey, offset1, offset2)) {
            read_size = offset2 - offset1;
            load_tensor(file_map, embeddings_word_embeddings_weight, sizeof(uint64_t) + header_size + offset1, read_size);
        }

        targetKey = "bert.pooler.dense.bias";
        if (parseKeyFromString(header, targetKey, offset1, offset2)) {
            read_size = offset2 - offset1;
            load_tensor(file_map, pooler_dense_bias, sizeof(uint64_t) + header_size + offset1, read_size);
        }

        targetKey = "bert.pooler.dense.weight";
        if (parseKeyFromString(header, targetKey, offset1, offset2)) {
            read_size = offset2 - offset1;
            load_tensor(file_map, pooler_dense_weight, sizeof(uint64_t) + header_size + offset1, read_size);
        }

        targetKey = "cls.predictions.bias";
        if (parseKeyFromString(header, targetKey, offset1, offset2)) {
            read_size = offset2 - offset1;
            load_tensor(file_map, cls_predictions_bias, sizeof(uint64_t) + header_size + offset1, read_size);
        }

        targetKey = "cls.predictions.transform.LayerNorm.beta";
        if (parseKeyFromString(header, targetKey, offset1, offset2)) {
            read_size = offset2 - offset1;
            load_tensor(file_map, cls_predictions_transform_LayerNorm_beta, sizeof(uint64_t) + header_size + offset1, read_size);
        }

        targetKey = "cls.predictions.transform.LayerNorm.gamma";
        if (parseKeyFromString(header, targetKey, offset1, offset2)) {
            read_size = offset2 - offset1;
            load_tensor(file_map, cls_predictions_transform_LayerNorm_gamma, sizeof(uint64_t) + header_size + offset1, read_size);
        }

        targetKey = "cls.predictions.transform.dense.bias";
        if (parseKeyFromString(header, targetKey, offset1, offset2)) {
            read_size = offset2 - offset1;
            load_tensor(file_map, cls_predictions_transform_dense_bias, sizeof(uint64_t) + header_size + offset1, read_size);
        }

        targetKey = "cls.predictions.transform.dense.weight";
        if (parseKeyFromString(header, targetKey, offset1, offset2)) {
            read_size = offset2 - offset1;
            load_tensor(file_map, cls_predictions_transform_dense_weight, sizeof(uint64_t) + header_size + offset1, read_size);
        }

        targetKey = "cls.seq_relationship.bias";
        if (parseKeyFromString(header, targetKey, offset1, offset2)) {
            read_size = offset2 - offset1;
            load_tensor(file_map, cls_seq_relationship_bias, sizeof(uint64_t) + header_size + offset1, read_size);
        }

        targetKey = "cls.seq_relationship.weight";
        if (parseKeyFromString(header, targetKey, offset1, offset2)) {
            read_size = offset2 - offset1;
            load_tensor(file_map, cls_seq_relationship_weight, sizeof(uint64_t) + header_size + offset1, read_size);
        }

        for (int i = 0; i < num_layers; ++i) {
            std::string layer_prefix = "bert.encoder.layer." + std::to_string(i) + ".";

            targetKey = layer_prefix + "attention.output.LayerNorm.beta";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, attention_output_LayerNorm_beta[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = layer_prefix + "attention.output.LayerNorm.gamma";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, attention_output_LayerNorm_gamma[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = layer_prefix + "attention.output.dense.bias";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, attention_output_dense_bias[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = layer_prefix + "attention.output.dense.weight";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, attention_output_dense_weight[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = layer_prefix + "attention.self.key.bias";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, attention_self_key_bias[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = layer_prefix + "attention.self.key.weight";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, attention_self_key_weight[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = layer_prefix + "attention.self.query.bias";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, attention_self_query_bias[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = layer_prefix + "attention.self.query.weight";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, attention_self_query_weight[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = layer_prefix + "attention.self.value.bias";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, attention_self_value_bias[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = layer_prefix + "attention.self.value.weight";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, attention_self_value_weight[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = layer_prefix + "intermediate.dense.bias";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, intermediate_dense_bias[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = layer_prefix + "intermediate.dense.weight";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, intermediate_dense_weight[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = layer_prefix + "output.LayerNorm.beta";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, output_LayerNorm_beta[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = layer_prefix + "output.LayerNorm.gamma";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, output_LayerNorm_gamma[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = layer_prefix + "output.dense.bias";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, output_dense_bias[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }

            targetKey = layer_prefix + "output.dense.weight";
            if (parseKeyFromString(header, targetKey, offset1, offset2)) {
                read_size = offset2 - offset1;
                load_tensor(file_map, output_dense_weight[i], sizeof(uint64_t) + header_size + offset1, read_size);
            }
        }

        munmap(file_map, file_size);
        close(fd);


        // Print the first 10 elements
        float h_data[10];
        cudaMemcpy(h_data, attention_output_LayerNorm_gamma[0], sizeof(float) * 10, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 10; ++i) {
            float f_val = float(h_data[i]);
            printf("Element[%d] = %e\n", i, f_val);
        }
        /**/
        
        // Print the first 10 elements of embed_tokens_weight in hex format
        /*
        float h_data[10];
        cudaMemcpy(h_data, embed_tokens_weight, sizeof(float) * 10, cudaMemcpyDeviceToHost);
        unsigned char* byte_ptr = reinterpret_cast<unsigned char*>(h_data);
        for (int i = 0; i < 10; ++i) {
            printf("Element[%d] = %02x %02x\n", i, byte_ptr[i * 2], byte_ptr[i * 2 + 1]);
        }
        */

    }
    void load_tensor(char* file_map, float*& weight, size_t offset, size_t read_size) {
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

    ~Bert_model_weights() {
        
    }

};