#include <iostream>
#include "src/llm/opt.cuh"
#include "src/llm/llama-3-8b.cuh"
#include "src/llm/TimeUtils.cuh"

int main(){
    CUDATimer cuTimer;
    cuTimer.start();
    std::string catalog = "./data/opt6.7b_tensor/";
    opt_model model = opt_model(catalog);
    //std::string catalog = "./data/llama-3-8b/";

    
    //llama_model model = llama_model("./data/llama-3-8b/");
    
    std::cout << "finished loading model" << std::endl;
    std::cout<<"loading model time: "<<cuTimer.stop()<<std::endl;
    
    getchar();
}