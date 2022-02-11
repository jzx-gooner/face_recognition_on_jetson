#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "calibrator.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


//#define USE_FP16  // comment out this if want to use FP32
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1
#define CONF_THRESH 0.75
#define IOU_THRESH 0.4

using namespace nvinfer1;

class retinaface {
  public:

    retinaface();
    ~retinaface();
    int WTSToEngine(std::string wtsfile,std::string enginefile);
    int Init(std::string enginefile);
    void UnInit();
    cv::Mat Inference_file(std::string imagefile);
    ICudaEngine* createEngine(const std::string wtsfile,unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);
    // void APIToModel(const std::string wtsfile,unsigned int maxBatchSize, IHostMemory** modelStream);
    void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
    
 private:

    IExecutionContext* context;
    ICudaEngine* engine;
    IRuntime* runtime;
    
    ILayer* conv_bn(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int s, float leaky);
    ILayer* conv_bn_no_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int s);
    ILayer* conv_bn1X1(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int s, float leaky ) ;
    ILayer* conv_dw(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inp, int oup, int s, float leaky);
    IActivationLayer* ssh(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup);
    ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);
    // stuff we know about the network and the input/output blobs

    Logger gLogger;
    };