#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

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
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1  // currently, only support BATCH=1

using namespace nvinfer1;

class arcfacem {
  public:
    arcfacem();
    ~arcfacem();
    int WTSToEngine(std::string wtsfile,std::string enginefile);
    int Init(std::string enginefile);
    void Inference_file(std::string imagefile,float *score);
    float * Inference_image(cv::Mat imagefile);
    float Compare(float *prob1,float *prob2);
    ICudaEngine* createEngine(const std::string wtsfile,unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);
    void APIToModel(const std::string wtsfile,unsigned int maxBatchSize, IHostMemory** modelStream);
    void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
    void UnInit();
    private:
    IExecutionContext* context;
    ICudaEngine* engine;
    IRuntime* runtime;
    IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);
    std::map<std::string, Weights> loadWeights(const std::string file);
    ILayer* conv_bn(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int k = 3, int p = 1, int s = 1, int groups=1);
    ILayer* conv_bn_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int k = 3, int p = 1, int s = 2, int groups=1);
    ILayer* addPRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname);
    ILayer* DepthWise(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inp, int oup, int groups, int s);
    ILayer* DWResidual(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inp, int oup, int groups, int s);
    int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);
    // stuff we know about the network and the input/output blobs
    static const int INPUT_H = 112;
    static const int INPUT_W = 112;
    static const int OUTPUT_SIZE = 128;
    const char* INPUT_BLOB_NAME = "data";   
    const char* OUTPUT_BLOB_NAME = "prob";
    Logger gLogger;
    };
