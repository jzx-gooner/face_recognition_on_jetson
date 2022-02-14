#include "arcfacem.h"
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>

arcfacem::arcfacem()
{
}
arcfacem::~arcfacem()
{
}

std::map<std::string, Weights> arcfacem::loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* arcfacem::addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + "_gamma"].values;
    float *beta = (float*)weightMap[lname + "_beta"].values;
    float *mean = (float*)weightMap[lname + "_moving_mean"].values;
    float *var = (float*)weightMap[lname + "_moving_var"].values;
    int len = weightMap[lname + "_moving_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* arcfacem::addPRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname) {
	float *gamma = (float*)weightMap[lname + "_gamma"].values;
	int len = weightMap[lname + "_gamma"].count;

	float *scval_1 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	float *scval_2 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		scval_1[i] = -1.0;
		scval_2[i] = -gamma[i];
	}
	Weights scale_1{ DataType::kFLOAT, scval_1, len };
	Weights scale_2{ DataType::kFLOAT, scval_2, len };

	float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		shval[i] = 0.0;
	}
	Weights shift{ DataType::kFLOAT, shval, len };

	float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		pval[i] = 1.0;
	}
	Weights power{ DataType::kFLOAT, pval, len };

	auto relu1 = network->addActivation(input, ActivationType::kRELU);
	assert(relu1);
	IScaleLayer* scale1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale_1, power);
	assert(scale1);
	auto relu2 = network->addActivation(*scale1->getOutput(0), ActivationType::kRELU);
	assert(relu2);
	IScaleLayer* scale2 = network->addScale(*relu2->getOutput(0), ScaleMode::kCHANNEL, shift, scale_2, power);
	assert(scale2);
	IElementWiseLayer* ew1 = network->addElementWise(*relu1->getOutput(0), *scale2->getOutput(0), ElementWiseOperation::kSUM);
	assert(ew1);
	return ew1;
}

ILayer* arcfacem::conv_bn_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int k, int p, int s, int groups) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, oup, DimsHW{k, k}, weightMap[lname + "_conv2d_weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(groups);
    auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_batchnorm", 1e-3);
    assert(bn1);
    auto act1 = addPRelu(network, weightMap, *bn1->getOutput(0), lname + "_relu");
    assert(act1);
    return act1;
}

ILayer* arcfacem::conv_bn(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int k, int p, int s, int groups) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, oup, DimsHW{k, k}, weightMap[lname + "_conv2d_weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(groups);
    auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_batchnorm", 1e-3);
    assert(bn1);
    return bn1;
}

ILayer* arcfacem::DepthWise(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inp, int oup, int groups, int s) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, groups, DimsHW{1, 1}, weightMap[lname + "_conv_sep_conv2d_weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setPaddingNd(DimsHW{0, 0});
    conv1->setNbGroups(1);
    auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_conv_sep_batchnorm", 1e-3);
    assert(bn1);
    auto act1 = addPRelu(network, weightMap, *bn1->getOutput(0), lname + "_conv_sep_relu");
    assert(act1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*act1->getOutput(0), groups, DimsHW{3, 3}, weightMap[lname + "_conv_dw_conv2d_weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{s, s});
    conv2->setPaddingNd(DimsHW{1, 1});
    conv2->setNbGroups(groups);
    auto bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "_conv_dw_batchnorm", 1e-3);
    assert(bn2);
    auto act2 = addPRelu(network, weightMap, *bn2->getOutput(0), lname + "_conv_dw_relu");
    assert(act2);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*act2->getOutput(0), oup, DimsHW{1, 1}, weightMap[lname + "_conv_proj_conv2d_weight"], emptywts);
    assert(conv3);
    conv3->setStrideNd(DimsHW{1, 1});
    conv3->setPaddingNd(DimsHW{0, 0});
    conv3->setNbGroups(1);
    auto bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "_conv_proj_batchnorm", 1e-3);
    assert(bn3);
    return bn3;
}


ILayer* arcfacem::DWResidual(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inp, int oup, int groups, int s) {

    auto dw1 = DepthWise(network, weightMap, input, lname, inp, oup, groups, s);
    IElementWiseLayer* ew1;
    ew1 = network->addElementWise(input, *dw1->getOutput(0), ElementWiseOperation::kSUM);
    assert(ew1);
    return ew1;
}


// Creat the engine using only the API and not any parser.
ICudaEngine* arcfacem::createEngine(const std::string wtsfile,unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wtsfile);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    auto conv_1 = conv_bn_relu(network, weightMap, *data, "conv_1", 64, 3, 1, 2);
    auto conv_2_dw = conv_bn_relu(network, weightMap, *conv_1->getOutput(0), "conv_2_dw", 64, 3, 1, 1, 64);
    auto conv_23 = DepthWise(network, weightMap, *conv_2_dw->getOutput(0), "dconv_23", 64, 64, 128, 2);
    auto res_3_block0 = DWResidual(network, weightMap, *conv_23->getOutput(0), "res_3_block0", 64, 64, 128, 1);
    auto res_3_block1 = DWResidual(network, weightMap, *res_3_block0->getOutput(0), "res_3_block1", 64, 64, 128, 1);
    auto res_3_block2 = DWResidual(network, weightMap, *res_3_block1->getOutput(0), "res_3_block2", 64, 64, 128, 1);
    auto res_3_block3 = DWResidual(network, weightMap, *res_3_block2->getOutput(0), "res_3_block3", 64, 64, 128, 1);
    auto conv_34 = DepthWise(network, weightMap, *res_3_block3->getOutput(0), "dconv_34", 64, 128, 256, 2);
    auto res_4_block0 = DWResidual(network, weightMap, *conv_34->getOutput(0), "res_4_block0", 128, 128, 256, 1);
    auto res_4_block1 = DWResidual(network, weightMap, *res_4_block0->getOutput(0), "res_4_block1", 128, 128, 256, 1);
    auto res_4_block2 = DWResidual(network, weightMap, *res_4_block1->getOutput(0), "res_4_block2", 128, 128, 256, 1);
    auto res_4_block3 = DWResidual(network, weightMap, *res_4_block2->getOutput(0), "res_4_block3", 128, 128, 256, 1);
    auto res_4_block4 = DWResidual(network, weightMap, *res_4_block3->getOutput(0), "res_4_block4", 128, 128, 256, 1);
    auto res_4_block5 = DWResidual(network, weightMap, *res_4_block4->getOutput(0), "res_4_block5", 128, 128, 256, 1);
    auto conv_45 = DepthWise(network, weightMap, *res_4_block5->getOutput(0), "dconv_45", 128, 128, 512, 2);
    auto res_5_block0 = DWResidual(network, weightMap, *conv_45->getOutput(0), "res_5_block0", 128, 128, 256, 1);
    auto res_5_block1 = DWResidual(network, weightMap, *res_5_block0->getOutput(0), "res_5_block1", 128, 128, 256, 1);
    auto conv_6_sep = conv_bn_relu(network, weightMap, *res_5_block1->getOutput(0), "conv_6sep", 512, 1, 0, 1);
    auto conv_6dw7_7 = conv_bn(network, weightMap, *conv_6_sep->getOutput(0), "conv_6dw7_7", 512, 7, 0, 1, 512);
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*conv_6dw7_7->getOutput(0), 128, weightMap["fc1_weight"], weightMap["pre_fc1_bias"]);
    assert(fc1);
    auto bn1 = addBatchNorm2d(network, weightMap, *fc1->getOutput(0), "fc1", 2e-5);
    assert(bn1);
    bn1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*bn1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void arcfacem::APIToModel(const std::string wtsfile,unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(arcfacem::gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(wtsfile,maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void arcfacem::doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int arcfacem::read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
                strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}
int arcfacem::WTSToEngine(const std::string wtsfile,std::string enginefile)
{
        IHostMemory* modelStream{nullptr};
        APIToModel(wtsfile,BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(enginefile, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
}

int arcfacem::Init(std::string enginefile)
{
    std::cout<<"loading model..."<<std::endl;
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char* trtModelStream{ nullptr };
    size_t size{ 0 };
    std::ifstream file(enginefile, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }else{
 std::cerr << "file is bad!" << std::endl;
 return -1;
    }
    arcfacem::runtime = createInferRuntime(arcfacem::gLogger);
    assert(arcfacem::runtime != nullptr);
    arcfacem::engine = arcfacem::runtime->deserializeCudaEngine(trtModelStream, size);
    assert(arcfacem::engine != nullptr);
    arcfacem::context = arcfacem::engine->createExecutionContext();
    assert(arcfacem::context != nullptr);
    delete[] trtModelStream;
    //engine->destroy();
    //runtime->destroy();
    std::cout<<"model loaded!"<< std::endl;
    return 0;
}
void arcfacem::UnInit()
{
    arcfacem::context->destroy();
    arcfacem::engine->destroy();
    arcfacem::runtime->destroy();
   
}
void arcfacem::Inference_file(std::string imagefile,float *score)
{
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    cv::Mat img = cv::imread(imagefile);
    if(!img.data)
    {
        std::cout<<"data is NULL"<<std::endl;
        return;
    }
    // prepare input data ---------------------------
    //normilize the dataset
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
        data[i + INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
    }
    // std::cout<<"inference..."<<std::endl;
    doInference(*context, data, prob, BATCH_SIZE);
    // std::cout<<"inference over..."<<std::endl;
    // std::cout<<sizeof(prob)/sizeof(float)<<std::endl;
    std::copy(prob,prob+128,score);
}

cv::Mat arcfacem::Inference_image(cv::Mat img)
{
    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
        data[i + INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
    }
    doInference(*context, data, prob, BATCH_SIZE);
    cv::Mat result(1,128,CV_64F,prob);
    return result;
}

float arcfacem::Compare(float* prob1, float* prob2)
{
    float p1[128];
    for(int i=0;i<128;i++)
    {
    p1[i]=prob1[i];
    }
    // std::cout<<prob1<<std::endl;
    cv::Mat out1(128, 1, CV_32FC1, p1);
    cv::Mat out_norm1;
    cv::normalize(out1, out_norm1);

    float p2[128];
    for(int i=0;i<128;i++)
    {
    p2[i]=prob2[i];
    }
    cv::Mat out2(1,128, CV_32FC1, p2);
    cv::Mat out_norm2;
    cv::normalize(out2, out_norm2);
    cv::Mat res = out_norm2 * out_norm1;
    float result= *(float*)res.data;
    return result;
}
