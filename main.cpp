#include "arcfacem.h"
#include<iostream>
#include<vector>
#include "retinaface.h"
#include "ilogger.hpp"
using namespace std;
int main()
{

	//0 .build detector
	retinaface rf;
	rf.Init("../retina_mnet.engine");
	//1.build recognition
	arcfacem af;
	//af.WTSToEngine("../arcface-mobilefacenet.wts","../arcface-mobilefacenet.engine");
	af.Init("../arcface-mobilefacenet.engine");
	
	//2.load face_lib (to do add faiss)
	float *p1=new float[128];
    af.Inference_file("/home/cookoo/face_lib/cl.jpg",p1);
    float *p2=new float[128];
    af.Inference_file("/home/cookoo/face_lib/liubin",p2);

	auto libs = iLogger::find_files("/home/cookoo/face_lib");

    auto merge_image = iLogger::mergeDiffPic(libs,1,"out");
    cv::resize(merge_image, merge_image, cv::Size(200, 600));


	auto file_path    = "/home/cookoo/images/image.jpg";
    std::string flag_path = "/home/cookoo/images/flag.txt";


	
	// cv::namedWindow("ID RECOGNITION",cv::WINDOW_AUTOSIZE);
	float *p3=new float[128];
	while(1)
	{
		 // sleep(1);
		ifstream f(flag_path.c_str());
		
		if(f.good()){
				cv::Mat face;
				cv::Mat result;
				std::cout<<"1 "<<std::endl;
				face = rf.Inference_file(file_path,result);
				std::cout<<"2 "<<std::endl;
				p3 = af.Inference_image(face);
				float ret = af.Compare(p1,p3);
				float ret1 = af.Compare(p2,p3);
				std::cout<<"compare result:"<<ret<<" "<<ret1<<std::endl;
				cv::resize(result, result, cv::Size(800, 600));
				std::vector<cv::Mat> imgs;
				imgs.push_back(result);
				imgs.push_back(merge_image);
				cv::Mat final_result;
				hconcat(imgs,final_result);
				cv::imshow("face recognition", final_result);
				cv::waitKey(1); //10ms
				remove(file_path);
				remove(flag_path.c_str());
		}



	}
	delete[] p1;
	delete[] p3;
	delete[] p3;
	rf.UnInit();
	af.UnInit();
	return 0;
}
