#include "arcfacem.h"
#include<iostream>
#include<vector>
#include "retinaface.h"
#include "ilogger.hpp"
using namespace std;
int main()
{
	//deepstream set 
	auto file_path    = "/home/cookoo/images/image.jpg";
    std::string flag_path = "/home/cookoo/images/flag.txt";

	//0 .build detector
	retinaface rf;
	rf.Init("../retina_mnet.engine");
	//1.build recognition
	arcfacem af;
	//af.WTSToEngine("../arcface-mobilefacenet.wts","../arcface-mobilefacenet.engine");
	af.Init("../arcface-mobilefacenet.engine");
	
	//2.load face_lib (to do add faiss)
	cv::Mat_<float> face_lib(0,128);
	std::vector<std::string> face_name;
	auto libs_path = iLogger::find_files("../face_lib/face");
	for(auto& file:libs_path){
		auto file_name = iLogger::file_name(file,false);
		cv::Mat detect_face,detect_result;
		if(rf.Inference_file(file,detect_face,detect_result,true)){
			cv::Mat face_feature = af.Inference_image(detect_face);	
			face_lib.push_back(face_feature);
			face_name.push_back(file_name);
		}else{
			std::cout<<"can not find face"<<std::endl;
		}
	}
	std::cout<<"face lib size : "<<face_lib.size()<<std::endl;
	auto face_library = std::make_tuple(face_lib,face_name);
	auto merge_path = iLogger::find_files("../face_lib/draw");
    auto merge_image = iLogger::mergeDiffPic(merge_path,1,"out");
    cv::resize(merge_image, merge_image, cv::Size(200, 600));

	
	// cv::namedWindow("ID RECOGNITION",cv::WINDOW_AUTOSIZE);
	while(1)
	{
		 // sleep(1);
		ifstream f(flag_path.c_str());
		
		if(f.good()){
				cv::Mat face;
				cv::Mat result;
				if(rf.Inference_file(file_path,face,result,false)){
					
							cv::Mat current_face = af.Inference_image(face);
							std::cout<<"!!!!!!!!!!!!!!!!!!!!"<<std::endl;
							std::cout<<(get<0>(face_library).size())<<std::endl;//[128,5]
							std::cout<<current_face.size()<<std::endl;//[128,1]
							std::cout<<current_face<<std::endl;											
							auto scores = cv::Mat(get<0>(face_library)*current_face.t());
							//normalize
    						cv::normalize(scores, scores);
							float* pscore = scores.ptr<float>(0);
							int label = std::max_element(pscore,pscore+scores.rows)-pscore;
							float match_score = max(0.0f,pscore[label]);
							for(auto& i : face_name){
								std::cout<<"face label : " <<i<<std::endl;
							}
							std::cout<<"match label : " << label<<"match score : "<<match_score<<std::endl;
							std::cout<< "name : "<< get<1>(face_library)[label].c_str()<<std::endl;
							
							std::string names;
							if(match_score>0.8){
								names = iLogger::format("%s[%.3f]",get<1>(face_library)[label-1].c_str(),match_score);
							}else{
								names = iLogger::format("%s[%.3f]","UNKOWN",0);
							}
							std::cout<<names<<std::endl;
			
							cv::resize(result, result, cv::Size(800, 600));
							std::vector<cv::Mat> imgs;
							imgs.push_back(result);
							imgs.push_back(merge_image);
							cv::Mat final_result;
							hconcat(imgs,final_result);
							cv::putText(final_result,names,cv::Point(80,80),0,1,cv::Scalar(0,255,0),1,16);
							cv::imshow("face recognition", final_result);
							cv::waitKey(1); //10ms
							remove(file_path);
							remove(flag_path.c_str());
				}
				else{
					remove(file_path);
					remove(flag_path.c_str());
					std::cout<<"no face "<<std::endl;
				}
		}

	}
	
	rf.UnInit();
	af.UnInit();
	return 0;
}
