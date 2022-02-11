#include "arcfacem.h"
#include<iostream>
#include<vector>
#include "retinaface.h"
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


	auto file_path    = "/home/cookoo/images/image.jpg";
    std::string flag_path = "/home/cookoo/images/flag.txt";







	
	

	float *p3=new float[128];
	while(1)
	{
		 // sleep(1);
		ifstream f(flag_path.c_str());
		
		if(f.good()){
				cv::Mat face;
				face = rf.Inference_file("/home/cookoo/images/image.jpg");
				
				p3 = af.Inference_image(face);
				float ret = af.Compare(p1,p3);
				float ret1 = af.Compare(p2,p3);
				
				std::cout<<"compare result:"<<ret<<" "<<ret1<<std::endl;
				remove(file_path);
				remove(flag_path.c_str());
		}



	}
	delete[] p3;
	rf.UnInit();
	af.UnInit();
	return 0;
}
