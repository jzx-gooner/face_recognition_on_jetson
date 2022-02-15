# sgai-face-recognition
一个简单的人脸检测系统
* 两个类：
* 一个retinaface类： 用于人脸检测：人脸检测->最大分数人脸->人脸对齐
* 一个arcface类：[128*1]的vector用于人脸距离比较
# todo:
* 检索加速
* 美化输出

# how to
mkdir build && cd build &&cmake .. && make 

./face_recognition
