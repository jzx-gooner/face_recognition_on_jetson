# simple-face-recognition
一个简单的人脸检测系统，可在安装tensorrt的设备上部署。测试设备有 jetson nx 和nano
![avatar](./show.png)
* 两个类：
* 一个retinaface类
  * 人脸检测
  * 人脸对齐
* 一个arcface类
  * [128*1]的vector用于人脸距离比较
  * 快速检索

# how to
mkdir build && cd build &&cmake .. && make 

./face_recognition
