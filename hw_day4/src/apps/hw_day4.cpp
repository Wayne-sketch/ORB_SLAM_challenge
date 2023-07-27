#include "ORB/global_defination/global_defination.h"
#include "ORB/ORBFeature.hpp"

using namespace ORB;

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;

    std::string config_path = WORK_SPACE_PATH + "/config/camera_para.yaml";
    std::string image_path = WORK_SPACE_PATH + "/image/distorted.png";

    //利用yaml文件把参数读进来
    std::shared_ptr<ORBFeature> orb_feature_ptr = std::make_shared<ORBFeature>(image_path, config_path);

    //存储提取到的关键点
    std::vector<cv::KeyPoint> vKeypoints;
    //存储描述子
    cv::Mat descriptor;
    //调用ORB_SLAM2中的特征提取函数
    orb_feature_ptr->ExtractORB(vKeypoints, descriptor);

    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    LOG(INFO) << "vKeypoints size " << vKeypoints.size();
    LOG(INFO) << "descriptor size " << descriptor.rows;

    cv::Mat outImage;
    cv::drawKeypoints(image, vKeypoints, outImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("这是ORB-SLAM2提取的特征点", outImage);

    //调用OpenCV特征提取函数
    orb_feature_ptr->ExtractORB();
    cv::waitKey();
    return 0;
}
