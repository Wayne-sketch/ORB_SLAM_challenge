#include<iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include<opencv2/imgproc.hpp>
#include "gms_matcher.h"

using namespace std;
using namespace cv;

#define ORB_N_FEATURE				500	// 需要提取的特征点数目
#define ORB_N_OCTAVE_LAYERS			8		// 8, 默认值
#define ORB_FAST_THRESHOLD			20		// 20, default value  ORB特征点检测中的FAST角点检测的阈值
#define ORB_EDGE_THRESHOLD			31		// 31, default value  ORB特征点检测中的边缘阈值。
#define ORB_PATCH_SIZE				31		// 31, default value  ORB特征点检测中的图像块大小
#define ORB_SCALE					1.2		//  default value 1.2 ORB特征点检测中的金字塔层数
int main( int argc, char** argv )
{
    Mat image1 = imread( "../1.png");
    Mat image2 = imread( "../2.png");
//    Mat image1 = imread( "../3.png");
//    Mat image2 = imread( "../4.png");
    assert(image1.data && image2.data && "can not load images");

    vector<KeyPoint> kp1, kp2;
    Mat desp1, desp2;

    Ptr<ORB> orb = ORB::create(ORB_N_FEATURE);
    orb->setFastThreshold(ORB_FAST_THRESHOLD);
    orb->setEdgeThreshold(ORB_EDGE_THRESHOLD);
    orb->setPatchSize(ORB_PATCH_SIZE);
    orb->setNLevels(ORB_N_OCTAVE_LAYERS);
    orb->setScaleFactor(ORB_SCALE);
    orb->setMaxFeatures(ORB_N_FEATURE);                     //设置ORB特征点检测的最大特征点数量。与第1行代码中的参数ORB_N_FEATURE相同。
    orb->setWTA_K(2);                                       //设置ORB特征描述子计算中的WTA_K参数。WTA_K是描述子计算时的选择像素数。
    orb->setScoreType(ORB::HARRIS_SCORE);                   // HARRIS_SCORE，标准Harris角点响应函数
    orb->detectAndCompute(image1, Mat(), kp1, desp1);       //使用ORB特征点检测和描述子计算器检测图像image1中的ORB特征点，并计算对应的ORB描述子。检测到的特征点将存储在kp1中，计算得到的描述子将存储在desp1中。
    orb->detectAndCompute(image2, Mat(), kp2, desp2);

    vector< DMatch > matches;

    BFMatcher matcher_bf(NORM_HAMMING, true); //使用汉明距离度量二进制描述子，允许交叉验证
    vector<DMatch> Matches_bf;//没用
    matcher_bf.match(desp1, desp2, matches);//用的matches

    cout<<"Find total "<<matches.size()<<" matches."<<endl;


//GMS筛点
    vector<DMatch> matches_gms;
    vector<bool> vbInliers;

    /**
     * kp1 第一张图特征点
     * image1.size() 第一张图像的宽和高 cv::Size
     * kp2 第二张图特征点
     * image2.size() 第二张图像的宽和高 cv::Size
     * matches 特征点匹配的结果 vector< DMatch > matches;
    */
   //在构造函数中kp1 kp2中的像素坐标归一化后(0-1) 存入gms.mvP1 mvP2  构造函数中完成了网格初始化
    gms_matcher gms(kp1, image1.size(), kp2, image2.size(), matches);
    //
    int num_inliers = gms.GetInlierMask(vbInliers, false, false);

    cout << "# Refine Matches (after GMS):" << num_inliers  << "/" << matches.size() <<endl;
    // 筛选正确的匹配
    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
        if (vbInliers[i] == true)
        {
            matches_gms.push_back(matches[i]);
        }
    }
    
        // 继续筛选匹配对
    vector< DMatch > goodMatches;
    double minDis = 9999.9;
    
    for ( size_t i=0; i<matches_gms.size(); i++ )
    {
        if ( matches_gms[i].distance < minDis )
            minDis = matches_gms[i].distance;
    }
    cout<<"mindistance"<<minDis<<endl;

    for ( size_t i=0; i<matches_gms.size(); i++ )
    {
        if (matches[i].distance <= max(2*minDis,30.0))
            goodMatches.push_back( matches[i] );
    }
    cout<<"good total number: "<<goodMatches.size()<<endl;
    

    Mat img_goodmatch_gms;
    drawMatches(image1,kp1,image2,kp2,goodMatches,img_goodmatch_gms);
    imshow("final matches",img_goodmatch_gms);

    vector< Point2f > pts1, pts2;
    for (size_t i=0; i<goodMatches.size(); i++)
    {
        pts1.push_back(kp1[goodMatches[i].queryIdx].pt);
        pts2.push_back(kp2[goodMatches[i].trainIdx].pt);
    }


    Mat statusF;//得出内外点状态，内点对应位置为1
    Mat statusH;
    //confidencce越高，将导致更多的迭代次数和计算时间，结果更精确
    Mat F21= findFundamentalMat(pts1, pts2,FM_RANSAC,1.0,0.99,statusF);
    Mat H21= findHomography(pts1,pts2,RANSAC,1.0,statusH,2000,0.99);

    Mat H12 = H21.inv();


    cout<<"F_matrix"<<F21<<endl;
    cout<<"H_matrix"<<H21<<endl;
    const int N=goodMatches.size();

    float scoreF=0;
    // 基于卡方检验计算出的阈值（自由度1）
    const float th = 3.841;
    // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
    const float thScore = 5.991;
    float sigma=1;
	// 信息矩阵，或 协方差矩阵的逆矩阵
    const float invSigmaSquare = 1.0/(sigma*sigma);

    // 提取基础矩阵中的元素数据
    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    // Step 1 获取从参考帧到当前帧的单应矩阵的各个元素
    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

	// 获取从当前帧到参考帧的单应矩阵的各个元素
    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);
    // 初始化scoreH值
    float scoreH=0;
// ----------- 开始你的代码 --------------//

    

 // ----------- 结束你的代码 --------------//
    cout<<"F矩阵评分"<< scoreF<<endl;
    cout<<"H矩阵评分"<< scoreH<<endl;
    float ratio=scoreH/(scoreH+scoreF);

    if(ratio > 0.4)
    cout<<"choose H"<<endl;
    else
    cout<<"choose F"<<endl;
    waitKey(0);
    return 0;
}

