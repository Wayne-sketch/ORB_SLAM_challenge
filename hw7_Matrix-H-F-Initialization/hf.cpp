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
    Mat image1 = imread( "../3.png");
    Mat image2 = imread( "../4.png");
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

    /**
     * DMatch类是OpenCV中用于表示特征点匹配的数据结构，它包含以下成员：
     * queryIdx: 一个整数，表示匹配中的查询图像的特征点的索引。
     * trainIdx: 一个整数，表示匹配中的训练图像（模板图像）的特征点的索引。
     * distance: 一个浮点数，表示匹配的距离或相似度。距离越小或相似度越大表示匹配越好。
     * imgIdx: 一个整数，当一个查询图像匹配多个训练图像时使用。一般情况下为0，表示只有一个训练图像。
    */
    vector< DMatch > matches;

    BFMatcher matcher_bf(NORM_HAMMING, true); //使用汉明距离度量二进制描述子，允许交叉验证
    vector<DMatch> Matches_bf;//没用
    matcher_bf.match(desp1, desp2, matches);//用的matches

    cout<<"Find total "<<matches.size()<<" matches."<<endl;


//GMS筛点
    //存GMS匹配后所有作为内点的特征匹配点,最终还会检测一次，存在good_matches里
    vector<DMatch> matches_gms;
    //存储matches匹配关系的有效情况
    vector<bool> vbInliers;

    /**
     * kp1 第一张图特征点
     * image1.size() 第一张图像的宽和高 cv::Size
     * kp2 第二张图特征点
     * image2.size() 第二张图像的宽和高 cv::Size
     * matches 原始特征点匹配的结果 vector< DMatch > matches;
    */
   //在构造函数中kp1 kp2中的像素坐标归一化后(0-1) 存入gms.mvP1 mvP2  构造函数中完成了网格初始化
    gms_matcher gms(kp1, image1.size(), kp2, image2.size(), matches);
    //不使用尺度 旋转匹配 内点匹配关系经过验证后存在vbInliers
    int num_inliers = gms.GetInlierMask(vbInliers, false, false);

    //验证后内点数量/匹配点数量
    cout << "# Refine Matches (after GMS):" << num_inliers  << "/" << matches.size() <<endl;
    // 筛选正确的匹配
    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
        if (vbInliers[i] == true)
        {
            //GMS匹配后有效的内点匹配结果
            matches_gms.push_back(matches[i]);
        }
    }
    
        // 继续筛选匹配对
    vector< DMatch > goodMatches;
    double minDis = 9999.9;
    
    //遍历所有内点匹配
    for ( size_t i=0; i<matches_gms.size(); i++ )
    {
        //获取最低匹配相似度（距离）
        if ( matches_gms[i].distance < minDis )
            minDis = matches_gms[i].distance;
    }
    cout<<"mindistance"<<minDis<<endl;

    //遍历所有内点匹配
    for ( size_t i=0; i<matches_gms.size(); i++ )
    {
        //匹配距离越小越好，小于30或小于最近匹配距离的二倍 成为goodMatches
        if (matches[i].distance <= max(2*minDis,30.0))
            goodMatches.push_back( matches[i] );
    }
    cout<<"good total number: "<<goodMatches.size()<<endl;
    
    //显示GMS匹配筛选后特征点匹配的输出图像
    Mat img_goodmatch_gms;
    /**
     * void drawMatches(const Mat& img1, const vector<KeyPoint>& keypoints1,
                 const Mat& img2, const vector<KeyPoint>& keypoints2,
                 const vector<DMatch>& matches1to2, Mat& outImg,
                 const Scalar& matchColor = Scalar::all(-1),
                 const Scalar& singlePointColor = Scalar::all(-1),
                 const vector<char>& matchesMask = vector<char>(),
                 int flags = DrawMatchesFlags::DEFAULT );
        img1：输入的第一幅图像。
        keypoints1：第一幅图像中的特征点（关键点）。
        img2：输入的第二幅图像。
        keypoints2：第二幅图像中的特征点（关键点）。
        matches1to2：特征点匹配的结果，即两幅图像中的特征点对。
        outImg：输出图像，即绘制了匹配结果的图像。
        matchColor：用于绘制匹配线的颜色，默认为Scalar::all(-1)，表示随机颜色。
        singlePointColor：用于绘制单个特征点的颜色，默认为Scalar::all(-1)，表示随机颜色。
        matchesMask：匹配掩码，用于指示是否绘制特定的匹配对，若为空则绘制所有匹配。
        flags：绘制匹配的标志，默认为DrawMatchesFlags::DEFAULT，可以通过设置不同的标志来控制绘制的方式，例如是否显示特征点，是否显示匹配线等
    */
   //画出匹配关系
    drawMatches(image1,kp1,image2,kp2,goodMatches,img_goodmatch_gms);
    imshow("final matches",img_goodmatch_gms);

    vector< Point2f > pts1, pts2;
    //遍历所有再次筛选后的内点
    for (size_t i=0; i<goodMatches.size(); i++)
    {
        //取出像素坐标
        pts1.push_back(kp1[goodMatches[i].queryIdx].pt);
        pts2.push_back(kp2[goodMatches[i].trainIdx].pt);
    }


    Mat statusF;//得出内外点状态，内点对应位置为1
    Mat statusH;
    /**
     * findFundamentalMat函数是OpenCV中用于估计基础矩阵的函数。基础矩阵是在两个视图之间进行几何校准和匹配的关键矩阵，它描述了两个相机视图之间的对应关系。
     * 基础矩阵可以用于计算两个视图之间的本质矩阵，从而实现相机的位姿恢复和三维重构。
     * 函数原型：
     * cv::findFundamentalMat(points1, points2, method, ransacThreshold, confidence, mask);
     * 参数说明：
     * points1：第一个视图中的特征点坐标，通常为std::vector<cv::Point2f>类型。
     * points2：第二个视图中的特征点坐标，与points1对应，也为std::vector<cv::Point2f>类型。
     * method：基础矩阵计算的方法，有两个选项：
     * cv::FM_RANSAC：使用RANSAC算法进行计算，适用于含有噪声和外点的数据。详细理论？？？？
     * cv::FM_8POINT：使用8点算法进行计算，适用于较小数据集。
     * ransacThreshold：RANSAC算法的阈值，用于判断数据点是否属于内点。
     * confidence：RANSAC算法的置信度，通常为0.99。
     * mask：输出的内点掩码，用于指示哪些数据点属于内点，哪些数据点属于外点。
     * 函数返回值：
     * 如果成功找到基础矩阵，则返回基础矩阵的cv::Mat对象。
     * 如果未找到基础矩阵，则返回空的cv::Mat对象。
     * findFundamentalMat函数通过特征点的对应关系估计两个视图之间的基础矩阵。这个函数在计算机视觉中广泛用于图像拼接、立体视觉、相机运动恢复等应用中 
    */
    //confidencce越高，将导致更多的迭代次数和计算时间，结果更精确
    Mat F21= findFundamentalMat(pts1, pts2,FM_RANSAC,1.0,0.99,statusF);
    /**
     * cv::findHomography(points1, points2, method, ransacThreshold, mask);
     * points1：第一个平面中的特征点坐标，通常为std::vector<cv::Point2f>类型。
     * points2：第二个平面中的特征点坐标，与points1对应，也为std::vector<cv::Point2f>类型。
     * method：单应矩阵计算的方法，有两个选项：
     * cv::RANSAC：使用RANSAC算法进行计算，适用于含有噪声和外点的数据。详细理论？？？？
     * cv::LMEDS：使用最小中值算法进行计算，适用于较小数据集。详细理论？？？？
     * ransacThreshold：RANSAC算法的阈值，用于判断数据点是否属于内点。
     * mask：输出的内点掩码，用于指示哪些数据点属于内点，哪些数据点属于外点。
     * 函数返回值：
     * 如果成功找到单应矩阵，则返回单应矩阵的cv::Mat对象。
     * 如果未找到单应矩阵，则返回空的cv::Mat对象。
    */
    Mat H21= findHomography(pts1,pts2,RANSAC,1.0,statusH,2000,0.99);

    Mat H12 = H21.inv();
    cout<<"F_matrix"<<F21<<endl;
    cout<<"H_matrix"<<H21<<endl;
    cout<<"H_inv_matrix"<<H12<<endl;
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
    // const float f11 = F21.at<float>(0,0);
    // const float f12 = F21.at<float>(0,1);
    // const float f13 = F21.at<float>(0,2);
    // const float f21 = F21.at<float>(1,0);
    // const float f22 = F21.at<float>(1,1);
    // const float f23 = F21.at<float>(1,2);
    // const float f31 = F21.at<float>(2,0);
    // const float f32 = F21.at<float>(2,1);
    // const float f33 = F21.at<float>(2,2);
    //提取矩阵元素的时候一定要用64位，因为OpenCV函数返回的就是64位，linux系统里float是32位，取错的话数是错的
    const float f11 = F21.at<double>(0,0);
    const float f12 = F21.at<double>(0,1);
    const float f13 = F21.at<double>(0,2);
    const float f21 = F21.at<double>(1,0);
    const float f22 = F21.at<double>(1,1);
    const float f23 = F21.at<double>(1,2);
    const float f31 = F21.at<double>(2,0);
    const float f32 = F21.at<double>(2,1);
    const float f33 = F21.at<double>(2,2);

    // Step 1 获取从参考帧到当前帧的单应矩阵的各个元素
    // const float h11 = H21.at<float>(0,0);
    // const float h12 = H21.at<float>(0,1);
    // const float h13 = H21.at<float>(0,2);
    // const float h21 = H21.at<float>(1,0);
    // const float h22 = H21.at<float>(1,1);
    // const float h23 = H21.at<float>(1,2);
    // const float h31 = H21.at<float>(2,0);
    // const float h32 = H21.at<float>(2,1);
    // const float h33 = H21.at<float>(2,2);
    const float h11 = H21.at<double>(0,0);
    const float h12 = H21.at<double>(0,1);
    const float h13 = H21.at<double>(0,2);
    const float h21 = H21.at<double>(1,0);
    const float h22 = H21.at<double>(1,1);
    const float h23 = H21.at<double>(1,2);
    const float h31 = H21.at<double>(2,0);
    const float h32 = H21.at<double>(2,1);
    const float h33 = H21.at<double>(2,2);

	// 获取从当前帧到参考帧的单应矩阵的各个元素
    // const float h11inv = H12.at<float>(0,0);
    // const float h12inv = H12.at<float>(0,1);
    // const float h13inv = H12.at<float>(0,2);
    // const float h21inv = H12.at<float>(1,0);
    // const float h22inv = H12.at<float>(1,1);
    // const float h23inv = H12.at<float>(1,2);
    // const float h31inv = H12.at<float>(2,0);
    // const float h32inv = H12.at<float>(2,1);
    // const float h33inv = H12.at<float>(2,2);
    const float h11inv = H12.at<double>(0,0);
    const float h12inv = H12.at<double>(0,1);
    const float h13inv = H12.at<double>(0,2);
    const float h21inv = H12.at<double>(1,0);
    const float h22inv = H12.at<double>(1,1);
    const float h23inv = H12.at<double>(1,2);
    const float h31inv = H12.at<double>(2,0);
    const float h32inv = H12.at<double>(2,1);
    const float h33inv = H12.at<double>(2,2);
    // 初始化scoreH值
    float scoreH=0;
// ----------- 开始你的代码 --------------//

//     // 说明：在已知n维观测数据误差服从N(0，sigma）的高斯分布时
//     // 其误差加权最小二乘结果为  sum_error = SUM(e(i)^T * Q^(-1) * e(i))
//     // 其中：e(i) = [e_x,e_y,...]^T, Q为观测数据协方差矩阵，即sigma * sigma组成的协方差矩阵
//     // 误差加权最小二乘结果越小，说明观测数据精度越高
//     // 那么，score = SUM((th - e(i)^T * Q^(-1) * e(i)))的分数就越高
//     // 算法目标： 检查单应变换矩阵
//     // 检查方式：通过H矩阵，进行参考帧和当前帧之间的双向投影，并计算起加权最小二乘投影误差

//     // 算法流程
//     // input: 单应性矩阵 H21, H12, 匹配点集 mvKeys1
//     //    do:
//     //        for p1(i), p2(i) in mvKeys:
//     //           error_i1 = ||p2(i) - H21 * p1(i)||2
//     //           error_i2 = ||p1(i) - H12 * p2(i)||2
//     //           
//     //           w1 = 1 / sigma / sigma
//     //           w2 = 1 / sigma / sigma
//     // 
//     //           if error1 < th
//     //              score +=   th - error_i1 * w1
//     //           if error2 < th
//     //              score +=   th - error_i2 * w2
//     // 
//     //           if error_1i > th or error_2i > th
//     //              p1(i), p2(i) are inner points
//     //              vbMatchesInliers(i) = true
//     //           else 
//     //              p1(i), p2(i) are outliers
//     //              vbMatchesInliers(i) = false
//     //           end
//     //        end
//     //   output: score, inliers

    vector<bool> vbInliers_goodmatches;//在这个代码里没什么实际作用
    vbInliers_goodmatches.resize(N);

    // Step 2 通过H矩阵，进行参考帧和当前帧之间的双向投影，并计算起加权重投影误差
    for(int i=0; i<N; i++){
        //一开始都默认为Inlier
        bool bIn = true;
        // Step 2.1 提取参考帧和当前帧之间的特征匹配点对
        const cv::KeyPoint &current_kp1 = kp1[goodMatches[i].queryIdx];
        const cv::KeyPoint &current_kp2 = kp2[goodMatches[i].trainIdx];
        const float u1 = current_kp1.pt.x;
        const float v1 = current_kp1.pt.y;
        const float u2 = current_kp2.pt.x;
        const float v2 = current_kp2.pt.y;

        // Step 2.2 计算 img2 到 img1 的重投影误差
        // x1 = H12*x2
        // 将图像2中的特征点通过单应变换投影到图像1中
        // |u1|   |h11inv h12inv h13inv||u2|   |u2in1|
        // |v1| = |h21inv h22inv h23inv||v2| = |v2in1| * w2in1inv
        // |1 |   |h31inv h32inv h33inv||1 |   |  1  |
		// 计算投影归一化坐标
        const float w2in1inv = 1.0/(h31inv * u2 + h32inv * v2 + h33inv);
        const float u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
        const float v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;
   
        // 计算重投影误差 = ||p1(i) - H12 * p2(i)||2
        const float squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);
        const float chiSquare1 = squareDist1 * invSigmaSquare;

        // Step 2.3 用阈值标记离群点，内点的话累加得分
        if(chiSquare1>thScore)
            bIn = false;    
        else
            // 误差越大，得分越低
            scoreH += thScore - chiSquare1;

        // 计算从img1 到 img2 的投影变换误差
        // x1in2 = H21*x1
        // 将图像2中的特征点通过单应变换投影到图像1中
        // |u2|   |h11 h12 h13||u1|   |u1in2|
        // |v2| = |h21 h22 h23||v1| = |v1in2| * w1in2inv
        // |1 |   |h31 h32 h33||1 |   |  1  |
		// 计算投影归一化坐标
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        // cout<<"w1in2inv="<<w1in2inv<<endl;//正常
        // cout<<"h31="<<h31<<endl;
        // cout<<"u1="<<u1<<endl;//正常
        // cout<<"h32="<<h32<<endl;
        // cout<<"v1="<<v1<<endl;//正常
        // cout<<"h33="<<h33<<endl;
        // cout<<"h11*u1+h12*v1+h13="<<h11*u1+h12*v1+h13<<endl;//很大
        // cout<<"h11="<<h11<<endl;
        // cout<<"h12="<<h12<<endl;
        // cout<<"h13="<<h13<<endl;
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        // 计算重投影误差 
        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);
        const float chiSquare2 = squareDist2*invSigmaSquare;
        //debug
        // cout<<"u2="<<u2<<endl;
        // cout<<"u1in2="<<u1in2<<endl; //有问题
        // cout<<"squareDist2="<<squareDist2<<endl;
        // cout<<"u2-u1in2="<<u2-u1in2<<endl;
        // cout<<"v2-v1in2="<<v2-v1in2<<endl;
        // cout<<"chiSquare2="<<chiSquare2<<endl;
        // 用阈值标记离群点，内点的话累加得分
        if(chiSquare2>thScore)
            bIn = false;
        else
            scoreH += thScore - chiSquare2;   
            // cout<<"scoreH="<<scoreH<<endl;

        // Step 2.4 如果从img2 到 img1 和 从img1 到img2的重投影误差均满足要求，则说明是Inlier point
        if(bIn)
            vbInliers_goodmatches[i]=true;
        else
            vbInliers_goodmatches[i]=false;
    }

    // 说明：在已值n维观测数据误差服从N(0，sigma）的高斯分布时
    // 其误差加权最小二乘结果为  sum_error = SUM(e(i)^T * Q^(-1) * e(i))
    // 其中：e(i) = [e_x,e_y,...]^T, Q维观测数据协方差矩阵，即sigma * sigma组成的协方差矩阵
    // 误差加权最小二次结果越小，说明观测数据精度越高
    // 那么，score = SUM((th - e(i)^T * Q^(-1) * e(i)))的分数就越高
    // 算法目标：检查基础矩阵
    // 检查方式：利用对极几何原理 p2^T * F * p1 = 0
    // 假设：三维空间中的点 P 在 img1 和 img2 两图像上的投影分别为 p1 和 p2（两个为同名点）
    //   则：p2 一定存在于极线 l2 上，即 p2*l2 = 0. 而l2 = F*p1 = (a, b, c)^T
    //      所以，这里的误差项 e 为 p2 到 极线 l2 的距离，如果在直线上，则 e = 0
    //      根据点到直线的距离公式：d = (ax + by + c) / sqrt(a * a + b * b)
    //      所以，e =  (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)

    // 算法流程
    // input: 基础矩阵 F 左右视图匹配点集 mvKeys1
    //    do:
    //        for p1(i), p2(i) in mvKeys:
    //           l2 = F * p1(i)
    //           l1 = p2(i) * F
    //           error_i1 = dist_point_to_line(x2,l2)
    //           error_i2 = dist_point_to_line(x1,l1)
    //           
    //           w1 = 1 / sigma / sigma
    //           w2 = 1 / sigma / sigma
    // 
    //           if error1 < th
    //              score +=   thScore - error_i1 * w1
    //           if error2 < th
    //              score +=   thScore - error_i2 * w2
    // 
    //           if error_1i > th or error_2i > th
    //              p1(i), p2(i) are inner points
    //              vbMatchesInliers(i) = true
    //           else 
    //              p1(i), p2(i) are outliers
    //              vbMatchesInliers(i) = false
    //           end
    //        end
    //   output: score, inliers

	// 预分配空间
    vbInliers_goodmatches.resize(N);

    // Step 2 计算img1 和 img2 在估计 F 时的score值
    for(int i=0; i<N; i++)
    {
		//默认为这对特征点是Inliers
        bool bIn_F = true;

	    // Step 2.1 提取参考帧和当前帧之间的特征匹配点对
        const cv::KeyPoint &current_kp1_F = kp1[goodMatches[i].queryIdx];
        const cv::KeyPoint &current_kp2_F = kp2[goodMatches[i].trainIdx];

		// 提取出特征点的坐标
        const float u1_F = current_kp1_F.pt.x;
        const float v1_F = current_kp1_F.pt.y;
        const float u2_F = current_kp2_F.pt.x;
        const float v2_F = current_kp2_F.pt.y;

        // Reprojection error in second image
        // Step 2.2 计算 img1 上的点在 img2 上投影得到的极线 l2 = F21 * p1 = (a2,b2,c2)
		const float a2 = f11*u1_F+f12*v1_F+f13;
        const float b2 = f21*u1_F+f22*v1_F+f23;
        const float c2 = f31*u1_F+f32*v1_F+f33;
    
        // Step 2.3 计算误差 e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
        const float num2 = a2*u2_F+b2*v2_F+c2;
        const float squareDist1_F = num2*num2/(a2*a2+b2*b2);
        // 带权重误差
        const float chiSquare1_F = squareDist1_F*invSigmaSquare;
		
        // Step 2.4 误差大于阈值就说明这个点是Outlier 
        // ? 为什么判断阈值用的 th（1自由度），计算得分用的thScore（2自由度）
        // ? 可能是为了和CheckHomography 得分统一？
        if(chiSquare1_F>th)
            bIn_F = false;
        else
            // 误差越大，得分越低
            scoreF += thScore - chiSquare1_F;

        // 计算img2上的点在 img1 上投影得到的极线 l1= p2 * F21 = (a1,b1,c1)
        const float a1 = f11*u2_F+f21*v2_F+f31;
        const float b1 = f12*u2_F+f22*v2_F+f32;
        const float c1 = f13*u2_F+f23*v2_F+f33;

        // 计算误差 e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
        const float num1 = a1*u1_F+b1*v1_F+c1;
        const float squareDist2_F = num1*num1/(a1*a1+b1*b1);

        // 带权重误差
        const float chiSquare2_F = squareDist2_F*invSigmaSquare;

        // 误差大于阈值就说明这个点是Outlier 
        if(chiSquare2_F>th)
            bIn_F = false;
        else
            scoreF += thScore - chiSquare2_F;
        
        // Step 2.5 保存结果
        if(bIn_F)
            vbInliers_goodmatches[i]=true;
        else
            vbInliers_goodmatches[i]=false;
    }
    
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

