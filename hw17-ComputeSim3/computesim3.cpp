#include <iostream>
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include "computesim3.h"

using namespace std;
using namespace cv;


void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p);

void pose_estimation_3d3d(
  const vector<Point3f> &pts1,
  const vector<Point3f> &pts2,
  Mat &R, Mat &t
);

void bundleAdjustment(
  const vector<Point3f> &points_3d,
  const vector<Point3f> &points_2d,
  Mat &R, Mat &t
);

/**
 * @brief 给出三个点,计算它们的质心以及去质心之后的坐标
 * 
 * @param[in] P     输入的3D点
 * @param[in] Pr    去质心后的点
 * @param[in] C     质心
 */
void ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
{
    // 矩阵P每一行求和，结果存在C。这两句也可以使用CV_REDUCE_AVG选项来实现
    cv::reduce(P,C,1,CV_REDUCE_SUM);
    C = C/P.cols;// 求平均

    for(int i=0; i<P.cols; i++)
    {
        Pr.col(i)=P.col(i)-C;//减去质心
    }
}

/**
 * @brief 按照给定的Sim3变换进行投影操作,得到三维点的2D投影点
 * 
 * @param[in] vP3Dw         3D点
 * @param[in & out] vP2D    投影到图像的2D点
 * @param[in] Tcw           Sim3变换
 */
void Project(vector<cv::Mat> &vP3Dw, vector<cv::Mat> &vP2D, cv::Mat Tcw)
{
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);
    vP2D.clear();
    vP2D.reserve(vP3Dw.size());
    // 对每个3D地图点进行投影操作
    for(size_t i=0, iend=vP3Dw.size(); i<iend; i++)
    {
        // 首先将对方关键帧的地图点坐标转换到这个关键帧的相机坐标系下
        cv::Mat P3Dc = Rcw*vP3Dw[i]+tcw;;
        // 投影
        const float invz = 1/(P3Dc.at<float>(2));
        const float x = P3Dc.at<float>(0)*invz;
        const float y = P3Dc.at<float>(1)*invz;
        vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
    }
}

/**
 * @brief 通过计算的Sim3投影，和自身投影的误差比较，进行内点检测
 * 
 */
void CheckInliers()
{
    // 用计算的Sim3 对所有的地图点投影，得到图像点
    vector<cv::Mat> vP1im2, vP2im1;
    Project(pts2,vP2im1,mT12i);// 把2系中的3D经过Sim3变换(mT12i)到1系中计算重投影坐标
    Project(pts1,vP1im2,mT21i);// 把1系中的3D经过Sim3变换(mT21i)到2系中计算重投影坐标
    
    mnInliersi=0;

    // 对于两帧的每一个匹配点
    for(size_t i=0; i<mvP1im1.size(); i++)
    {
        // 当前关键帧中的地图点直接在当前关键帧图像上的投影坐标mvP1im1，mvP2im2
        // 对于这对匹配关系,在两帧上的投影点距离都要进行计算
        cv::Mat dist1 = mvP1im1[i]-vP2im1[i];
        cv::Mat dist2 = vP1im2[i]-mvP2im2[i];

        // 取距离的平方作为误差
        const float err1 = dist1.dot(dist1);
        const float err2 = dist2.dot(dist2);

        // 根据之前确定的这个最大容许误差来确定这对匹配点是否是外点
        if(err1<mvnMaxError1[i] && err2<mvnMaxError2[i])
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
            mvbInliersi[i]=false;
    }// 遍历其中的每一对匹配点
    
}

/**
 * @brief 根据两组匹配的3D点,计算P2到P1的Sim3变换
 * @param[in] pts1                  匹配的3D点(三个,每个的坐标都是列向量形式,三个点组成了3x3的矩阵)(当前关键帧)
 * @param[in] Pts2                  匹配的3D点(闭环关键帧)
 * @param[in] vAvailableIndices     匹配点对id
 * @return cv::Mat                  计算得到的Sim3矩阵
 */
cv::Mat ComputeSim3(vector<cv::Mat> &pts1, vector<cv::Mat> &pts2, vector<size_t> &vAvailableIndices)
{
    //RANSAC参数
    double mRansacProb = probability;              // 0.99
    int mRansacMinInliers = minInliers;         // 20
    int mRansacMaxIts = maxIterations;          // 最大迭代次数 300

    // 匹配点的数目
    int N1 = vAvailableIndices.size(); // number of correspondences
    std::cout << "N1:" << N1 << endl;
    // 内点标记向量
    mvbInliersi.resize(N1);

    // Adjust Parameters according to number of correspondences
    //随机取一个点是内点的概率
    float epsilon = (float)mRansacMinInliers/N1;

    // Set RANSAC iterations according to probability, epsilon, and max iterations 
    // 计算迭代次数的理论值，也就是经过这么多次采样，其中至少有一次采样中,三对点都是内点
    // epsilon 表示了在这 N 对匹配点中,我随便抽取一对点是内点的概率; 
    // 为了计算Sim3,我们需要从这N对匹配点中取三对点;那么如果我有放回的从这些点中抽取三对点,取这三对点均为内点的概率是 p0=epsilon^3
    // 相应地,如果取三对点中至少存在一对匹配点是外点, 概率为p1=1-p0
    // 当我们进行K次采样的时候,其中每一次采样中三对点中都存在至少一对外点的概率就是p2=p1^k
    // K次采样中,至少有一次采样中三对点都是内点的概率是p=1-p2
    // 候根据 p2=p1^K 我们就可以导出 K 的公式：K=\frac{\log p2}{\log p1}=\frac{\log(1-p)}{\log(1-epsilon^3)}
    // 也就是说，我们进行K次采样,其中至少有一次采样中,三对点都是内点; 因此我们就得到了RANSAC迭代次数的理论值
    int nIterations;

    if(mRansacMinInliers==N1)        
        nIterations=1; // 这种情况的时候最后计算得到的迭代次数的确就是一次
    else
        nIterations = ceil(log(1-mRansacProb)/log(1-pow(epsilon,3)));   

    // 外层的max保证RANSAC能够最少迭代一次;
    // 内层的min的目的是,如果理论值比给定值要小,那么我们优先选择使用较少的理论值来节省时间(其实也有极大概率得到能够达到的最好结果);
    // 如果理论值比给定值要大,那么我们也还是有限选择使用较少的给定值来节省时间
    mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts));

    // 当前正在进行的迭代次数
    int mnIterations = 0;
    
    bool bNoMore = false;    
    // 现在还没有达到最好的效果
    vector<bool> vbInliers = vector<bool>(N1,false);    // 的确和最初传递给这个解算器的地图点向量是保持一致

    // Step 1 如果匹配点比要求的最少内点数还少，不满足Sim3 求解条件，返回空
    // mRansacMinInliers 表示RANSAC所需要的最少内点数目
    
    if(N1<mRansacMinInliers)
    {
        bNoMore = true;  // 表示求解失败
        return cv::Mat();
    }

    // nCurrentIterations：     当前迭代的次数
    // nIterations：            理论迭代次数
    // mnIterations：           总迭代次数
    // mRansacMaxIts：          最大迭代次数
    int nCurrentIterations = 0;
    // int mnIterations = 0;
    // Step 2 随机选择三个点，用于求解后面的Sim3
    // 条件1: 已经进行的总迭代次数还没有超过限制的最大总迭代次数
    // 条件2: 当前迭代次数还没有超过理论迭代次数
    while(mnIterations<mRansacMaxIts && nCurrentIterations<nIterations)
    {
        nCurrentIterations++;// 这个函数中迭代的次数
        mnIterations++;      // 总的迭代次数，默认为最大为300

        vector<Point3f> Point1, Point2;
        cv::Mat P1(3,3,CV_32F); // Relative coordinates to centroid (set 1)
        cv::Mat P2(3,3,CV_32F);
        for(short i = 0; i < 3; ++i)
        {
          // DBoW3中的随机数生成函数
          int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

          int idx1 = vAvailableIndices[randi];

          // P3Dc1i和P3Dc2i中点的排列顺序：
          // x1 x2 x3 ...
          // y1 y2 y3 ...
          // z1 z2 z3 ...
          
          pts1[idx1].copyTo(P1.col(i));
          pts2[idx1].copyTo(P2.col(i));
          
          
          // 从"可用索引列表"中删除这个点的索引 
          vAvailableIndices[randi] = vAvailableIndices.back();
          vAvailableIndices.pop_back();
        }
        
        /***************请开始你的代码*****************/
        // 参考ORB-SLAM2源码补充
        // Sim3计算过程参考论文:
        // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

        // Step 1: 定义3D点质心及去质心后的点
        // O1和O2分别为P1和P2矩阵中3D点的质心
        // Pr1和Pr2为减去质心后的3D点
        cv::Mat Pr1(P1.size(),P1.type()); // Relative coordinates to centroid (set 1)
        cv::Mat Pr2(P2.size(),P2.type()); // Relative coordinates to centroid (set 2)
        cv::Mat O1(3,1,Pr1.type()); // Centroid of P1
        cv::Mat O2(3,1,Pr2.type()); // Centroid of P2

        ComputeCentroid(P1,Pr1,O1);
        ComputeCentroid(P2,Pr2,O2);

        // Step 2: 计算论文中三维点数目n>3的 M 矩阵。这里只使用了3个点
        // Pr2 对应论文中 r_l,i'，Pr1 对应论文中 r_r,i',计算的是P2到P1的Sim3，论文中是left 到 right的Sim3
        cv::Mat M = Pr2*Pr1.t();

        // Step 3: 计算论文中的 N 矩阵

        double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

        cv::Mat N(4,4,P1.type());

        N11 = M.at<float>(0,0)+M.at<float>(1,1)+M.at<float>(2,2);   // Sxx+Syy+Szz
        N12 = M.at<float>(1,2)-M.at<float>(2,1);                    // Syz-Szy
        N13 = M.at<float>(2,0)-M.at<float>(0,2);                    // Szx-Sxz
        N14 = M.at<float>(0,1)-M.at<float>(1,0);                    // ...
        N22 = M.at<float>(0,0)-M.at<float>(1,1)-M.at<float>(2,2);
        N23 = M.at<float>(0,1)+M.at<float>(1,0);
        N24 = M.at<float>(2,0)+M.at<float>(0,2);
        N33 = -M.at<float>(0,0)+M.at<float>(1,1)-M.at<float>(2,2);
        N34 = M.at<float>(1,2)+M.at<float>(2,1);
        N44 = -M.at<float>(0,0)-M.at<float>(1,1)+M.at<float>(2,2);

        N = (cv::Mat_<float>(4,4) << N11, N12, N13, N14,
                                    N12, N22, N23, N24,
                                    N13, N23, N33, N34,
                                    N14, N24, N34, N44);

        // Step 4: 特征值分解求最大特征值对应的特征向量，就是我们要求的旋转四元数
        cv::Mat eval, evec;  // val vec
        // 特征值默认是从大到小排列，所以evec[0] 是最大值
        cv::eigen(N,eval,evec); 

        // N 矩阵最大特征值（第一个特征值）对应特征向量就是要求的四元数（q0 q1 q2 q3），其中q0 是实部
        // 将(q1 q2 q3)放入vec（四元数的虚部）
        cv::Mat vec(1,3,evec.type());
        (evec.row(0).colRange(1,4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)

        // Rotation angle. sin is the norm of the imaginary part, cos is the real part
        // 四元数虚部模长 norm(vec)=sin(theta/2), 四元数实部 evec.at<float>(0,0)=q0=cos(theta/2)
        // 这一步的ang实际是theta/2，theta 是旋转向量中旋转角度
        // ? 这里也可以用 arccos(q0)=angle/2 得到旋转角吧
        double ang=atan2(norm(vec),evec.at<float>(0,0));

        // vec/norm(vec)归一化得到归一化后的旋转向量,然后乘上角度得到包含了旋转轴和旋转角信息的旋转向量vec
        vec = 2*ang*vec/norm(vec); //Angle-axis x. quaternion angle is the half

        mR12i.create(3,3,P1.type());
        // 旋转向量（轴角）转换为旋转矩阵
        cv::Rodrigues(vec,mR12i); // computes the rotation matrix from angle-axis

        // Step 5: Rotate set 2
        // 利用刚计算出来的旋转将三维点旋转到同一个坐标系，P3对应论文里的 r_l,i', Pr1 对应论文里的r_r,i'
        cv::Mat P3 = mR12i*Pr2;

        // Step 6: 计算尺度因子 Scale

        // 论文中有2个求尺度方法。一个是p632右中的位置，考虑了尺度的对称性
        // 代码里实际使用的是另一种方法，这个公式对应着论文中p632左中位置的那个
        // Pr1 对应论文里的r_r,i',P3对应论文里的 r_l,i',(经过坐标系转换的Pr2), n=3, 剩下的就和论文中都一样了
        double nom = Pr1.dot(P3);
        // 准备计算分母
        cv::Mat aux_P3(P3.size(),P3.type());
        aux_P3=P3;
        // 先得到平方
        cv::pow(P3,2,aux_P3);
        double den = 0;

        // 然后再累加
        for(int i=0; i<aux_P3.rows; i++)
        {
            for(int j=0; j<aux_P3.cols; j++)
            {
                den+=aux_P3.at<float>(i,j);
            }
        }

        ms12i = nom/den;

        // Step 7: 计算平移Translation
        mt12i.create(1,3,P1.type());
        // 论文中平移公式
        mt12i = O1 - ms12i*mR12i*O2;

        // Step 8: 计算双向变换矩阵，目的是在后面的检查的过程中能够进行双向的投影操作

        // Step 8.1 用尺度，旋转，平移构建变换矩阵 T12
        mT12i = cv::Mat::eye(4,4,P1.type());

        cv::Mat sR = ms12i*mR12i;

        //         |sR t|
        // mT12i = | 0 1|
        sR.copyTo(mT12i.rowRange(0,3).colRange(0,3));
        mt12i.copyTo(mT12i.rowRange(0,3).col(3));

        // Step 8.2 T21

        mT21i = cv::Mat::eye(4,4,P1.type());

        cv::Mat sRinv = (1.0/ms12i)*mR12i.t();
        sRinv.copyTo(mT21i.rowRange(0,3).colRange(0,3));
        cv::Mat tinv = -sRinv*mt12i;
        tinv.copyTo(mT21i.rowRange(0,3).col(3));

        /***************结束代码*********************/
        
        CheckInliers();

        int mnBestInliers = 0;
        cv::Mat mBestT12;                           // 存储最好的一次迭代中得到的变换矩阵
        cv::Mat mBestRotation;                      // 存储最好的一次迭代中得到的旋转
        cv::Mat mBestTranslation;
        float mBestScale;
        std::vector<bool> mvbBestInliers;
        // Step 2.4 记录并更新最多的内点数目及对应的参数
        if(mnInliersi>=mnBestInliers)
        {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i.clone();
            mBestRotation = mR12i.clone();
            mBestTranslation = mt12i.clone();
            mBestScale = ms12i;
            if(mnInliersi>mRansacMinInliers) // 只要计算得到一次合格的Sim变换，就直接返回
            {
                // 返回值,告知得到的内点数目
                // nInliers = mnInliersi;
                for(int i=0; i<N1; i++)
                    if(mvbInliersi[i])
                        // 标记为内点
                        vbInliers[vAvailableIndices[i]] = true;
                return mBestT12;
            } // 如果当前次迭代已经合格了,直接返回
        } // 更新最多的内点数目
    } // 迭代循环

    // Step 3 如果已经达到了最大迭代次数了还没得到满足条件的Sim3，说明失败了，放弃，返回
    if(mnIterations>=mRansacMaxIts)
        bNoMore=true;

    return cv::Mat();   // no more的时候返回的是一个空矩阵
}


/**
 * @brief        传入两张图片，计算并匹配特征点
 * @param[in] img_1 第一张图片
 * @param[in] img_2 第二张图片
 * @param[in out] keypoints_1  存第一张图片里的特征点
 * @param[in out] keypoints_2  存第二张图片里的特征点
 * @param[in out] matches  存有效的匹配关系，描述子距离过大的不要了
 * @return 
*/
void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
  //-- 初始化
  Mat descriptors_1, descriptors_2;
  // used in OpenCV3
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  // use this if you are in OpenCV2
  // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
  // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> match;
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}

Point2d pixel2cam(const Point2d &p) {
  return Point2d(
    (p.x - K.at<float>(0, 2)) / K.at<float>(0, 0),
    (p.y - K.at<float>(1, 2)) / K.at<float>(1, 1)
  );
}

float ComputerSigmaSquare(cv::KeyPoint keypoints){
  float sigma = 1.2f;
  float scaleFactor = 1.2f;
  int nLevels = 8;
  float sigmaSquare = pow(sigma*scaleFactor, 2)*(1 << 2*keypoints.octave);
  return sigmaSquare;
}

int main(int argc, char **argv) {
  
  string save_path = "../";
  Mat img_1 = imread(save_path + "1.png", 0);
  Mat img_2 = imread(save_path + "2.png", 0);
  assert(img_1.data && img_2.data && "Can not load images!");


  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;

  // 建立3D点
  Mat depth1 = imread(save_path + "1_depth.png", 0);
  Mat depth2 = imread(save_path + "2_depth.png", 0);
  assert(depth1.data && depth2.data && "Can not load images!");

  size_t idx = 0;
  vAvailableIndices.reserve(matches.size());
  //遍历所有匹配关系
  for (DMatch m:matches) {
    ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
    if (d1 == 0 || d2 == 0)   // bad depth
      continue;
    
    Mat mat1(2,1,CV_32F);
    Mat mat2(2,1,CV_32F);
    mat1.at<Vec2f>(0,0) = Vec2f(keypoints_1[m.queryIdx].pt.x,keypoints_1[m.queryIdx].pt.y);
    //匹配特征点对第一张图中的xy坐标
    mvP1im1.push_back(mat1);
    float SigmaSquare1 = ComputerSigmaSquare(keypoints_1[m.queryIdx]);

    mvnMaxError1.push_back(9.210*SigmaSquare1);
    //匹配特征点对第二张图中的xy坐标
    mat2.at<Vec2f>(0,0) = Vec2f(keypoints_2[m.trainIdx].pt.x,keypoints_2[m.trainIdx].pt.y);
    mvP2im2.push_back(mat2);
    float SigmaSquare2 = ComputerSigmaSquare(keypoints_2[m.trainIdx]);

    mvnMaxError2.push_back(9.210*SigmaSquare2);

    mvnIndices1.push_back(m.queryIdx);

    /***************请开始你的代码*****************/
    // 构建点对
    //目前在遍历两张图的匹配关系
    //构建出pts1 pts2 每一列是一个匹配特征点在自身相机坐标系下的三维坐标
    //vAvailableIndices中存储的id是pts1 pts2中匹配点对的id
    // kp1和kp2是匹配的图像特征点
    // const cv::KeyPoint &kp1 = keypoints_1[m.queryIdx];
    // const cv::KeyPoint &kp2 = keypoints_2[m.trainIdx];
    //计算三维点坐标 结果要3行1列的形式
    //取出相机内参数
    const float &fx = K.at<float>(0,0);
    const float &fy = K.at<float>(1,1);
    const float &cx = K.at<float>(0,2);
    const float &cy = K.at<float>(1,2);
    //像素坐标转相机系坐标
    float pt1_normalization_x = (mat1.at<float>(0, 0)-cx)/fx*static_cast<float>(d1);
    float pt1_normalization_y = (mat1.at<float>(1, 0)-cy)/fy*static_cast<float>(d1);
    float pt1_normalization_z = static_cast<float>(d1);
    float pt2_normalization_x = (mat2.at<float>(0, 0)-cx)/fx*static_cast<float>(d2);
    float pt2_normalization_y = (mat2.at<float>(1, 0)-cy)/fy*static_cast<float>(d2);
    float pt2_normalization_z = static_cast<float>(d2);
    //第一张图中特征点的相机系坐标
    // cv::Mat pt1_normalization = (Mat_<float>(3, 1) << pt1_normalization_x, pt1_normalization_y, pt1_normalization_z);
    //第二张图中特征点的相机系坐标
    // cv::Mat pt2_normalization = (Mat_<float>(3, 1) << pt2_normalization_x, pt2_normalization_y, pt2_normalization_z);
    //存入pts1 pts2
    pts1.push_back((cv::Mat_<float>(3, 1) << pt1_normalization_x, pt1_normalization_y, pt1_normalization_z));
    pts2.push_back((cv::Mat_<float>(3, 1) << pt2_normalization_x, pt2_normalization_y, pt2_normalization_z));
    // 所有有效三维点的索引
    //下面这句会导致内存错误
    vAvailableIndices.push_back(idx);
    std::cout << "idx:" << idx << endl;
    idx++;
     /***************结束代码*********************/
  }

  cv::Mat Scm  = ComputeSim3(pts1, pts2, vAvailableIndices);
  cout << "Scm: " << endl << Scm << endl;

}