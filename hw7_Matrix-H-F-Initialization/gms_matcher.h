#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <ctime>
using namespace std;
using namespace cv;

#define THRESH_FACTOR 6

// 8 possible rotation and each one is 3 X 3  二维数组
const int mRotationPatterns[8][9] = {
	1,2,3,
	4,5,6,
	7,8,9,

	4,1,2,
	7,5,3,
	8,9,6,

	7,4,1,
	8,5,2,
	9,6,3,

	8,7,4,
	9,5,1,
	6,3,2,

	9,8,7,
	6,5,4,
	3,2,1,

	6,9,8,
	3,5,7,
	2,1,4,

	3,6,9,
	2,5,8,
	1,4,7,

	2,3,6,
	1,5,9,
	4,7,8
};

// 5 level scales
//尺度变换比
const double mScaleRatios[5] = { 1.0, 1.0 / 2, 1.0 / sqrt(2.0), sqrt(2.0), 2.0 };


//Grid-based Motion Statistics
/**
 * GMS代表"Grid-based Motion Statistics"，是一种特征点匹配算法。GMS算法是一种改进的特征点匹配算法，旨在提高特征点匹配的鲁棒性和准确性。
 * GMS算法基于两个关键观察：
 * 特征点在图像中的分布通常具有空间相关性，即相邻的特征点倾向于匹配到相邻的位置。
 * 特征点匹配的错误通常是局部化的，即只有少数特征点之间存在错误匹配。
 * GMS算法通过在特征点空间上建立网格并统计匹配的运动信息来提高特征点匹配的质量。具体而言，它将图像分成多个网格，并在每个网格内统计匹配点对的数量。
 * 然后，通过统计每个网格内匹配点对的数量来确定特征点的匹配对。这种方法可以帮助过滤掉匹配错误的点对，从而提高匹配的准确性。
 * GMS算法在特征点匹配任务中取得了良好的性能，尤其在噪声较大、场景变化较大的情况下，它能够更好地处理误匹配和局部不一致的问题。因此，GMS算法在计算机
 * 视觉领域中被广泛应用于图像拼接、图像配准和SLAM等任务中。
*/
class gms_matcher
{
public:
	// OpenCV Keypoints & Correspond Image Size & Nearest Neighbor Matches 
	//构造函数
	gms_matcher(const vector<KeyPoint> &vkp1, const Size size1, const vector<KeyPoint> &vkp2, const Size size2, const vector<DMatch> &vDMatches) 
	{
		// Input initialize
		//mvP1 mvP2存像素坐标归一化后的坐标(0-1)
		NormalizePoints(vkp1, size1, mvP1);
		NormalizePoints(vkp2, size2, mvP2);
		//匹配关系数量
		mNumberMatches = vDMatches.size();
		//把vDMatches转换成mvMatches直接对特征关系中特征点的索引进行存储
		ConvertMatches(vDMatches, mvMatches);

		// Grid initialize 网格初始化
		mGridSizeLeft = Size(20, 20);
		//网格数
		mGridNumberLeft = mGridSizeLeft.width * mGridSizeLeft.height;

		// Initialize the neihbor of left grid 
		//Mat::zeros(行数，列数，矩阵元素数据类型int 通道1)  每行存储一个格子的邻居
		mGridNeighborLeft = Mat::zeros(mGridNumberLeft, 9, CV_32SC1);
		InitalizeNiehbors(mGridNeighborLeft, mGridSizeLeft);
	};
	~gms_matcher() {};

private:

	// Normalized Points
	//存两幅图像像素坐标归一化后的坐标(0-1)
	vector<Point2f> mvP1, mvP2;

	// Matches
	//这个 vector 存储了特征点的匹配关系。每个匹配对由两个索引组成，具体是哪个？？？
	vector<pair<int, int> > mvMatches;

	// Number of Matches
	//unsigned long long
	//这个变量表示特征点的匹配数量，即 mvMatches 中的匹配对数量。
	size_t mNumberMatches;

	// Grid Size  OpenCV定义的数据类型
	//这两个变量分别表示左图像和右图像的网格大小，用于进行网格划分
	Size mGridSizeLeft, mGridSizeRight;
	//这两个变量表示左图像和右图像的网格数量，用于进行网格划分
	int mGridNumberLeft;
	int mGridNumberRight;

	// x	  : left grid idx
	// y      :  right grid idx
	// value  : how many matches from idx_left to idx_right
	//这是一个 OpenCV 的矩阵（Mat），用于存储匹配点对在网格中的统计信息。
	Mat mMotionStatistics;

	// 这个 vector 存储了左图像中每个网格中包含的特征点数量。
	vector<int> mNumberPointsInPerCellLeft;

	// Inldex  : grid_idx_left
	// Value   : grid_idx_right
	// 这个 vector 存储了每个左图像网格与其对应的右图像网格的索引。
	vector<int> mCellPairs;

	// Every Matches has a cell-pair 
	// first  : grid_idx_left
	// second : grid_idx_right
	//这个 vector 存储了匹配点对所在的网格索引对。
	vector<pair<int, int> > mvMatchPairs;

	// Inlier Mask for output
	//如果是经过验证的特征点对匹配关系，对应位置是true vector每个位置对应一个匹配特征点对
	vector<bool> mvbInlierMask;

	//这两个是 OpenCV 的矩阵（Mat），用于存储网格的邻接关系，用于后续的优化和匹配过程。存储形式？
	Mat mGridNeighborLeft;
	Mat mGridNeighborRight;

public:

	// Get Inlier Mask
	// Return number of inliers 
	// 函数的作用是获取内点，即根据一些条件判断特征点是否为内点，并将结果保存在 vbInliers 中。
	int GetInlierMask(vector<bool> &vbInliers, bool WithScale = false, bool WithRotation = false);

private:

	// Normalize Key Points to Range(0 - 1)
	//用于将特征点坐标归一化到指定范围。
	/*const vector<KeyPoint> &kp：输入特征点向量，表示包含多个关键点的容器。
	const Size &size：指定的尺寸，表示图像的大小或区域的大小。
	vector<Point2f> &npts：输出参数，归一化后的特征点坐标向量，保存了对应于输入特征点的归一化后的坐标。*/
	void NormalizePoints(const vector<KeyPoint> &kp, const Size &size, vector<Point2f> &npts) {
		//获取输入特征点的数量 numP
		const size_t numP = kp.size();
		//从 size 中获取图像的宽度和高度，分别保存在变量 width 和 height 中
		const int width   = size.width;
		const int height  = size.height;
		//使用 npts.resize(numP) 对输出特征点坐标向量 npts 进行大小的调整，使其能够容纳 numP 个特征点坐标。
		npts.resize(numP);

		for (size_t i = 0; i < numP; i++)
		{
			npts[i].x = kp[i].pt.x / width;
			npts[i].y = kp[i].pt.y / height;
		}
	}

	// Convert OpenCV DMatch to Match (pair<int, int>)
	void ConvertMatches(const vector<DMatch> &vDMatches, vector<pair<int, int> > &vMatches) {
		//mNumberMatches 特征点的匹配数量
		vMatches.resize(mNumberMatches);
		for (size_t i = 0; i < mNumberMatches; i++)
		{
			vMatches[i] = pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
		}
	}

	//传入归一化坐标，传出网格索引 type：转换方式
	int GetGridIndexLeft(const Point2f &pt, int type) {
		int x = 0, y = 0;

		if (type == 1) {
			//返回向下取整的int
			x = floor(pt.x * mGridSizeLeft.width);
			y = floor(pt.y * mGridSizeLeft.height);

			if (y >= mGridSizeLeft.height || x >= mGridSizeLeft.width){
				return -1;
			}
		}

		if (type == 2) {
			x = floor(pt.x * mGridSizeLeft.width + 0.5);
			y = floor(pt.y * mGridSizeLeft.height);

			if (x >= mGridSizeLeft.width || x < 1) {
				return -1;
			}
		}

		if (type == 3) {
			x = floor(pt.x * mGridSizeLeft.width);
			y = floor(pt.y * mGridSizeLeft.height + 0.5);

			if (y >= mGridSizeLeft.height || y < 1) {
				return -1;
			}
		}

		if (type == 4) {
			x = floor(pt.x * mGridSizeLeft.width + 0.5);
			y = floor(pt.y * mGridSizeLeft.height + 0.5);

			if (y >= mGridSizeLeft.height || y < 1 || x >= mGridSizeLeft.width || x < 1) {
				return -1;
			}
		}

		return x + y * mGridSizeLeft.width;
	}

	/**
	 * 传入归一化坐标 返回所在右网格索引，和取左网格索引相比，没有那么多转换方式
	*/
	int GetGridIndexRight(const Point2f &pt) {
		int x = floor(pt.x * mGridSizeRight.width);
		int y = floor(pt.y * mGridSizeRight.height);

		return x + y * mGridSizeRight.width;
	}

	// Assign Matches to Cell Pairs 
	void AssignMatchPairs(int GridType);

	// Verify Cell Pairs
	void VerifyCellPairs(int RotationType);

	// Get Neighbor 9
	/**
	 * idx: 存数网格相邻关系矩阵的行数索引 相当于第几个网格
	 * GridSize: 图像划分成多少个网格
	 * return 某个网格对应的9个相邻网格的索引
	*/
	vector<int> GetNB9(const int idx, const Size& GridSize) {
		//初始化是-1
		vector<int> NB9(9, -1);
		//求网格行数索引
		int idx_x = idx % GridSize.width;
		//求网格列数索引
		int idx_y = idx / GridSize.width;

		for (int yi = -1; yi <= 1; yi++)
		{
			for (int xi = -1; xi <= 1; xi++)
			{	
				//求9个相邻网格的行列索引（包括自己）
				int idx_xx = idx_x + xi;
				int idx_yy = idx_y + yi;
				//边缘网格处理
				if (idx_xx < 0 || idx_xx >= GridSize.width || idx_yy < 0 || idx_yy >= GridSize.height)
					continue;
				//存网格索引（图像中的第几个网格） 先存完一行，再存下一行
				/**
				 * 某个网格相邻网格索引存储到NB中的对应关系
				 * NB[1]|NB[2]|NB[3]
				 * NB[4]|NB[5]|NB[6]
				 * NB[7]|NB[8]|NB[9]
				*/
				NB9[xi + 4 + yi * 3] = idx_xx + idx_yy * GridSize.width;
			}
		}
		return NB9;
	}

	/**
	 * neighbor:存网格相邻关系的矩阵 每个网格是一行 9列：9个邻接矩阵
	 * GridSize:网格尺寸
	*/
	void InitalizeNiehbors(Mat &neighbor, const Size& GridSize) {
		//遍历每个网格
		for (int i = 0; i < neighbor.rows; i++)
		{
			//获取该行对应网格的9个邻居矩阵
			vector<int> NB9 = GetNB9(i, GridSize);
			//将Mat第i行指针(uchar *)转成(int *)赋值给data
			int *data = neighbor.ptr<int>(i);
			//把NB9中的索引存入Mat
			memcpy(data, &NB9[0], sizeof(int) * 9);
		}
	}

	//设置匹配器的尺度变换 把左图网格按照尺度比例赋值给右图网格
	void SetScale(int Scale) {
		// Set Scale
		mGridSizeRight.width = mGridSizeLeft.width  * mScaleRatios[Scale];
		mGridSizeRight.height = mGridSizeLeft.height * mScaleRatios[Scale];
		mGridNumberRight = mGridSizeRight.width * mGridSizeRight.height;

		// Initialize the neihbor of right grid 
		//初始化右图网格相邻关系矩阵
		mGridNeighborRight = Mat::zeros(mGridNumberRight, 9, CV_32SC1);
		InitalizeNiehbors(mGridNeighborRight, mGridSizeRight);
	}

	// Run 
	int run(int RotationType);
};

/**
 * vbInliers：传引用，存储内点匹配关系
 * WithScale：是否使用不同尺度匹配
 * WithRotation：是否使用旋转匹配
 * return：返回内点数量
*/
//函数的作用是获取内点，即根据一些条件判断特征点是否为内点，并将结果保存在 vbInliers 中。
int gms_matcher::GetInlierMask(vector<bool> &vbInliers, bool WithScale, bool WithRotation) {

	int max_inlier = 0;
	//只进行尺度不变的匹配
	if (!WithScale && !WithRotation)
	{	
		//右图网格和左图一样
		SetScale(0);
		//执行匹配过程并返回内点的数量
		max_inlier = run(1);
		//如果是经过验证的特征点对匹配关系，对应位置是true vector每个位置对应一个匹配特征点对
		vbInliers = mvbInlierMask;
		return max_inlier;
	}

	//尺度 旋转都要用
	if (WithRotation && WithScale)
	{
		//遍历五种尺度 八种旋转
		for (int Scale = 0; Scale < 5; Scale++)
		{
			SetScale(Scale);
			for (int RotationType = 1; RotationType <= 8; RotationType++)
			{
				int num_inlier = run(RotationType);
				
				//选内点最多的返回，但是成员变量是最后一次的？？？？
				if (num_inlier > max_inlier)
				{
					vbInliers = mvbInlierMask;
					max_inlier = num_inlier;
				}
			}
		}
		return max_inlier;
	}

	if (WithRotation && !WithScale)
	{
		SetScale(0);
		for (int RotationType = 1; RotationType <= 8; RotationType++)
		{
			int num_inlier = run(RotationType);

			if (num_inlier > max_inlier)
			{
				vbInliers = mvbInlierMask;
				max_inlier = num_inlier;
			}
		}
		return max_inlier;
	}

	if (!WithRotation && WithScale)
	{
		for (int Scale = 0; Scale < 5; Scale++)
		{
			SetScale(Scale);

			int num_inlier = run(1);

			if (num_inlier > max_inlier)
			{
				vbInliers = mvbInlierMask;
				max_inlier = num_inlier;
			}
			
		}
		return max_inlier;
	}

	return max_inlier;
}

/**
 * GridType：决定取网格索引的转换方式
 * 将匹配点对分配到对应的网格单元，更新mMotionStatistics矩阵和mNumberPointsInPerCellLeft ector
*/
void gms_matcher::AssignMatchPairs(int GridType) {
	//遍历所有匹配点对
	for (size_t i = 0; i < mNumberMatches; i++)
	{
		//取归一化后的坐标
		Point2f &lp = mvP1[mvMatches[i].first];
		Point2f &rp = mvP2[mvMatches[i].second];

		//取所在左网格索引 原mvMatches中存储的就是特征点索引，这里重新计算一下索引
		int lgidx = mvMatchPairs[i].first = GetGridIndexLeft(lp, GridType);
		int rgidx = -1;

		if (GridType == 1)
		{	
			//重新计算右索引
			rgidx = mvMatchPairs[i].second = GetGridIndexRight(rp);
		}
		else
		{
			//不重新计算右索引，直接取出索引值
			rgidx = mvMatchPairs[i].second;
		}

		if (lgidx < 0 || rgidx < 0)	continue;

		/**
		 * 	//这是一个 OpenCV 的矩阵（Mat），用于存储匹配点对在网格中的统计信息 Mat mMotionStatistics;
		*/
		//Mat行数对应左网格索引 列数对应右网格索引 如果两个网格间有特征点对，把该位置+1，记录左右网格匹配点对的数量
		mMotionStatistics.at<int>(lgidx, rgidx)++;
		//这个 vector 存储了左图像中每个网格中包含的特征点数量 vector<int> mNumberPointsInPerCellLeft;
		//记录对应网格里的特征点数量
		mNumberPointsInPerCellLeft[lgidx]++;
	}

}

/**
 * 用于验证每个左图网格单元与右图网格单元的匹配关系是否可靠，并进行筛选
 * RotationType：旋转类型，用于选择旋转模式。GMS 算法定义了 8 种旋转模式，取值范围为 1 到 8  对应mRotationPatterns（二维数组）中的8个3X3矩阵 矩阵意义？？？？？
*/
void gms_matcher::VerifyCellPairs(int RotationType) {
	//首先根据给定的旋转模式 RotationType，获取对应的旋转模式数据，存储在指针 CurrentRP 中
	const int *CurrentRP = mRotationPatterns[RotationType - 1];
	//遍历左图网格
	for (int i = 0; i < mGridNumberLeft; i++)
	{	
		//对于每个左图网格单元，检查其对应的右图网格单元是否存在匹配关系。若该左图网格单元内没有匹配点对，则将其右图网格单元索引 mCellPairs[i] 置为 -1，并跳过继续处理下一个网格单元。
		if (sum(mMotionStatistics.row(i))[0] == 0)
		{
			//这个 vector 存储了每个左图像网格与其对应的右图像网格的索引 vector<int> mCellPairs;
			mCellPairs[i] = -1;
			continue;
		}

		/*对于有匹配关系的左图网格单元，遍历所有右图网格单元，找到与该左图网格单元匹配的右图网格单元，
		即具有最大匹配点对数量的右图网格单元。*/
		int max_number = 0;
		//遍历右网格
		for (int j = 0; j < mGridNumberRight; j++)
		{
			//返回Mat第i行指针 int *类型
			int *value = mMotionStatistics.ptr<int>(i);
			//第i行 j列为1代表 左第i网格 右第j网格有匹配点对
			if (value[j] > max_number)
			{
				//这个 vector 存储了每个左图像网格的最匹配右图像网格的索引 匹配点对最多的右网格是与左网格最匹配的右网格 vector<int> mCellPairs;
				mCellPairs[i] = j;
				//存储当前左网格和右网格最大的匹配点数
				max_number = value[j];
			}
		}

		//取左网格对应的最匹配右网格索引
		int idx_grid_rt = mCellPairs[i];

		//传入左网格索引，获取左网格邻接的九个网格索引
		const int *NB9_lt = mGridNeighborLeft.ptr<int>(i);
		//传入右网格索引，获取右网格邻接的九个网格索引，
		const int *NB9_rt = mGridNeighborRight.ptr<int>(idx_grid_rt); 

		int score = 0;
		double thresh = 0;
		int numpair = 0;

		//遍历九个相邻网格
		for (size_t j = 0; j < 9; j++)
		{
			//左网格索引
			int ll = NB9_lt[j];
			//CurrentRP：旋转模式数据
			int rr = NB9_rt[CurrentRP[j] - 1];
			//NB9初始化是-1，这里判断如果是-1说明到了边缘，没有对应的相邻矩阵
			if (ll == -1 || rr == -1)	continue;

			//累加左右相邻网格匹配点对数
			score += mMotionStatistics.at<int>(ll, rr);
			//这个 vector 存储了左图像中每个网格中包含的特征点数量 vector<int> mNumberPointsInPerCellLeft;
			//累加左相邻网格特征点数目
			thresh += mNumberPointsInPerCellLeft[ll];
			//如果左右相邻网格有匹配关系，网格匹配数+1
			numpair++;
		}

		//文件开头宏定义 #define THRESH_FACTOR 6
		//为什么是这个公式？？？？？
		/**
		 * 在GMS（Grid-based Motion Statistics）匹配算法中，旋转模式用于对匹配的旋转进行不同角度的验证。GMS算法是一种基于网格的特征点匹配方法，
		 * 它利用特征点在网格内的分布情况来计算匹配的可靠性，并通过多个旋转模式进行验证，提高匹配的准确性和鲁棒性。
		 * 旋转模式是一组不同的旋转角度，通常是8个角度，用于对匹配点对进行旋转验证。GMS算法假设匹配的特征点对在一定程度上具有旋转不变性，即它们在不同的旋转角度下仍然可以保持匹配关系。
		 * 因此，通过在不同旋转角度下验证匹配点对的一致性，可以提高匹配的可靠性。
		 * 在函数VerifyCellPairs中，通过循环遍历不同的旋转模式，并计算每个模式下的得分（score），然后根据得分与阈值的比较，对匹配点对进行验证和筛选。如果得分小于阈值（thresh），
		 * 则认为该匹配点对不可靠，将其标记为 -2，否则保留该匹配点对。
		 * 旋转模式的使用可以使GMS算法对旋转变换具有一定的适应性，从而提高在具有旋转变换的图像匹配任务中的表现。同时，旋转模式的设置也可以根据具体应用场景进行调整，以达到最佳的匹配效果。
		 * 
		 * 平方根操作会对值进行平滑，并且可以将大范围的值缩小到一个较小的范围内。这样做的目的是将计算得到的平均匹配点对数量缩小，使得阈值范围更加合适。
		 * 这种缩小的效果有助于提高算法对不同场景和数据的适应性。
		 * 通过开平方，可以使得阈值在不同数据集上变得更加一致，避免了过度拟合或者局限于某个具体场景的情况。同时，开平方还有助于处理可能出现的异常值，
		 * 降低其对阈值计算的影响，增强了算法的鲁棒性。
		*/
		thresh = THRESH_FACTOR * sqrt(thresh / numpair);

		if (score < thresh)
		//匹配关系不可靠，左网格匹配的右网格索引置-2 表示匹配关系被筛选掉了
			mCellPairs[i] = -2;
	}//结束遍历左网格
}

//执行匹配过程并返回内点的数量 内点就是经过验证 匹配关系可靠的点，外点是匹配关系不可靠（误匹配的点）
int gms_matcher::run(int RotationType) {
	//mvbInlierMask定义：vector<bool> mvbInlierMask;
	//mNumberMatches：匹配关系的数量
	//mvbInlinerMask里替换成mNumberMatches个false
	mvbInlierMask.assign(mNumberMatches, false);

	/*接下来，初始化运动统计矩阵mMotionStatistics，用于存储左图网格和右图网格之间的运动统计信息。
	同时，初始化一个包含匹配对（match pairs）的向量mvMatchPairs，其中每个元素是一个存储左图网格索引和右图网格索引的二元组。*/
	// Initialize Motion Statisctics
	mMotionStatistics = Mat::zeros(mGridNumberLeft, mGridNumberRight, CV_32SC1);
	//这个 vector 存储了匹配点对所在的网格索引对。vector<pair<int, int> > mvMatchPairs;
	mvMatchPairs.assign(mNumberMatches, pair<int, int>(0, 0));

	for (int GridType = 1; GridType <= 4; GridType++) 
	{
		// initialize
		//所有值置0
		mMotionStatistics.setTo(0);
		//这个 vector 存储了每个左图像网格与其对应的右图像网格的索引 vector<int> mCellPairs; 置-1
		mCellPairs.assign(mGridNumberLeft, -1);
		//这个 vector 存储了左图像中每个网格中包含的特征点数量 vector<int> mNumberPointsInPerCellLeft;
		mNumberPointsInPerCellLeft.assign(mGridNumberLeft, 0);
		
		//更新mMotionStatistics和mNumberPointsInPerCellLeft
		AssignMatchPairs(GridType);
		//按照所选的旋转模式验证匹配关系，得分低的匹配关系筛掉 mCellPairs中剩下的是经过验证的网格匹配关系
		VerifyCellPairs(RotationType);

		// Mark inliers
		//遍历所有匹配特征点对
		for (size_t i = 0; i < mNumberMatches; i++)
		{	
			//这个 vector 存储了匹配点对所在的网格索引对。vector<pair<int, int> > mvMatchPairs;
			if (mvMatchPairs[i].first >= 0) {
				if (mCellPairs[mvMatchPairs[i].first] == mvMatchPairs[i].second)//如果是经过验证的网格匹配关系
				{
					//如果是经过验证的特征点对匹配关系，对应位置是true vector每个位置对应一个匹配特征点对
					mvbInlierMask[i] = true;
				}
			}
		}//结束遍历所有匹配特征点对
	}//为什么每个GridType都算一遍？？？？
	//计算内点数量，有没有可能一个特征点对应两个特征点？？？应该没有 这里的[0]是因为调用的sum是OpenCV函数，返回值是cv::Scalar
	int num_inlier = sum(mvbInlierMask)[0];
	//返回内点数量
	return num_inlier;
}