#ifndef ORB_PARAMETER_HPP_
#define ORB_PARAMETER_HPP_

#include "common_include.h"

namespace ORB {
    class BaseParameter{
        public:
            BaseParameter(const std::string config_file);

            void ReadPara();

        public:
            std::string file_path;
            double fx, fy, cx, cy;  // 相机内参
            double k1, k2, p1, p2;  // 相机畸变参数
    };

    class Parameter: public BaseParameter {
        public:
            Parameter(const std::string config_file);

            void ReadOtherPara();

        public:
            int nFeatures;       ///<整个图像金字塔中，要提取的特征点数目
            double scalseFactor; ///<图像金字塔层与层之间的缩放因子
            int nLevels;         ///<图像金字塔的层数
            int iniThFAST;       ///<初始的FAST响应值阈值
            int minThFAST;       ///<最小的FAST响应值阈值
    };
    
}


#endif