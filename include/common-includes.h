#pragma once
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include  <iostream>
#include <string>
#include <eigen3/Eigen/Core>
#include <sstream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <set>
#include <cmath>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_base.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
namespace fs = std::filesystem;