#pragma once
#include  <iostream>
#include <string>
#include <eigen3/Eigen/Core>
#include <sstream>
#include <fstream>
#include <vector>
#ifdef USE_NEW_FILESYSTEM_HEADER
    #include <filesystem>
#else
    #include <experimental/filesystem>
#endif
#include <set>
#include <cstdlib>
#include <map>
#include <cvsba/cvsba.h>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
#ifdef USE_NEW_FILESYSTEM_HEADER
namespace fs = std::filesystem;
#else
namespace fs = std::experimental::filesystem;
#endif