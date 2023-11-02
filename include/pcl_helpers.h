#pragma once
#include "common-includes.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_base.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include "cv_helpers.h"

namespace pclHelpers{
void savePointCloud_to_PCD(vector<DataPoint>& point_cloud);
}