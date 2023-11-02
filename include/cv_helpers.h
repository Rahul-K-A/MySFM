#pragma once
#include "common-includes.h"
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <cmath>



/// @brief Basic point cloud structure. Contains the location, color, and feature correspondence for a 3D point 
struct DataPoint {
    cv::Point3f point;
    vector<int> keypoint_index;
    cv::Vec3b color;
};

/// @brief Basic camera information structure. Contains the image, keypoints and descriptors, pose, and projection matrix data for each camera view
struct CameraState{
    cv::Mat Image;
    cv::Mat Descriptor;
    vector<cv::KeyPoint> keypoints;
    cv::Mat R;
    cv::Mat t;
    cv::Mat P;
};



namespace cvHelpers{

/// @brief Reads a set of images and calculates the keypoints and descriptors using surf
/// @param image_paths Set of image paths  
void readImages(set<fs::path>& image_paths);

/// @brief Function to calculate correspondence data between the images
void calculateFeatureCorrespondance();

/// @brief Function to calculate the view with the best match for the given input view from already evaluated views
/// @param view_to_eval Input view to find correspondence for
/// @return The view with the best correspondence to the image
int find_best_correspondence(int view_to_eval);

/// @brief Given a DMatch vector, creates a new DMatch vector where the query and train indices are flipped
/// @param matches_vector Input DMatch vector  
/// @return The flipped vector
vector<cv::DMatch> flip_match(const vector<cv::DMatch>& matches_vector);

/// @brief Given 2 views, finds the common features and corresponding points in the global pointcloud
/// @param view1 First view
/// @param view2 Second view
/// @param imagePoints Projections of common features in view 2
/// @param ppoint_cloud 3D location of feature points
/// @param kp1 Common feature keypoints of view 1
/// @param kp2 Common feature keypoints of view 2
/// @param kpidx1 Vector containing the indices of kp1 in CameraState[view1]
/// @param kpidx2 Vector containing the indices of kp2 in cameraSate[view2]
void find_2d_3d_matches(int view1, int view2, vector<cv::Point2f>& imagePoints, vector<cv::Point3f>& ppoint_cloud, vector<cv::KeyPoint>& kp1, vector<cv::KeyPoint>& kp2, vector<int>& kpidx1, vector<int>& kpidx2);

/// @brief Adds a set of points to the global point cloud
/// @param prevView View 1 which was used to triangulate the points
/// @param currentView View 2 used for triangulation
/// @param pc_to_add Point cloud to be added 
void addToGlobalPC(int prevView, int currentView, vector<DataPoint>& pc_to_add);

/// @brief Computes the initial point cloud
void computeFirstPointCloud();

/// @brief Initialises SURF and feature matching
void initFeatureMatching();

/// @brief Perform Structure-from-Motion
void sfm();

/// @brief Get the current global point cloud data
/// @return Datapoint vector containing the global PC
vector<DataPoint>& getGlobalPC(); 

/// @brief Perform bundle adjustment to get more accurate results
void performBA();

}