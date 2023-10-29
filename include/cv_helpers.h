#pragma once
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <cmath>


/// @brief Basic point cloud structure. Contains the location, color, and feature correspondence for a 3D point 
struct DataPoint {
    Point3f point;
    vector<int> keypoint_index;
    Vec3b color;
};

/// @brief Basic camera information structure. Contains the image, keypoints and descriptors, pose, and projection matrix data for each camera view
struct CameraState{
    Mat Image;
    Mat Descriptor;
    vector<KeyPoint> keypoints;
    Mat R;
    Mat t;
    Mat P;
};



class Point2f;

/// @brief Helper function to find the euclidean distance between two points
/// @param a First Point
/// @param b Second point
/// @return The distance between the 2 points
float dist(Point2f a, Point2f b)
{
   float dist =  (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
   dist = sqrtf(dist);
   return dist;
};

/// @brief Reads a set of images and calculates the keypoints and descriptors using surf
/// @param image_paths Set of image paths  
/// @param cameraStates Output vector in which the data is stored
void readImages(set<fs::path>& image_paths, vector<CameraState>& cameraStates);

/// @brief Function to calculate correspondence data between the images
/// @param cameraStates 
void calculateFeatureCorrespondance(vector<CameraState>& cameraStates);

/// @brief Function to calculate the view with the best match for the given input view from already evaluated views
/// @param view_to_eval Input view to find correspondence for
/// @return The view with the best correspondence to the image
int find_best_correspondence(int view_to_eval);

vector<DMatch> flip_match(const vector<DMatch>& matches_vector);
void find_2d_3d_matches(int view1, int view2, vector<Point2f>& imagePoints, vector<Point3f>& ppoint_cloud, vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, vector<int>& kpidx1, vector<int>& kpidx2);
void addToGlobalPC(int prevView, int currentView, vector<DataPoint>& pc_to_add);