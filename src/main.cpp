#include "common-includes.h"

#define scaler 1;

/*
TODO:
    1. Redo how relative scale works (need informtaion from 3 consecutive frames. aka 2 consecutive frame pairs) 
    2. https://research.latinxinai.org/papers/cvpr/2021/pdf/59_CameraReady_59.pdf refer for using absolute scale
*/

float dist(Point2f a, Point2f b)
{
   float dist =  (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
   dist = sqrtf(dist);
   return dist;
}


const float dThreshold = 1.f;
Mat cRot = Mat::eye(3, 3, CV_64F);
Mat cT = ( Mat1d(3,1) << 0.f , 0.f , 0.f );
Mat K = (Mat1d(3,3) << 2759.48, 0, 1520.69, 
                            0, 2764.16, 1006.81, 
                            0, 0, 1); 

struct CloudPoint {
cv::Point3d pt;
vector<int>index_of_2d_origin;
};


struct CamState{
    Mat Image;
    Mat Descriptor;
    vector<KeyPoint> keypoints;
};

vector<CloudPoint> pcloud; //our global 3D point cloud
vector<CamState> cameraStates;




int main(int argc, const char **argv)
{
    if(CV_VERSION_MAJOR < 4)
    {
        cout<<"This project was built using OpenCV 4. If you are using older versions, please update to a newer version \n";
        assert(CV_VERSION_MAJOR >= 4);
    }

    if(argc != 2)
    {
        cout<<"Wrong Command given. Please use ./MySFM <path to dataset>\n";
        return 0;
    }
    
    string dataset_source_dir =  string(argv[1]);
    if(dataset_source_dir.back() == '/')
    {
        dataset_source_dir.pop_back();
    }

    set<fs::path> im_paths;
    string dataset_image_path = dataset_source_dir + string("/images");
    for (const auto & entry : fs::directory_iterator(dataset_image_path))
    {
        im_paths.insert(entry.path());
    }
    auto begin = im_paths.begin();
    auto next = std::next(begin,1);
    Ptr<SURF> surf = SURF::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    for(auto image_path : im_paths)
    {
        Mat img = imread(image_path.c_str());
        vector<KeyPoint> keypoint;
        Mat descriptor;
        CamState currCamState;
        img.copyTo(currCamState.Image);
        Mat img_gray;
        cvtColor(img, img_gray, COLOR_BGR2GRAY);
        surf->detectAndCompute(img_gray, noArray(), keypoint, descriptor);
        currCamState.keypoints = keypoint;
        descriptor.copyTo(currCamState.Descriptor);
        cameraStates.push_back(currCamState);
        
    }
    for(uint16_t i = 0; i < cameraStates.size(); i++)
    {
        cout << cameraStates[i].Descriptor << endl;
    }

    // vector<KeyPoint> kp1, kp2;
    // Mat desc1, desc2;
    // surf->detectAndCompute(img1c, noArray(), kp1, desc1);
    // surf->detectAndCompute(img2c, noArray(), kp2, desc2);


    // vector<vector<DMatch>> knn_matches;
    // matcher->knnMatch(desc1, desc2, knn_matches, 2);
    // const float ratio_thresh = 0.7f;
    // std::vector<DMatch> good_matches;
    // for (size_t i = 0; i < knn_matches.size(); i++)
    // {
    //     if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    //     {
    //         good_matches.push_back(knn_matches[i][0]);
    //     }
    // }
    // cout << good_matches.size() << endl;

    // vector<Point2f> p2d1, p2d2;
    // for(uint16_t i = 0; i < good_matches.size(); i++)
    // {
    //     p2d1.push_back( kp1[good_matches[i].queryIdx].pt );
    //     p2d2.push_back( kp2[good_matches[i].trainIdx].pt );
    // }

    // Mat mask;
 
    // Mat E = findEssentialMat(p2d1, p2d2, K , RANSAC, 0.999, 1.0, mask);

    // vector<cv::Point2f> inlier_match_points1, inlier_match_points2;
    // for(int i = 0; i < mask.rows; i++) {
    //     if(mask.at<unsigned char>(i)){
    //         inlier_match_points1.push_back(p2d1[i]);
    //         inlier_match_points2.push_back(p2d2[i]);
    //     }
    // }
    // Mat R,t;
    // mask.release();
    // recoverPose(E, inlier_match_points1, inlier_match_points2, K, R, t, mask);

    // if( (abs(determinant(R)) - 1.f) > 1e-7 )
    // {
    //     cout<<"Fucked up\n";
    // }
    // vector<cv::Point2f> triangulation_points1, triangulation_points2;
    // for(int i = 0; i < mask.rows; i++) {
    //     if(mask.at<unsigned char>(i)){

    //         triangulation_points1.push_back (inlier_match_points1[i]);
    //         triangulation_points2.push_back(inlier_match_points2[i]);
        
    //     }
    // }

    // cout << R.type() << endl << t.type() <<endl;
    // Mat rw1, rw2;
    // hconcat(cRot, cT, rw1);
    // hconcat(R ,t, rw2);
    // Mat matched_img;
    // drawMatches(img1c, kp1, img2c, kp2, good_matches, matched_img,  Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Mat homogenised_3d_points;

    // vector<Point3f> points_3f;
    // Mat P = K * rw1;
    // Mat P1 = K * rw2;
    // P.convertTo(P, CV_32F);
    // P1.convertTo(P1, CV_32F);

    // cout << t << endl;
    // triangulatePoints(P, P1, triangulation_points1, triangulation_points2, homogenised_3d_points );
    // cout <<"homogenised_3d_points.type() :" << homogenised_3d_points.type() << endl;

    // assert(homogenised_3d_points.type() == CV_32F);
    // for(uint16_t i = 0; i < homogenised_3d_points.cols; i++)
    // {
    //     Mat currPoint3d =  homogenised_3d_points.col(i);
    //     currPoint3d /= currPoint3d.at<float>(3, 0);
    //     Point3f p(
    //         currPoint3d.at<float>(0, 0),
    //         currPoint3d.at<float>(1, 0),
    //         currPoint3d.at<float>(2, 0)
    //     );
    //     points_3f.push_back(p);
    // }

    // vector<Point2f> reprojected_points, final_2d_points;
    // vector<Point3f> final_3d_points;
    // vector<float> reprojection_error;
    // vector<double> dummy;
    // projectPoints(points_3f, R, t, K, dummy, reprojected_points);
    // float average_reprojection_error = 0;
    // cout<< reprojected_points[0] <<"   " << triangulation_points2[0] <<  "\n\n\n\n\n\n";
    // float d;
    // for(uint16_t i = 0; i  < triangulation_points2.size(); i++)
    // {
    //     d = dist(triangulation_points2[i], reprojected_points[i]);
    //     average_reprojection_error +=d;
    //     if ( d < 1.f)
    //     {
    //         final_3d_points.push_back(points_3f[i]);
    //         final_2d_points.push_back(triangulation_points1[i]);
    //     }
    // }
    // average_reprojection_error = average_reprojection_error/triangulation_points2.size();
    // cout << "Average reprojection error: " << average_reprojection_error << endl;
    // d = 0; 
    // average_reprojection_error = 0;
























    // //Convert Point3D into PointXYZRGB
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    // for(uint16_t i = 0; i < final_3d_points.size(); i++)
    // {
    //     pcl::PointXYZRGB basic_point;
    //     basic_point.x = -1.f * final_3d_points[i].x;
    //     basic_point.y = -1.f * final_3d_points[i].y;
    //     basic_point.z = final_3d_points[i].z;
    //     Vec3b color = img1c.at<Vec3b>(final_2d_points[i]);
    //     basic_point.b = color[0];
    //     basic_point.g = color[1];
    //     basic_point.r = color[2];
    //     //cout<<"Basic point: "<< basic_point << endl; 
    //     point_cloud_ptr->points.push_back(basic_point);
    // }

    // point_cloud_ptr->width = (int)point_cloud_ptr->points.size();
    // point_cloud_ptr->height = 1;
    // //Since the point cloud is sparse, remove all the NaN values automatically set by PCL
    // vector<int> indices;
    // point_cloud_ptr->is_dense = false;

    // pcl::io::savePCDFileASCII("final.pcd", *point_cloud_ptr);
    // resize(matched_img, matched_img, Size(matched_img.cols/4, matched_img.rows/4));
    // imshow("matched_img", matched_img);
    // if (waitKey(0) == 27)
    // {
    //     destroyAllWindows();
    //     return 1;
    // }
}