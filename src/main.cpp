#include "VisualOdometry.h"
#define scaler 1;

/*
TODO:
    1. Redo how relative scale works (need informtaion from 3 consecutive frames. aka 2 consecutive frame pairs) 
    2. https://research.latinxinai.org/papers/cvpr/2021/pdf/59_CameraReady_59.pdf refer for using absolute scale
*/

Mat cRot = Mat::eye(3, 3, CV_64F);
Mat cT = ( Mat1d(3,1) << 0.f , 0.f , 0.f );
   Mat K = (Mat1d(3,3) << 2759.48, 0, 1520.69, 
                            0, 2764.16, 1006.81, 
                            0, 0, 1); 

int main(int argc, const char **argv)
{
    if(CV_VERSION_MAJOR < 4)
    {
        cout<<"This project was built using OpenCV 4. If you are using older versions, please update to a newer version \n";
        assert(CV_VERSION_MAJOR >= 4);
    }

    if(argc != 2)
    {
        cout<<"Wrong Command given. Please use ./VisualOdometry <path to dataset> <Camera FOV type>\n";
        cout<<"Camera FOV dtypes: WIDE = 1, NARROW = 2\n";
        return 0;
    }
    
    string dataset_source_dir =  string(argv[1]);
    if(dataset_source_dir.back() == '/')
    {
        dataset_source_dir.pop_back();
    }

    string dataset_image_path = dataset_source_dir + string("/images");
    for (const auto & entry : fs::directory_iterator(dataset_image_path))
    {
        im_paths.insert(entry.path());
    }
    auto begin = im_paths.begin();
    auto next = std::next(begin,1);
    Ptr<SURF> surf = SURF::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    Mat img1c =  imread( "/home/rahul/rahul_sw/SfM_quality_evaluation/Benchmarking_Camera_Calibration_2008/fountain-P11/images/0000.jpg" );
    Mat img2c =  imread( "/home/rahul/rahul_sw/SfM_quality_evaluation/Benchmarking_Camera_Calibration_2008/fountain-P11/images/0001.jpg" );
    // ImageCalibrator* iCalibrator = new ImageCalibrator(dataset_source_dir, camType, calibSettings);
    Mat img1 = imread( "/home/rahul/rahul_sw/SfM_quality_evaluation/Benchmarking_Camera_Calibration_2008/fountain-P11/images/0000.jpg", IMREAD_GRAYSCALE );
    Mat img2 = imread( "/home/rahul/rahul_sw/SfM_quality_evaluation/Benchmarking_Camera_Calibration_2008/fountain-P11/images/0001.jpg", IMREAD_GRAYSCALE );

    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;
    surf->detectAndCompute(img1c, noArray(), kp1, desc1);
    surf->detectAndCompute(img2c, noArray(), kp2, desc2);


    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(desc1, desc2, knn_matches, 2);
    const float ratio_thresh = 0.4f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    cout << good_matches.size() << endl;

    vector<Point2f> p2d1, p2d2;
    for(uint16_t i = 0; i < good_matches.size(); i++)
    {
        p2d1.push_back( kp1[good_matches[i].queryIdx].pt );
        p2d2.push_back( kp2[good_matches[i].trainIdx].pt );
    }

    Mat mask;
 
    Mat E = findEssentialMat(p2d1, p2d2, K , RANSAC, 0.997, 1.0, mask);

    vector<cv::Point2f> inlier_match_points1, inlier_match_points2;
    for(int i = 0; i < mask.rows; i++) {
        if(mask.at<unsigned char>(i)){
            inlier_match_points1.push_back(p2d1[i]);
            inlier_match_points2.push_back(p2d2[i]);
        }
    }
    Mat R,t;
    mask.release();
    recoverPose(E, inlier_match_points1, inlier_match_points2, K, R, t, mask);

    if( (abs(determinant(R)) - 1.f) > 1e-7 )
    {
        cout<<"Fucked up\n";
    }
    vector<cv::Point2f> triangulation_points1, triangulation_points2;
    for(int i = 0; i < mask.rows; i++) {
        if(mask.at<unsigned char>(i)){

            triangulation_points1.push_back (inlier_match_points1[i]);
            triangulation_points2.push_back(inlier_match_points2[i]);
        
        }
    }


    cout << R.type() << endl << t.type() <<endl;
    Mat rw1, rw2;
    hconcat(cRot, cT, rw1);
    hconcat(R.inv() ,t * -1.0 , rw2);
    Mat matched_img;
    drawMatches(img1, kp1, img2, kp2, good_matches, matched_img,  Scalar::all(-1),
 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    Mat homogenised_3d_points;
    Mat Kf;
    K.convertTo(Kf, CV_32F);
    vector<Point3f> points_3f;
    Mat P = K * rw1;
    Mat P1 = K * rw2;
    cout << P.type() << endl;
    cout << P1.type() << endl;
    P.convertTo(P, CV_32F);
    P1.convertTo(P1, CV_32F);
        cout << P.type() << endl;
    cout << P1.type() << endl;

    cout << R<< endl;
    cout << t << endl;
    cout << rw2 << endl;
    triangulatePoints(P, P1, triangulation_points1, triangulation_points2, homogenised_3d_points );
    cout << homogenised_3d_points.type() << endl;

    for(uint16_t i = 0; i < homogenised_3d_points.cols; i++)
    {
        Mat currPoint3d =  homogenised_3d_points.col(i);
        currPoint3d /= currPoint3d.at<float>(3, 0);
        Point3f p(
            (float)currPoint3d.at<float>(0, 0),
            (float)currPoint3d.at<float>(1, 0),
            (float)currPoint3d.at<float>(2, 0)
        );
        points_3f.push_back(p);
    }
    //Convert Point3D into PointXYZRGB
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    for(uint16_t i = 0; i < points_3f.size(); i++)
    {
        pcl::PointXYZRGB basic_point;
        basic_point.x =(float) points_3f[i].x;
        basic_point.y =(float) points_3f[i].y;
        basic_point.z = (float)points_3f[i].z;
        // if(points_3f[i].z > 1e20)
        // {
        //     basic_point.z = 14.f;
        // }
        // else{
        //     basic_point.z = points_3f[i].z * 1e5;
        // }
        Vec3b color = img1c.at<Vec3b>(triangulation_points1[i]);
        basic_point.b = color[0];
        basic_point.g = color[1];
        basic_point.r = color[2];
        //cout<<"Basic point: "<< basic_point << endl; 
        point_cloud_ptr->points.push_back(basic_point);
    }

    point_cloud_ptr->width = (int)point_cloud_ptr->points.size();
    point_cloud_ptr->height = 1;
    //Since the point cloud is sparse, remove all the NaN values automatically set by PCL
    vector<int> indices;
    point_cloud_ptr->is_dense = false;

    pcl::io::savePCDFileASCII("final.pcd", *point_cloud_ptr);
    resize(matched_img, matched_img, Size(matched_img.cols/2, matched_img.rows/2));
    imshow("matched_img", matched_img);
    if (waitKey(0) == 27)
    {
        destroyAllWindows();
        return 1;
    }
}