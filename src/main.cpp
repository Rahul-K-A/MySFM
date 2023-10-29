#include "common-includes.h"
#include "cv_helpers.h"
#include "pcl_helpers.h"



const float dThreshold = 1.f;
/*Ratio thresh needs to be kept around 0.5 to produce good BA values*/
const float ratio_thresh = 0.8f;
const float acceptable_error = 20.f;

Mat K = (Mat1d(3,3) <<      2759.48, 0, 1520.69, 
                            0, 2764.16, 1006.81, 
                            0, 0, 1); 


vector<DataPoint> point_cloud; //our global 3D point cloud
vector<CameraState> cameraStates;
map<int, vector<vector<DMatch>> > match_table;
vector<int> done_views;
Ptr<SURF> surf;
Ptr<DescriptorMatcher> matcher;
cvsba::Sba BA; 
cvsba::Sba::Params params ;

void performBA()
{
    std::vector<cv::Point3f> points;
    std::vector<std::vector<cv::Point2f> > imagePoints;
    std::vector<std::vector<int> > visibility_mask;
    std::vector<cv::Mat> cameraMatrix = vector<Mat>(done_views.size(), K);
    std::vector<cv::Mat> R;
    std::vector<cv::Mat> T;
    std::vector<cv::Mat> distCoeffs = vector<Mat>(done_views.size(), ( Mat1d(1,5) << 0,0,0,0,0 )  );

    for(const DataPoint& cloud_point : point_cloud)
    {
        points.push_back(cloud_point.point);
    }
    
    for(int view : done_views)
    {
        vector<Point2f> current_view_points;
        vector<int> current_visibility;
        for(uint32_t idx = 0; idx < point_cloud.size(); idx++)
        {
                        int kpidx = point_cloud[idx].keypoint_index[view];
            if(kpidx != -1)
            {
                current_view_points.push_back(cameraStates[view].keypoints[kpidx].pt);
                current_visibility.push_back(1);
            }
            else{
                current_view_points.push_back(Point2f(0,0));
                current_visibility.push_back(0);
            }
        }
        
        imagePoints.push_back(current_view_points);
        visibility_mask.push_back(current_visibility);
        R.push_back(cameraStates[view].R.clone());
        T.push_back(cameraStates[view].t.clone());
    }

    params.type = cvsba::Sba::MOTIONSTRUCTURE;
    params.iterations = 500;
    params.minError = 1e-7;
    params.fixedIntrinsics = 5;
    params.fixedDistortion = 5;
    params.verbose = true;
    BA.setParams(params);

    cout << points.size() << endl;
    cout <<  "imagePoints :"<< imagePoints.size() << endl;
    cout << "R :"<< R.size() << endl;
    cout <<  "T :"<< T.size() << endl;

    BA.run( points, imagePoints, visibility_mask, cameraMatrix,  R, T, distCoeffs);
    for(uint32_t idx = 0; idx < point_cloud.size(); idx++)
    {
        point_cloud[idx].point = points[idx];
    }


    for(uint16_t i = 0; i < done_views.size(); i++)
    {
        int curr_idx = done_views[i];
        R[i].convertTo(cameraStates[curr_idx].R, CV_32F);
        T[i].convertTo(cameraStates[curr_idx].t, CV_32F);
        Mat transform;
        hconcat(R[i], T[i], transform);
        cameraStates[curr_idx].P = K * transform;
        cameraStates[curr_idx].P.convertTo(cameraStates[curr_idx].P, CV_32F);

    }

    std::cout << "Optimization. Initial error=" << BA.getInitialReprjError() << " and Final error=" << BA.getFinalReprjError() << std::endl;
}

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

    set<fs::path> image_paths;
    string dataset_image_path = dataset_source_dir + string("/images");
    for (const auto & entry : fs::directory_iterator(dataset_image_path))
    {
        image_paths.insert(entry.path());
    }
    
    readImages(image_paths);
    calculateImageFeatures();

    computeFirstPointCloud();

    for(uint16_t current_view = 2; current_view < cameraStates.size(); current_view++ )
    {
        cout << "Calculating image number: " << current_view << endl;
        std::vector<cv::Point3f> ppoint_cloud;           
        std::vector<cv::Point2f> imgPoints;
        vector<KeyPoint> kp1, kp2;
        vector<int> kpidx1, kpidx2;
        int best_match_view = find_best_correspondance(current_view);
        cout << "Found best correspondance: " << best_match_view << endl;
        find_2d_3d_matches(best_match_view, current_view, imgPoints, ppoint_cloud, kp1, kp2, kpidx1, kpidx2);
        cv::Mat t,rvec,R;
        assert(ppoint_cloud.size() == imgPoints.size());
        cv::solvePnPRansac(ppoint_cloud, imgPoints, K, noArray(), rvec, t, false);
        //get rotation in 3x3 matrix form
        Rodrigues(rvec, R);
        cameraStates[current_view].R = R.clone();
        cameraStates[current_view].t = t.clone();

        Mat P, currpose, homogenised_3d_points;
        hconcat(R, t, currpose);
        cout << "Pose of "<< current_view <<" is " << currpose <<endl;
        P = K * currpose;
        P.convertTo(P, CV_32F);
        vector<Point2f> prevpts, currpts;
        KeyPoint::convert(kp1, prevpts);
        KeyPoint::convert(kp2, currpts);
        cout<<"Triangulate Points\n";
        triangulatePoints(cameraStates[best_match_view].P,  P, prevpts ,currpts, homogenised_3d_points );
        cameraStates[current_view].P = P.clone();
        assert(homogenised_3d_points.type() == CV_32F);
        vector<Point3f> points_3f;
        vector<Point2f> reprojected_points;
        vector<float> reprojection_error;
        float average_reprojection_error = 0;
        float distance; 
        for(uint16_t col = 0; col < homogenised_3d_points.cols; col++)
        {
            Mat currPoint3d =  homogenised_3d_points.col(col);
            currPoint3d /= currPoint3d.at<float>(3, 0);
            Point3f p(
                currPoint3d.at<float>(0, 0),
                currPoint3d.at<float>(1, 0),
                currPoint3d.at<float>(2, 0)
            );
            points_3f.push_back(p);
        }

        projectPoints(points_3f, R, t, K, noArray(), reprojected_points);
        assert(currpts.size() == reprojected_points.size());
        assert(kpidx1.size() == reprojected_points.size());
        cout<<"Calculating reprojection error...\n";
        vector<DataPoint> pc_to_add;
        for(uint16_t pt_idx = 0; pt_idx  < currpts.size(); pt_idx++)
        {
            distance = dist(currpts[pt_idx], reprojected_points[pt_idx]);
            average_reprojection_error +=distance;
            if ( distance < acceptable_error) 
            {
                DataPoint data_point;
                data_point.keypoint_index = vector<int>(cameraStates.size(), -1);
                data_point.point = points_3f[pt_idx]; 
                data_point.keypoint_index[best_match_view] = kpidx1[pt_idx];
                data_point.keypoint_index[current_view] = kpidx2[pt_idx];
                data_point.color = cameraStates[best_match_view].Image.at<Vec3b>( cameraStates[best_match_view].keypoints[kpidx1[pt_idx]].pt);
                pc_to_add.push_back(data_point);
            }
        }


        addToGlobalPC(best_match_view, current_view, pc_to_add);

        average_reprojection_error = average_reprojection_error/currpts.size();
        cout<<"Average RE: " << average_reprojection_error << endl;
        cout<<endl<<endl;



        done_views.push_back(current_view);
        performBA();

    }























    cout << "Creating PCL Point Cloud...\n";
    //Convert Point3D into PointXYZRGB
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    for(uint16_t i = 0; i < point_cloud.size(); i++)
    {
        pcl::PointXYZRGB basic_point;
        basic_point.x = -1.f * point_cloud[i].point.x;
        basic_point.y = -1.f * point_cloud[i].point.y;
        basic_point.z = point_cloud[i].point.z;
        Vec3b color = point_cloud[i].color;
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
    if (waitKey(0) == 27)
    {
        destroyAllWindows();
        return 1;
    }
}