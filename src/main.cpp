#include "common-includes.h"

float dist(Point2f a, Point2f b)
{
   float dist =  (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
   dist = sqrtf(dist);
   return dist;
}


const float dThreshold = 1.f;
const float ratio_thresh = 0.8f;
const float acceptable_error = 20.f;

Mat K = (Mat1d(3,3) << 2759.48, 0, 1520.69, 
                            0, 2764.16, 1006.81, 
                            0, 0, 1); 

struct DataPoint {
    Point3f point;
    vector<int> keypoint_index;
    Vec3b color;
};


struct CamState{
    Mat Image;
    Mat Descriptor;
    vector<KeyPoint> keypoints;
    Mat R;
    Mat t;
    Mat P;
};

vector<DataPoint> point_cloud; //our global 3D point cloud
vector<CamState> cameraStates;
map<int, vector<vector<DMatch>> > match_table;
vector<int> done_views;


void computeFirstPointCloud()
{
    cout <<"Evaluating images 0 and 1 \n";
    vector<Point2f> p2d1, p2d2;
    vector<int> indices1, indices2;
    vector<KeyPoint> kp1, kp2;
    kp1 = cameraStates[0].keypoints;
    kp2 = cameraStates[1].keypoints;

    vector<DMatch> good_matches = match_table[0][1];

    for(uint16_t i = 0; i < good_matches.size(); i++)
    {
        indices1.push_back(good_matches[i].queryIdx);
        indices2.push_back(good_matches[i].trainIdx);
        p2d1.push_back( kp1[good_matches[i].queryIdx].pt );
        p2d2.push_back( kp2[good_matches[i].trainIdx].pt );
    }

    Mat mask;
    cameraStates[0].R = Mat::eye(3, 3, CV_64F);
    cameraStates[0].t = ( Mat1d(3,1) << 0.f , 0.f , 0.f );
 
    Mat E = findEssentialMat(p2d1, p2d2, K , RANSAC, 0.999, 1.0, mask);
    vector<int> inlier_indices1, inlier_indices2;
    vector<cv::Point2f> inlier_match_points1, inlier_match_points2;
    for(int i = 0; i < mask.rows; i++) {
        if(mask.at<unsigned char>(i)){
            inlier_indices1.push_back(indices1[i]);
            inlier_indices2.push_back(indices2[i]);
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
    vector<int> triangulation_indices1, triangulation_indices2;
    vector<cv::Point2f> triangulation_points1, triangulation_points2;
    for(int i = 0; i < mask.rows; i++) {
        if(mask.at<unsigned char>(i)){
            triangulation_indices1.push_back(inlier_indices1[i]);
            triangulation_indices2.push_back(inlier_indices2[i]);
            triangulation_points1.push_back (inlier_match_points1[i]);
            triangulation_points2.push_back(inlier_match_points2[i]);
        
        }
    }

    cameraStates[1].R = R.clone();
    cameraStates[1].t = t.clone();
    cout << R.type() << endl << t.type() <<endl;
    Mat pose1, pose2;
    hconcat(cameraStates[0].R, cameraStates[0].t, pose1);
    hconcat(R ,t, pose2);
    cout<<"OG P is " << pose2 << endl;
    Mat matched_img;

    Mat homogenised_3d_points;

    vector<Point3f> points_3f;
    Mat P = K * pose1;
    Mat P1 = K * pose2;
    P.convertTo(P, CV_32F);
    cameraStates[0].P = P.clone();
    P1.convertTo(P1, CV_32F);
    cameraStates[1].P = P1.clone();

    triangulatePoints(P, P1, triangulation_points1, triangulation_points2, homogenised_3d_points );
    cout <<"homogenised_3d_points.type() :" << homogenised_3d_points.type() << endl;

    assert(homogenised_3d_points.type() == CV_32F);
    for(uint16_t i = 0; i < homogenised_3d_points.cols; i++)
    {
        Mat currPoint3d =  homogenised_3d_points.col(i);
        currPoint3d /= currPoint3d.at<float>(3, 0);
        Point3f p(
            currPoint3d.at<float>(0, 0),
            currPoint3d.at<float>(1, 0),
            currPoint3d.at<float>(2, 0)
        );
        points_3f.push_back(p);
    }

    vector<Point2f> reprojected_points;
    vector<float> reprojection_error;
    vector<double> dummy;
    projectPoints(points_3f, R, t, K, dummy, reprojected_points);
    float average_reprojection_error = 0;
    float d;
    for(uint16_t i = 0; i  < triangulation_points2.size(); i++)
    {
        d = dist(triangulation_points2[i], reprojected_points[i]);
        average_reprojection_error +=d;
        if ( d < 1.f)
        {   
            DataPoint pc;
            pc.keypoint_index = vector<int>(cameraStates.size(), -1);
            pc.point = points_3f[i]; 
            pc.keypoint_index[0] = triangulation_indices1[i];
            pc.keypoint_index[1] = triangulation_indices2[i];
            pc.color = cameraStates[0].Image.at<Vec3b>(kp1[triangulation_indices1[i]].pt);
            point_cloud.push_back(pc);
        }
    }
    average_reprojection_error = average_reprojection_error/triangulation_points2.size();
    cout << "Average reprojection error: " << average_reprojection_error << endl;
    d = 0; 
    average_reprojection_error = 0;

    done_views.push_back(0);
    done_views.push_back(1);

}


int find_best_correspondance(int view_to_eval)
{
    int max_correspondances = -1;
    int best_match_view = -1;
    vector<int> point_cloud_status(point_cloud.size(),0);
    for(uint16_t i = 0; i < done_views.size(); i++ )
    {
        int done_view_idx = done_views[i];
        int correspondances = 0;
        vector<DMatch> good_matches = match_table[done_view_idx][view_to_eval];
        for (unsigned int match  = 0; match < good_matches.size(); match++) 
        {
            // the index of the matching 2D point in <old_view>
            int idx_in_old_view = good_matches[match].queryIdx;
            //scan the existing cloud to see if this point from <old_view>exists 
            for (unsigned int pc_idx = 0; pc_idx < point_cloud.size(); pc_idx++) 
            {
                // see if this 2D point from <old_view> contributed to this 3D point in the cloud
                if (idx_in_old_view == point_cloud[pc_idx].keypoint_index[done_view_idx] && point_cloud_status[pc_idx] == 0) //prevent duplicates
                {
                    //2d point in image <working_view>
                    point_cloud_status[pc_idx] = 1;
                    point_cloud[pc_idx].keypoint_index[view_to_eval] = good_matches[match].trainIdx;
                    correspondances++;
                    break;
                }
            }
        }
        if(correspondances > max_correspondances)
        {
            max_correspondances = correspondances;
            best_match_view = done_view_idx;
        }
    }
    cout << "Best match for image "<< view_to_eval <<" is image "<< best_match_view << " with " << max_correspondances << " matches\n";
    return best_match_view;
}


void find_2d_3d_matches(int view1, int view2, vector<Point2f>& imagePoints, vector<Point3f>& ppoint_cloud, vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, vector<int>& kpidx1, vector<int>& kpidx2)
{
    ppoint_cloud.clear();
    kp1.clear();
    kp2.clear();
    kpidx1.clear();
    kpidx2.clear();

    vector<int> point_cloud_status(point_cloud.size(),0);
    vector<DMatch> good_matches = match_table[view1][view2];
    for (unsigned int match  = 0; match < good_matches.size(); match++) 
    {
        // the index of the matching 2D point in <old_view>
        int idx_in_old_view = good_matches[match].queryIdx;
        //scan the existing cloud to see if this point from <old_view>exists 
        for (unsigned int pcldp = 0; pcldp < point_cloud.size(); pcldp++) 
        {
            // see if this 2D point from <old_view> contributed to this 3D point in the cloud
            if (idx_in_old_view == point_cloud[pcldp].keypoint_index[view1] && point_cloud_status[pcldp] == 0) //prevent duplicates
            {
                ppoint_cloud.push_back(point_cloud[pcldp].point);
                imagePoints.push_back( cameraStates[view2].keypoints[ good_matches[match].trainIdx ].pt) ;
                kp1.push_back(cameraStates[view1].keypoints[ good_matches[match].queryIdx ]);
                kp2.push_back(cameraStates[view2].keypoints[ good_matches[match].trainIdx ]);
                kpidx1.push_back(good_matches[match].queryIdx);
                kpidx2.push_back(good_matches[match].trainIdx);
                //2d point in image <working_view>
                point_cloud_status[pcldp] = 1;
                break;
            }
        }
    }
    
}


void addToGlobalPC(int currentView, vector<DataPoint>& pc_to_add)
{
    int matches = 0;
    cout << "Appending PC of size " << pc_to_add.size() << endl;
    for(uint16_t idx = 0; idx < pc_to_add.size(); idx++)
    {
        int index_in_current_view = pc_to_add[idx].keypoint_index[currentView];
        for(uint16_t i = 0; i < done_views.size(); i++ )
        {
            int view_to_eval = done_views[i];
            vector<DMatch> view1Match = match_table[view_to_eval][currentView];
            for(uint16_t match = 0; match < view1Match.size(); match++)
            {
                if(index_in_current_view == view1Match[match].trainIdx)
                {
                    pc_to_add[idx].keypoint_index[view_to_eval] = view1Match[match].queryIdx;
                    matches++;
                    break;
                }
            } 
            
        }

    }
    cout<<"Matches : " << matches << endl;
    point_cloud.insert(point_cloud.end(),pc_to_add.begin(), pc_to_add.end());
    cout << pc_to_add.size() <<" points added to global point cloud! Current size is " << point_cloud.size() << endl;
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

    set<fs::path> im_paths;
    string dataset_image_path = dataset_source_dir + string("/images");
    for (const auto & entry : fs::directory_iterator(dataset_image_path))
    {
        im_paths.insert(entry.path());
    }
    
    Ptr<SURF> surf = SURF::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    cout << "Reading images...\n";
    for(auto image_path : im_paths)
    {
        cout << "Reading image: " << image_path.c_str() << endl;
        Mat img = imread(image_path.c_str(), IMREAD_COLOR);
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
    cout << "Finished reading images...\n";

    if(cameraStates.size() <= 2)
    {
        cout << "The SFM pipeline needs atleast 3 images to function!" << endl;
        assert(cameraStates.size() > 2);
    }

    //Populate match table with empty vectors
    for(uint16_t i =0; i < cameraStates.size(); i++)
    {
        for(uint16_t j = 0; j < cameraStates.size(); j++)
        {
            match_table[i].push_back(vector<DMatch>());
        }
    }

    cout << "Calculating matches between images...\n";
    for(uint16_t i = 0; i < cameraStates.size(); i++)
    {
        for(uint16_t j = 0; j < cameraStates.size(); j++)
        {
            //If correspondance to itself or if it has already been calc, disregard
            if ( i == j ) continue;
            Mat desc1 = cameraStates[i].Descriptor;
            Mat desc2 = cameraStates[j].Descriptor;
            vector<vector<DMatch>> knn_matches;
            matcher->knnMatch(desc1, desc2, knn_matches, 2);
            std::vector<DMatch> good_matches;
            for (size_t idx = 0; idx < knn_matches.size(); idx++)
            {
                if (knn_matches[idx][0].distance < ratio_thresh * knn_matches[idx][1].distance)
                {
                    good_matches.push_back(knn_matches[idx][0]);
                }
            }

            match_table[i][j] = good_matches;
        }
    }

    cout << "Finished calculating matches between images...\n";

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
        vector<double> dummy;
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

        projectPoints(points_3f, R, t, K, dummy, reprojected_points);
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

        addToGlobalPC(current_view, pc_to_add);

        average_reprojection_error = average_reprojection_error/currpts.size();
        cout<<"Average RE: " << average_reprojection_error << endl;
        cout<<endl<<endl;



        done_views.push_back(current_view);
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