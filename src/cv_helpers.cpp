#include "common-includes.h"
#include "cv_helpers.h"

using namespace cv;
using namespace cv::xfeatures2d;

static map<int, vector<vector<DMatch>> > match_table;
static vector<DataPoint> point_cloud; //our global 3D point cloud
static vector<CameraState> cameraStates;
static vector<int> done_views;
static Ptr<SURF> surf;
static Ptr<DescriptorMatcher> matcher;
const float dThreshold = 1.f;
/*Ratio thresh needs to be kept around 0.5 to produce good BA values*/
const float ratio_thresh = 0.6f;
const float acceptable_error = 15.f;

static Mat K = (Mat1d(3,3) <<      2759.48, 0, 1520.69, 
                            0, 2764.16, 1006.81, 
                            0, 0, 1); 

cvsba::Sba BA; 
cvsba::Sba::Params params ;


/// @brief Helper function to find the euclidean distance between two points in an image
/// @param a First Point
/// @param b Second point
/// @return The distance between the 2 points
float eDistance(cv::Point2f a, cv::Point2f b)
{
   float dist =  (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
   dist = sqrtf(dist);
   return dist;
};

vector<DataPoint>& getGlobalPC()
{
    return point_cloud;
}

void readImages(set<fs::path>& image_paths)
{
    cout << "Reading images...\n";
    for(auto image_path : image_paths)
    {
        cout << "Reading image: " << image_path.c_str() << endl;
        Mat img = imread(image_path.c_str(), IMREAD_COLOR);
        vector<KeyPoint> keypoint;
        CameraState currCameraState;
        img.copyTo(currCameraState.Image);
        Mat img_gray, descriptor;
        cvtColor(img, img_gray, COLOR_BGR2GRAY);
        surf->detectAndCompute(img_gray, noArray(), keypoint, descriptor);
        currCameraState.keypoints = keypoint;
        descriptor.copyTo(currCameraState.Descriptor);
        cameraStates.push_back(currCameraState);
        
    }
    cout << "Finished reading " << cameraStates.size() <<" images...\n";
    if(cameraStates.size() <= 2)
    {
        cout << "The SFM pipeline needs atleast 3 images to function!" << endl;
        assert(cameraStates.size() > 2);
    }
    return;
}


void calculateFeatureCorrespondance()
{
    //Populate match table with empty vectors
    for(uint16_t i = 0; i < cameraStates.size(); i++)
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
            if ( i == j || match_table[i][j].size()) continue;
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
            cout << "There are " << good_matches.size() << " good matches between images " << i <<" and " << j << endl;
            match_table[i][j] = good_matches;
            match_table[j][i] = flip_match(good_matches);
        }
    }

    cout << "Finished calculating matches between images...\n";
}

int find_best_correspondence(int view_to_eval)
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
                    //point_cloud[pc_idx].keypoint_index[view_to_eval] = good_matches[match].trainIdx;
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


vector<DMatch> flip_match(const vector<DMatch>& matches_vector)
{
    vector<DMatch> flipped_vector;
    for(const DMatch& match : matches_vector)
    {
        DMatch current_match = match;
        std::swap(current_match.queryIdx, current_match.trainIdx);
        flipped_vector.push_back(current_match);
        flipped_vector.push_back(current_match);
    }
    return flipped_vector;
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

/// @brief Adds a set of points to the global point cloud
/// @param prevView View 1 which was used to triangulate the points
/// @param currentView View 2 used for triangulation
/// @param pc_to_add Point cloud to be added 
void addToGlobalPC(int prevView, int currentView, vector<DataPoint>& pc_to_add)
{
    int matches = 0;
    cout << "Appending PC of size " << pc_to_add.size() << endl;
    for(uint16_t idx = 0; idx < pc_to_add.size(); idx++)
    {
        int index_in_current_view = pc_to_add[idx].keypoint_index[currentView];
        int index_in_previous_view = pc_to_add[idx].keypoint_index[prevView];
        bool found = false;
        for(uint16_t i = 0; i < done_views.size(); i++ )
        {
            int view_to_eval = done_views[i];
            vector<DMatch> view1Match1 = match_table[view_to_eval][currentView];
            for(uint16_t match = 0; match < view1Match1.size(); match++)
            {
                if(index_in_current_view == view1Match1[match].trainIdx)
                {
                    pc_to_add[idx].keypoint_index[view_to_eval] = view1Match1[match].queryIdx;
                    matches++;
                    found = true;
                    break;
                }
            }
        }

        /* Also fill in old view correspondences*/
        for(uint16_t i = 0; i < done_views.size(); i++ )
        {
            int view_to_eval = done_views[i];
            vector<DMatch> view1Match1 = match_table[view_to_eval][prevView];
            for(uint16_t match = 0; match < view1Match1.size(); match++)
            {
                if(index_in_previous_view == view1Match1[match].trainIdx)
                {
                    if(pc_to_add[idx].keypoint_index[view_to_eval] != -1)
                    {
                        pc_to_add[idx].keypoint_index[view_to_eval] = view1Match1[match].queryIdx;
                        matches++;
                    }

                    break;
                }
            }

        }
        

    }
    point_cloud.insert(point_cloud.end(),pc_to_add.begin(), pc_to_add.end());
    cout << pc_to_add.size() <<" points added to global point cloud! Current size is " << point_cloud.size() << endl;
    
    

}



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
        cout<<"R matrix is messed up!\n";
        assert(false);
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
        d = eDistance(triangulation_points2[i], reprojected_points[i]);
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

    done_views.push_back(0);
    done_views.push_back(1);

    //performBA(point_cloud, 0,1);
}

void initFeatureMatching()
{
    surf = SURF::create();
    matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
}


void sfm()
{
    for(uint16_t current_view = 2; current_view < cameraStates.size(); current_view++ )
    {
        cout << "Calculating image number: " << current_view << endl;
        std::vector<cv::Point3f> ppoint_cloud;           
        std::vector<cv::Point2f> imgPoints;
        vector<KeyPoint> kp1, kp2;
        vector<int> kpidx1, kpidx2;
        int best_match_view = find_best_correspondence(current_view);
        cout << "Found best correspondance: " << best_match_view << endl;
        find_2d_3d_matches(best_match_view, current_view, imgPoints, ppoint_cloud, kp1, kp2, kpidx1, kpidx2);
        Mat t,rvec,R;
        assert(ppoint_cloud.size() == imgPoints.size());
        solvePnPRansac(ppoint_cloud, imgPoints, K, noArray(), rvec, t, false);
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
            distance = eDistance(currpts[pt_idx], reprojected_points[pt_idx]);
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
}


void performBA()
{
    cout <<"\n\nStarting BA!\n";
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

    // cout << points.size() << endl;
    // cout <<  "imagePoints :"<< imagePoints.size() << endl;
    // cout << "R :"<< R.size() << endl;
    // cout <<  "T :"<< T.size() << endl;

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

    std::cout << "Optimization. Initial error=" << BA.getInitialReprjError() << " and Final error=" << BA.getFinalReprjError() << endl << endl << endl;
};
