#include "cv_helpers.h"
#include "pcl_helpers.h"





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
    initFeatureMatching();
    calculateFeatureCorrespondance();
    computeFirstPointCloud();
    sfm();

    vector<DataPoint>& point_cloud = getGlobalPC(); 

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
        cv::Vec3b color = point_cloud[i].color;
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
}