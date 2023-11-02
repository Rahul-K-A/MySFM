#include "pcl_helpers.h"

using namespace pcl;

void savePointCloud_to_PCD(vector<DataPoint>& point_cloud)
{
    cout << "Creating PCL Point Cloud...\n";
    //Convert Point3D into PointXYZRGB
    PointCloud<PointXYZRGB>::Ptr point_cloud_ptr(new PointCloud<PointXYZRGB>);
    PointCloud<PointXYZRGB>::Ptr point_cloud_ptr_filtered(new PointCloud<PointXYZRGB>);
    for(uint16_t i = 0; i < point_cloud.size(); i++)
    {
        PointXYZRGB basic_point;
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

    // Remove outliers
    StatisticalOutlierRemoval<PointXYZRGB> sor;
    sor.setInputCloud (point_cloud_ptr);
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter (*point_cloud_ptr_filtered);
    io::savePCDFileASCII("final.pcd", *point_cloud_ptr_filtered);
}