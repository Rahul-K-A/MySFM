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
    sfm();

    vector<DataPoint>& point_cloud = getGlobalPC(); 

    savePointCloud_to_PCD(point_cloud);

}