//
// Created by kx on 18-11-25.
//
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
//#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <boost/thread/thread.hpp>
#include <iostream>
#include <vector>
#include <string.h> //包含strcmp的头文件,也可用: #include <ctring>
#include <dirent.h>

#include "omp.h"

using namespace std;

//100 243 564 907 4300 7772
//#define FILE_PCD "/media/work/data/kitti/odometry/submap/00/000130.pcd"
//#define OUT_PCD "/home/kx/project/3D/pcl_visualize/out_000130.pcd"
//#define CLUSTER_PCD "/home/kx/project/3D/pcl_visualize/cluster_000130.pcd"
//#define CLUSTER_TXT "/media/data/csc105/孔/单帧/cloud7772_cluster.txt"


void getFileNames(const std::string path, std::vector<std::string>& filenames, const std::string suffix = "")
{
    DIR *pDir;
    struct dirent* ptr;
    if (!(pDir = opendir(path.c_str())))
        return;
    while ((ptr = readdir(pDir))!=0)
    {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
        {
            std::string file = path + "/" + ptr->d_name;
            if (opendir(file.c_str()))
            {
                getFileNames(file, filenames, suffix);
            }
            else
            {
                if (suffix == file.substr(file.size() - suffix.size()))
                {
                    filenames.push_back(file);
                }
            }
        }
    }
    closedir(pDir);
}

/**
 * 字符串替换函数
 * #function name   : replace_str()
 * #param str       : 操作之前的字符串
 * #param before    : 将要被替换的字符串
 * #param after     : 替换目标字符串
 * #return          : void
 */
void replace_str(std::string& str, const std::string& before, const std::string& after)
{
    for (std::string::size_type pos(0); pos != std::string::npos; pos += after.length())
    {
        pos = str.find(before, pos);
        if (pos != std::string::npos)
            str.replace(pos, before.length(), after);
        else
            break;
    }
}

int segment_cloud(string pcd_file)
{
    string output_file = pcd_file;
    replace_str(output_file, "submap", "submap_seg_pcd"); // TODO: auto name
    string bin_file = output_file;
    replace_str(bin_file, "pcd", "bin");
    replace_str(bin_file, "submap_seg_pcd", "submap_seg_bin");

    cout << "input file: " << pcd_file <<endl;
    cout << "output file: " << output_file <<endl;
    cout << "bin file: " << bin_file <<endl;
    pcl::PointCloud<pcl::PointXYZL>::Ptr input_cloud_map(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::io::loadPCDFile(pcd_file,*input_cloud_map);
    if(input_cloud_map->size()==0){
        cout<<"input map size = 0 !!!"<<endl;
        return -1;
    }
    cout<<"input map size: "<< input_cloud_map->size()<<endl;
    struct timeval t1,t2;
    double timeuse;
    gettimeofday(&t1,NULL);

    // ROI box
    pcl::PointXYZ point(0,0,0);
    float box_z_min = -5.0;
    float box_z_max = +20.0;
    float box_x_min = -50.0;
    float box_x_max = +50.0;
    float box_y_min = -50.0;
    float box_y_max = +50.0;
    pcl::PointCloud <pcl::PointXYZL>::Ptr filter_cloud(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PassThrough<pcl::PointXYZL> pass;
    pass.setInputCloud (input_cloud_map);
    pass.setFilterFieldName ("z");
//  pass.setFilterLimits (point.z-0.8, point.z+1.5);// 0 1
    pass.setFilterLimits (point.z+box_z_min, point.z+box_z_max);// 0 1
    pass.filter (*filter_cloud);
    pass.setInputCloud (filter_cloud);
    pass.setFilterFieldName ("x");
//  pass.setFilterLimits (point.x-3.0, point.x+3.0);// 0 1
    pass.setFilterLimits (point.x+box_x_min, point.x+box_x_max);// 0 1
    pass.filter (*filter_cloud);
    pass.setInputCloud (filter_cloud);
    pass.setFilterFieldName ("y");
//  pass.setFilterLimits (point.y-3.0, point.y+3.0);// 0 1
    pass.setFilterLimits (point.y+box_y_min, point.y+box_y_max);// 0 1
    pass.filter (*input_cloud_map);


    pcl::VoxelGrid<pcl::PointXYZL> sor;
    sor.setInputCloud(input_cloud_map);
    sor.setLeafSize(0.1f, 0.1f, 0.1f);
//    sor.setLeafSize(0.05f, 0.05f, 0.05f); //sor.setLeafSize(0.1f, 0.1f, 0.1f);
    sor.filter(*input_cloud_map);
//    pcl::visualization::CloudViewer viewer ("Cluster viewer");
//    pcl::visualization::PCLVisualizer::Ptr visualizer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//    pcl::io::savePCDFileBinary("voxelize.pcd", *input_cloud_map);
//   fit ground plane

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZL> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
//    Eigen::Vector3f axis_z = Eigen::Vector3f(0,0,1);
//    seg.setAxis(axis_z);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.5);
//    seg.
    seg.setInputCloud (input_cloud_map);
    seg.segment (*inliers, *coefficients);
//    std::cout << coefficients->values[0] <<" "
//              << coefficients->values[1] <<" "
//              << coefficients->values[2] <<" "
//              << coefficients->values[3] << std::endl;

    if(inliers->indices.empty()){
        cout << "cannot find plane !"<<endl;
        cout << pcd_file << endl;
        return -1;
    } else{
//        cout << "plane points size: " << inliers->indices.size()<<endl;
    }

    // Extract non-ground returns
//    pcl::PointIndices::Ptr ground(new pcl::PointIndices(*inliers));
    pcl::ExtractIndices<pcl::PointXYZL> extract;
    extract.setInputCloud(input_cloud_map);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloud_no_ground);
    cout<<"cloud_no_ground: "<<cloud_no_ground->size()<<endl;
//    viewer.showCloud(cloud_no_ground);
//    pcl::io::savePCDFileBinary("no_ground.pcd", *cloud_no_ground);

    // Remove outlier
    pcl::RadiusOutlierRemoval<pcl::PointXYZL> outrem;
    outrem.setInputCloud(cloud_no_ground);
    outrem.setRadiusSearch(1.0); //radius 2m
    outrem.setMinNeighborsInRadius (50);// 50points
    // apply filter
    outrem.filter(*cloud_no_ground);
//    viewer.showCloud(cloud_no_ground);
//    pcl::io::savePCDFileBinary("remove_outlier.pcd", *cloud_no_ground);
//    cout<<"remove outlier: "<<cloud_no_ground->size()<<endl;

    //euclidean cluster
    pcl::search::KdTree<pcl::PointXYZL>::Ptr KdTree(new pcl::search::KdTree<pcl::PointXYZL>);
    KdTree->setInputCloud(cloud_no_ground);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZL> ec;
    ec.setClusterTolerance(0.10);//0.02
    ec.setMinClusterSize(200);  //200
    ec.setSearchMethod(KdTree);
    ec.setMaxClusterSize(250000);//25000
    ec.setInputCloud(cloud_no_ground);
    ec.extract(cluster_indices);
    std::cout << "Number of cluster_indices is equal to " << cluster_indices.size() << std::endl;

    ofstream outfile(bin_file,ios::binary);
    if(!outfile){
        return -1;
    }
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZL>);
    for(size_t i=0;i<cluster_indices.size();i++){
        for(size_t j=0;j<cluster_indices[i].indices.size();j++){
            pcl::PointXYZL tmp_pt;
            tmp_pt.x = cloud_no_ground->points[cluster_indices[i].indices[j]].x;
            tmp_pt.y = cloud_no_ground->points[cluster_indices[i].indices[j]].y;
            tmp_pt.z = cloud_no_ground->points[cluster_indices[i].indices[j]].z;
//            tmp_pt.r = static_cast<uint8_t>(cloud_no_ground->points[cluster_indices[i].indices[j]].intensity);
//            tmp_pt.g = static_cast<uint8_t>(cloud_no_ground->points[cluster_indices[i].indices[j]].intensity);
//            tmp_pt.b = static_cast<uint8_t>(cloud_no_ground->points[cluster_indices[i].indices[j]].intensity);
            tmp_pt.label = (int)i;
            cloud_cluster->points.push_back(tmp_pt);
            outfile.write((char*)&tmp_pt.x, sizeof(tmp_pt.x));
            outfile.write((char*)&tmp_pt.y, sizeof(tmp_pt.y));
            outfile.write((char*)&tmp_pt.z, sizeof(tmp_pt.z));
            outfile.write((char*)&tmp_pt.label, sizeof(tmp_pt.label));
        }
    }

//    cout<<"successfully convert"<<endl;
    cout<<cloud_cluster->size()<<endl;
    pcl::io::savePCDFileBinary(output_file, *cloud_cluster);
    outfile.close();

//    visualizer->addPointCloud(cloud_cluster,"euclidean cluster");
//    while(!visualizer->wasStopped())
//    {
//        visualizer->spinOnce(100);
//    }
//    while (!viewer.wasStopped ())
//    {
//    }
    gettimeofday(&t2,NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
    printf("Use Time:%f\n",timeuse);

    return 0;
}




int main()
{
// TODO: auto mkdirs
//    string path = "/media/work/data/kitti/odometry/submap/08";
    string path = "/media/data/submap/08";
    std::vector<std::string> file_paths;
    getFileNames(path, file_paths);
//    omp_set_num_threads();
    #pragma omp parallel for
    for(auto i=0; i < file_paths.size(); i++)
    {
        segment_cloud(file_paths[i]);
        cout << endl;
        printf("i = %d, I am Thread %d\n", i, omp_get_thread_num());
//        printf("i = %d, I am Thread %d\n", i, 1);
    }

    return 0;
}
