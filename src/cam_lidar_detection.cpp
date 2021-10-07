#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <vector>
#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/visualization/pcl_visualizer.h>
// #include </usr/include/pcl-1.8/pcl/visualization/pcl_visualizer.h>
// #include </usr/local/include/pcl-1.8/pcl/visualization/pcl_visualizer.h>
// #include </usr/local/include/pcl-1.8/pcl/visualization/impl/pcl_visualizer.hpp>

class Cam_lidar_detection{
public:
    ros::NodeHandle nh;
    ros::NodeHandle nhPrivate;
    void run();
    void lidar_callback(const sensor_msgs::PointCloud2& data);
    //void detect_callback(const std::vector& data);
    void clustering(void);
    void fix_clustering(void);
	double ComputeTolerance(const pcl::PointXYZ& point);
	bool CustomCondition(const pcl::PointXYZ& spped_point, const pcl::PointXYZ& candidate_point, float squared_distance);
	void downsampling(void);
	// void visualization(void);
	void rviz_visualization(void);
    ros::Publisher pub_marker;
    ros::Subscriber sub_lidar;
    ros::Subscriber sub_datect;

    //pcl::visualization::PCLVisualizer *viewer = new pcl::visualization::PCLVisualizer ("3D Viewer");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_nan {new pcl::PointCloud<pcl::PointXYZ>}; // NaN値あり
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_nofilter {new pcl::PointCloud<pcl::PointXYZ>}; // filer前
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud {new pcl::PointCloud<pcl::PointXYZ>}; // NaN値なし

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;

	visualization_msgs::MarkerArray marker_array;
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr voxcel_tmp {new pcl::PointCloud<pcl::PointXYZ>};
	pcl::VoxelGrid<pcl::PointXYZ> vg;

    double cluster_tolerance;
	double ratio_depth_tolerance;
	int min_cluster_size;
	int max_cluster_size;
	double min_tolerance;
	double max_tolerance;

	double leafsize;
	

	//Clusteringで実行必要とされる変数たち

	/*クラスタリング後のインデックスが格納されるベクトル*/
	std::vector<pcl::PointIndices> cluster_indices;
	/*今回の主役*/
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ece;
	pcl::ExtractIndices<pcl::PointXYZ> ei;
	/*extract*/
	pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_clustered_points {new pcl::PointCloud<pcl::PointXYZ>};
	pcl::PointIndices::Ptr tmp_clustered_indices {new pcl::PointIndices};

	//Visualizationに使用する変数たち
	pcl::PointXYZ minPt, maxPt;
	// std::random_device rnd;

	//filter
	pcl::PassThrough<pcl::PointXYZ> pass;
};

void Cam_lidar_detection::run()
{
    pub_marker = nh.advertise<visualization_msgs::MarkerArray>("clustering", 100);

    sub_lidar = nh.subscribe("/points_no_ground", 100, &Cam_lidar_detection::lidar_callback, this);

    // viewer->setBackgroundColor(1.0, 1.0, 1.0, 0);
	// viewer->addCoordinateSystem(1.0, "axis");
	// viewer->setCameraPosition(0.0, 0.0, 35.0, 0.0, 0.0, 0.0);

	nhPrivate.param("cluster_tolerance", cluster_tolerance, 0.3);
	nhPrivate.param("ratio_depth_tolerance", ratio_depth_tolerance, 0.1);
	nhPrivate.param("min_cluster_size", min_cluster_size, 100);
	nhPrivate.param("min_cluster_size", max_cluster_size, 100000);
	nhPrivate.param("min_tolerance", min_tolerance, 0.1);
	nhPrivate.param("max_tolerance", max_tolerance, 0.5);
	nhPrivate.param("leafsize", leafsize, 0.07);

	std::cout << "cluster_tolerance = " << cluster_tolerance << std::endl;
	std::cout << "min_cluster_size = " << min_cluster_size << std::endl;

    ros::Rate loop_rate(100);

    while(ros::ok()){
        ros::spinOnce();
		loop_rate.sleep();
    }
}

void Cam_lidar_detection::lidar_callback(const sensor_msgs::PointCloud2& data){
    //sensor_msgs::PointCloud2からpcl::PointXYZに変換
    pcl::fromROSMsg(data, *cloud_nan);

    // NaN値が入ってるといろいろ面倒なので除去
    std::vector<int> nan_index;
    pcl::removeNaNFromPointCloud(*cloud_nan, *cloud_nofilter, nan_index);

	pass.setInputCloud(cloud_nofilter);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(-0.6, 0.6);
	pass.filter(*cloud);

	downsampling();
    clusters.clear();
    clustering();
	//fix_clustering();
	rviz_visualization();
    // visualization();
}

void Cam_lidar_detection::clustering(void){
    double time_start = ros::Time::now().toSec();
    /*clustering*/
	/*kd-treeクラスを宣言*/
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree {new pcl::search::KdTree<pcl::PointXYZ>};
	/*探索する点群をinput*/
	tree->setInputCloud(cloud);
	/*距離の閾値を設定*/
	ece.setClusterTolerance(cluster_tolerance);
	/*各クラスタのメンバの最小数を設定*/
	ece.setMinClusterSize(min_cluster_size);
	/*各クラスタのメンバの最大数を設定*/
	ece.setMaxClusterSize(cloud->points.size());
	//ece.setMaxClusterSize(100);
	/*探索方法を設定*/
	ece.setSearchMethod(tree);
	/*クラスリング対象の点群をinput*/
	ece.setInputCloud(cloud);
	/*クラスリング実行*/
	cluster_indices.clear();
	ece.extract(cluster_indices);

	std::cout << "cluster_indices.size() = " << cluster_indices.size() << std::endl;
	ROS_INFO("hoe many clustering : %d", cluster_indices.size());

	/*dividing（クラスタごとに点群を分割）*/
	ei.setInputCloud(cloud);
	ei.setNegative(false);
	for(size_t i=0;i<cluster_indices.size();i++){
		*tmp_clustered_indices = cluster_indices[i];
		ei.setIndices(tmp_clustered_indices);
		ei.filter(*tmp_clustered_points);
		/*input*/
		clusters.push_back(tmp_clustered_points);
		// pcl::PointXYZ minPt, maxPt;
		//pcl::getMinMax3D(*)
	}
	// Clusteringの中身を表示
	// for (size_t i=0; i < clusters.size(); i++){
	// 	for (size_t pt=0; pt < clusters[i]->size(); pt++){
	// 		std::cout << clusters[i]->points[pt].x << " ";
	// 	}
	// 	std::cout << "------------------------------------" << std::endl;
	// }

	// std::cout << "clustering time [s] = " << ros::Time::now().toSec() - time_start << std::endl;
}

// void Cam_lidar_detection::fix_clustering(void){ // https://lilaboc.work/archives/20178032.html
// 	double time_start = ros::Time::now().toSec();

// 	/*search config*/
// 	/*kd-treeクラスを宣言*/
// 	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
// 	/*探索する点群をinput*/
// 	kdtree.setInputCloud(cloud);
// 	max_cluster_size = cloud->points.size();
// 	/*objects*/
// 	std::vector<pcl::PointIndices> cluster_indices;
// 	std::vector<bool> processed(cloud->points.size(), false);
// 	std::vector<int> nn_indices;
// 	std::vector<float> nn_distances;
// 	/*clustering*/
// 	for(size_t i=0;i<cloud->points.size();++i){
// 		if(processed[i])	continue;	//既に分類されているかチェック
// 		/*set seed（シード点を設定）*/
// 		std::vector<int> seed_queue;
// 		int sq_idx = 0;
// 		seed_queue.push_back(i);
// 		processed[i] = true;
// 		/*clustering*/
// 		while(sq_idx < seed_queue.size()){	//探索しきるまでループ
// 			/*search*/
// 			double tolerance = ComputeTolerance(cloud->points[seed_queue[sq_idx]]);
// 			int ret = kdtree.radiusSearch(cloud->points[seed_queue[sq_idx]], tolerance, nn_indices, nn_distances);
// 			if(ret == -1){
// 				PCL_ERROR("[pcl::extractEuclideanClusters] Received error code -1 from radiusSearch\n");
// 				exit(0);
// 			}
// 			/*check*/
// 			for(size_t j=0;j<nn_indices.size();++j){
// 				/*//既に分類されているかチェック*/
// 				if(nn_indices[j]==-1 || processed[nn_indices[j]])	continue;
// 				/*カスタム条件でチェック*/
// 				if(CustomCondition(cloud->points[seed_queue[sq_idx]], cloud->points[nn_indices[j]], nn_distances[j])){
// 					seed_queue.push_back(nn_indices[j]);
// 					processed[nn_indices[j]] = true;
// 				}
// 			}
// 			sq_idx++;
// 		}
// 		/*judge（クラスタのメンバ数が条件を満たしているか）*/
// 		if(seed_queue.size()>=min_cluster_size && seed_queue.size()<=max_cluster_size){
// 			pcl::PointIndices tmp_indices;
// 			tmp_indices.indices = seed_queue;
// 			cluster_indices.push_back(tmp_indices);
// 		}
// 	}
// 	std::cout << "cluster_indices.size() = " << cluster_indices.size() << std::endl;
// 	/*extraction（クラスタごとに点群を分割）*/
// 	pcl::ExtractIndices<pcl::PointXYZ> ei;
// 	ei.setInputCloud(cloud);
// 	ei.setNegative(false);
// 	for(size_t i=0;i<cluster_indices.size();i++){
// 		/*extract*/
// 		pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_clustered_points (new pcl::PointCloud<pcl::PointXYZ>);
// 		pcl::PointIndices::Ptr tmp_clustered_indices (new pcl::PointIndices);
// 		*tmp_clustered_indices = cluster_indices[i];
// 		ei.setIndices(tmp_clustered_indices);
// 		ei.filter(*tmp_clustered_points);
// 		/*input*/
// 		clusters.push_back(tmp_clustered_points);
// 	}

// 	std::cout << "clustering time [s] = " << ros::Time::now().toSec() - time_start << std::endl;
// }

// double Cam_lidar_detection::ComputeTolerance(const pcl::PointXYZ& point) //https://lilaboc.work/archives/20178032.html
// {
// 	/*センサからの距離（depth）*/
// 	double depth = sqrt(
// 		point.x * point.x
// 		+ point.y * point.y
// 		+ point.z * point.z
// 	);

// 	double tolerance = ratio_depth_tolerance*depth;	//距離に比例
// 	if(tolerance < min_tolerance)	tolerance = min_tolerance;
// 	if(tolerance > max_tolerance)	tolerance = max_tolerance;

// 	return tolerance;
// }

// bool Cam_lidar_detection::CustomCondition(const pcl::PointXYZ& seed_point, const pcl::PointXYZ& candidate_point, float squared_distance){
// 	return true;
// }//https://lilaboc.work/archives/20178032.html
	

void Cam_lidar_detection::rviz_visualization(void){
	marker_array.markers.resize(clusters.size());
	// marker_array.markers.resize(1);
	for(size_t i=0; i<clusters.size(); i++){
		// int i = 0;
		marker_array.markers[i].header.frame_id = "/velodyne";
		marker_array.markers[i].header.stamp = ros::Time::now();
		marker_array.markers[i].ns = "clustering_object";
		marker_array.markers[i].id = i;
		marker_array.markers[i].lifetime = ros::Duration(1.0);

		marker_array.markers[i].type = visualization_msgs::Marker::CUBE;
		marker_array.markers[i].scale.x = maxPt.x - minPt.x;
		marker_array.markers[i].scale.y = maxPt.y - minPt.y;
		marker_array.markers[i].scale.z = maxPt.z - minPt.z;

		pcl::getMinMax3D(*clusters[i], minPt, maxPt);
		marker_array.markers[i].pose.position.x = (maxPt.x + minPt.x) / 2;
		marker_array.markers[i].pose.position.y = (maxPt.y + minPt.y) / 2;
		marker_array.markers[i].pose.position.z = (maxPt.z + minPt.z) / 2;

		marker_array.markers[i].pose.orientation.x = 0;
		marker_array.markers[i].pose.orientation.y = 0;
		marker_array.markers[i].pose.orientation.z = 0;
		marker_array.markers[i].pose.orientation.w = 1;

		marker_array.markers[i].color.r = 0.0f;
		marker_array.markers[i].color.g = 0.0f;
		marker_array.markers[i].color.b = 1.0f;
		marker_array.markers[i].color.a = 0.4f;
	}
	pub_marker.publish(marker_array);
	marker_array.markers.clear();
	std::cout << "published!!!!!" << std::endl;
}

void Cam_lidar_detection::downsampling(void){
	vg.setInputCloud(cloud);
	vg.setLeafSize(leafsize, leafsize, leafsize);
	vg.filter(*voxcel_tmp);
	*cloud = *voxcel_tmp;
}

// void Cam_lidar_detection::visualization(void)
// {
// 	/*前ステップの可視化をリセット*/
// 	viewer->removeAllPointClouds();

// 	/*cloud*/
// 	viewer->addPointCloud(cloud, "cloud");
// 	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "cloud");
// 	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
// 	/*clusters*/
// 	double rgb[3] = {};
// 	const int channel = 3;	//RGB
// 	const double step = ceil(pow(clusters.size()+2, 1.0/(double)channel));	//exept (000),(111)
// 	const double max = 1.0;
// 	/*クラスタをいい感じに色分け*/
// 	for(size_t i=0;i<clusters.size();i++){
// 		std::string name = "cluster_" + std::to_string(i);
// 		rgb[0] += 1/step;
// 		for(int j=0;j<channel-1;j++){
// 			if(rgb[j]>max){
// 				rgb[j] -= max + 1/step;
// 				rgb[j+1] += 1/step;
// 			}
// 		}
// 		viewer->addPointCloud(clusters[i], name);
// 		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, rgb[0], rgb[1], rgb[2], name);
// 		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, name);
// 	}
// 	/*表示の更新*/
// 	viewer->spinOnce();
// }

int main(int argc, char** argv){
    ros::init(argc, argv, "cam_lid_detection");
    Cam_lidar_detection cam_lidar_detection;
    cam_lidar_detection.run();

	//ros::spin();

    return 0;
}