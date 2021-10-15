#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <vector>
#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
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
	void detection_callback(const std_msgs::Float32MultiArray::ConstPtr& data);
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
	
	// IoUのためのクラスタリングからxのmaxminを取り出す
	std::vector<float> msg_vec;
	Eigen::MatrixXf detection_deg_max;
	Eigen::MatrixXf detection_deg_min;
	Eigen::MatrixXf virtual_coord;
	Eigen::MatrixXf mole_deg_max;
	Eigen::MatrixXf deno_deg_max;
	Eigen::MatrixXf clusters_deg_max;
	Eigen::MatrixXf mole_deg_min;
	Eigen::MatrixXf deno_deg_min;
	Eigen::MatrixXf clusters_deg_min;
	std::vector<float> x_maxmin;
	std::vector<int> object_num; // IoUが閾値以上の番号を追加

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
	sub_datect = nh.subscribe("/person_box_coord", 100, &Cam_lidar_detection::detection_callback, this);

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

void Cam_lidar_detection::detection_callback(const std_msgs::Float32MultiArray::ConstPtr& data){
	msg_vec = data->data;
	// for(size_t i=0; i<msg_vec.size(); i++){
	// 	ROS_WARN("coord : %f", msg_vec[i]);
	// }

	float image_width = 1200;
	const int h = data->layout.dim[0].size;
	const int w = data->layout.dim[1].size;

	Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> detection_mat(msg_vec.data(), h, w);
	// 角度の抽出
	detection_mat = detection_mat/image_width * 360;
	// std::cout << "detecton_mat : \n" << detection_mat << std::endl;
	// Eigen::MatrixXf detection_deg_max = detection_mat.block(0, 0, 1, h);
	// Eigen::MatrixXf detection_deg_min = detection_mat.block(0, 2, 1, h);
	// std::cout << "detection_mat :\n" << detection_mat << std::endl;
	detection_deg_min =  detection_mat.block(0, 0, h, 1);
	detection_deg_max =  detection_mat.block(0, 2, h, 1);
	// std::cout << "detection : \n" << detection_mat.block(0, 2, h, 1) << std::endl;

	int num_clusters = x_maxmin.size() / 4;
	Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> cluster_mat(x_maxmin.data(), num_clusters, 4);
	ROS_WARN("-----------cluster---------\n");
	std::cout << "cluster_mat : \n" << cluster_mat << std::endl;
	// 基準となる(x, y)=(0, 10)の座標を用意
	object_num.clear();
	if((cluster_mat.size() != 0) && (detection_deg_max.size() != 0)){
		virtual_coord = Eigen::MatrixXf::Zero(num_clusters, 2);
		virtual_coord.col(0) = Eigen::VectorXf::Constant(num_clusters, -10);
		virtual_coord.col(1) = Eigen::VectorXf::Constant(num_clusters, 0);
		// std::cout << "virtual_coord dot : \n" << virtual_coord << std::endl;
		// std::cout << "num_clusters : \n" << num_clusters << std::endl;
		// std::cout << "cluster_mat dot : \n" << cluster_mat.block(0, 0, num_clusters, 2).dot(virtual_coord.transpose()) << std::endl;
		// std::cout << "cluster_mat block : \n" << cluster_mat.block(0, 0, num_clusters, 2) << std::endl;
		// std::cout << "virtual_coord transpose : \n" << virtual_coord.transpose() << std::endl;
		mole_deg_max = (cluster_mat.block(0, 0, num_clusters, 2) * virtual_coord.transpose()).col(0);
		// std::cout << "mole_deg_max : \n" <<  mole_deg_max << std::endl;
		deno_deg_max = 
				((cluster_mat.block(0, 0, num_clusters, 1).cwiseAbs2() + cluster_mat.block(0, 1, num_clusters, 1).cwiseAbs2()).cwiseSqrt()).array() * 
				((virtual_coord.block(0, 0, num_clusters, 1).cwiseAbs2() + virtual_coord.block(0, 1, num_clusters, 1).cwiseAbs2()).cwiseSqrt()).array(); 
		// std::cout << "deno_deg_max : \n" << deno_deg_max << std::endl;
		clusters_deg_max = ((mole_deg_max.array() / deno_deg_max.array()).acos() * 180 / M_PI) + 90; //+90はゼロ点調整
		// std::cout << "clusters_deg_max : \n" << clusters_deg_max << std::endl;

		mole_deg_min = (cluster_mat.block(0, 2, num_clusters, 2) * virtual_coord.transpose()).col(0);
		// std::cout << "mole_deg_min : \n" <<  mole_deg_max << std::endl;
		deno_deg_min = 
				((cluster_mat.block(0, 2, num_clusters, 1).cwiseAbs2() + cluster_mat.block(0, 3, num_clusters, 1).cwiseAbs2()).cwiseSqrt()).array() * 
				((virtual_coord.block(0, 0, num_clusters, 1).cwiseAbs2() + virtual_coord.block(0, 1, num_clusters, 1).cwiseAbs2()).cwiseSqrt()).array(); 
		// std::cout << "deno_deg_min : \n" << deno_deg_min << std::endl;
		clusters_deg_min = ((mole_deg_min.array() / deno_deg_min.array()).acos() * 180 / M_PI) + 90; //+90はゼロ点調整
		// std::cout << "clusters_deg_min : \n" << clusters_deg_min << std::endl;

		// caliculate IoU
		float overlap_left = 0;
		float overlap_right = 0;
		float alloffield = 0;
		float overlap_field = 0;
		float IoU = 0;
		for(size_t i=0; i < clusters_deg_max.size(); i++){
			ROS_WARN("---Number : %d---\n", i);
			for(size_t j=0; j < detection_deg_max.size(); j++){
				std::cout << "clusters_def_max\n" << clusters_deg_max(i) << std::endl;
				std::cout << "clusters_def_min\n" << clusters_deg_min(i) << std::endl;
				std::cout << "detection_def_max\n" << detection_deg_max(j) << std::endl;
				std::cout << "detection_def_min\n" << detection_deg_min(j) << std::endl;
				if ((detection_deg_max(j) - clusters_deg_max(i)) > 0){
					overlap_right = clusters_deg_max(i);
				}else{
					overlap_right = detection_deg_max(j);
				}
				if ((detection_deg_min(j) - clusters_deg_min(i)) > 0){
					overlap_left = detection_deg_min(j);
				}else{
					overlap_left = clusters_deg_min(i);
				}
				alloffield = clusters_deg_max(i) - clusters_deg_min(i) + detection_deg_max(j) - detection_deg_min(j);
				overlap_field = overlap_right - overlap_left;

				IoU	= overlap_field / (alloffield - overlap_field);
				if (IoU > 0){
					ROS_WARN("IoU : %f\n", IoU);
					object_num.push_back(i);
					break;
				}
			}
		}

		// for(size_t i = 0; i < object_num.size(); i++){
		// 	std::cout << "cluster number :" << object_num[i] << "\n";
		// }std::cout << "\n" << std::endl;
	}
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

	// std::cout << "cluster_indices.size() = " << cluster_indices.size() << std::endl;
	// ROS_INFO("hoe many clustering : %d", cluster_indices.size());

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

void Cam_lidar_detection::rviz_visualization(void){
	marker_array.markers.resize(clusters.size() + object_num.size());
	// marker_array.markers.resize(1);
	x_maxmin.clear();
	for(size_t i=0; i<clusters.size(); i++){
		// int i = 0;
		// boxを描写
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
		
		// boxのx座標の最大値と最小値を取り出す(IoUの計算のため)
		// rvizと実際の軸方向より　x↔y, y→-y というふうに変換する必要がある (なんで3Dplot系は縦方向がx軸なの?)
		if(maxPt.x > 0){
			x_maxmin.push_back(-1 * minPt.y);
			x_maxmin.push_back(minPt.x);
			x_maxmin.push_back(-1 * maxPt.y);
			x_maxmin.push_back(maxPt.x);
		}else{
			x_maxmin.push_back(-1 * minPt.y);
			x_maxmin.push_back(maxPt.x);
			x_maxmin.push_back(-1 * maxPt.y);
			x_maxmin.push_back(minPt.x);
		}
	}
	for(size_t part=0; part<object_num.size(); part++){
		int i = object_num.at(part);
		// int i = 0;
		// boxを描写
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

		marker_array.markers[i].color.r = 1.0f;
		marker_array.markers[i].color.g = 0.0f;
		marker_array.markers[i].color.b = 0.0f;
		marker_array.markers[i].color.a = 0.6f;
	}


	pub_marker.publish(marker_array);
	marker_array.markers.clear();
	// std::cout << "published!!!!!" << std::endl;
}

void Cam_lidar_detection::downsampling(void){
	vg.setInputCloud(cloud);
	vg.setLeafSize(leafsize, leafsize, leafsize);
	vg.filter(*voxcel_tmp);
	*cloud = *voxcel_tmp;
}

int main(int argc, char** argv){
    ros::init(argc, argv, "cam_lid_detection");
    Cam_lidar_detection cam_lidar_detection;
    cam_lidar_detection.run();

	//ros::spin();

    return 0;
}