#include <pcl/registration/registration.h>
#include <chrono>
#include <ros/ros.h>
#include <hdl_graph_slam/registrations.hpp>
#include <pcl/impl/point_types.hpp>
#include "Eigen/src/Core/Matrix.h"
#include "pcl/point_cloud.h"
#include "ros/init.h"
#include "ros/publisher.h"
#include <Eigen/Core>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>

using PointT = pcl::PointXYZINormal;

pcl::PointCloud<PointT>::Ptr load_cloud_from_disk(const std::string &filename) {
	pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
	if(pcl::io::loadPCDFile<PointT>(filename, *cloud) == -1) {
		std::cerr << "failed to load " << filename << std::endl;
		return nullptr;
	}
	return cloud;
}

Eigen::Matrix4f load_transform(const std::string &filename) {
	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	std::ifstream ifs(filename);
	if(ifs.good()) {
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				ifs >> transform(i, j);
			}
		}
		ifs.close();
	} else {
		std::cerr << "failed to load " << filename << std::endl;
	}
	return transform;
}

void pub_cloud(ros::Publisher &pub, pcl::PointCloud<PointT>::Ptr cloud) {
    sensor_msgs::PointCloud2 points_msg;
    pcl::toROSMsg(*cloud, points_msg);
	cloud->header.frame_id = "robotcar_0/base_link";
	pub.publish(cloud);
}

// main
int main(int argc, char** argv) {
	// init node and acquire private node handle
	ros::init(argc, argv, "loop_debugger");
	ros::NodeHandle pnh("~");

	pcl::Registration<PointT, PointT>::Ptr reg = hdl_graph_slam::select_registration_method(pnh);

	// load clouds from disk
	double fitness_score_max_range = pnh.param<double>("fitness_score_max_range", std::numeric_limits<double>::max());
	std::string loop_debug_dir = pnh.param<std::string>("loop_debug_output", "");
	std::string target_loop = pnh.param<std::string>("target_loop", "");
	loop_debug_dir = loop_debug_dir + "/" + target_loop;
	std::string target_cloud_fname = loop_debug_dir + "/target.pcd";
	std::string source_cloud_fname = loop_debug_dir + "/src.pcd";

	pcl::PointCloud<PointT>::Ptr target_cloud = load_cloud_from_disk(target_cloud_fname);
	pcl::PointCloud<PointT>::Ptr source_cloud = load_cloud_from_disk(source_cloud_fname);

	std::cout << "target_cloud: " << target_cloud->size() << " points" << std::endl;
	std::cout << "source_cloud: " << source_cloud->size() << " points" << std::endl;

	// load transform from disk
	Eigen::Matrix4f init_guess = load_transform(loop_debug_dir + "/init_guess.txt");
	std::cout << "init_guess: " << std::endl << init_guess << std::endl;

	Eigen::Matrix4f orig_tf = load_transform(loop_debug_dir + "/relative_pose.txt");
	std::cout << "orig_tf: " << std::endl << orig_tf << std::endl;

	pcl::PointCloud<PointT>::Ptr orig_aligned(new pcl::PointCloud<PointT>());
	pcl::transformPointCloud(*source_cloud, *orig_aligned, orig_tf);

	pcl::PointCloud<PointT>::Ptr guess_aligned(new pcl::PointCloud<PointT>());
	pcl::transformPointCloud(*source_cloud, *guess_aligned, init_guess);

	auto t0 = std::chrono::high_resolution_clock::now();
	// set input clouds
	reg->setInputSource(source_cloud);
	reg->setInputTarget(target_cloud);

	// run registration
	pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
	reg->align(*aligned, init_guess);

	auto t1 = std::chrono::high_resolution_clock::now();
	auto delta_t = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

	// get final transformation

	Eigen::Matrix4f final_transform = reg->getFinalTransformation();
	double fitness_score = reg->getFitnessScore(fitness_score_max_range);
	std::cout << "final_transform: " << std::endl << final_transform << std::endl;
	std::cout << "fitness_score: " << fitness_score << std::endl;
	std::cout << "has_converged: " << reg->hasConverged() << std::endl;
	std::cout << "took: " << delta_t << "ms" << std::endl;

	// write stats to file
	std::string reg_method = pnh.param<std::string>("registration_method", "NDT_OMP");
	std::ofstream ofs("/workspaces/collaborative-scene-graphs/src/loop_debugger/res/stats_" + reg_method + ".csv", std::ios::app);
	if (ofs.bad()) {
		std::cerr << "failed to open stats file" << std::endl;
		return 1;
	}
	ofs << target_loop << "," << fitness_score << "," << delta_t << std::endl;
	ofs.close();


	// publish clouds
	ros::Publisher target_pub = pnh.advertise<pcl::PointCloud<PointT>>("target_cloud", 1, true);
	ros::Publisher source_pub = pnh.advertise<pcl::PointCloud<PointT>>("source_cloud", 1, true);
	ros::Publisher aligned_pub = pnh.advertise<pcl::PointCloud<PointT>>("aligned_cloud", 1, true);
	ros::Publisher orig_aligned_pub = pnh.advertise<pcl::PointCloud<PointT>>("orig_aligned_cloud", 1, true);
	ros::Publisher guess_aligned_pub = pnh.advertise<pcl::PointCloud<PointT>>("guess_aligned_cloud", 1, true);

	pub_cloud(target_pub, target_cloud);
	pub_cloud(source_pub, source_cloud);
	pub_cloud(aligned_pub, aligned);
	pub_cloud(orig_aligned_pub, orig_aligned);
	pub_cloud(guess_aligned_pub, guess_aligned);

	if (pnh.param("keepalive", true)) {
		ros::spin();
	}

	ros::spinOnce();
	ros::shutdown();

	return 0;
	
}
