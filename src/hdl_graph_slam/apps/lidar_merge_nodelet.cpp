#include <ctime>
#include <mutex>
#include <atomic>
#include <memory>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <unordered_map>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include "pcl/filters/voxel_grid.h"
#include <exception>

#include <ros/ros.h>
#include "pcl_ros/transforms.h"
#include "pcl_ros/impl/transforms.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>

#include <std_msgs/Time.h>
#include <std_msgs/String.h>
#include <nav_msgs/Odometry.h>
#include <nmea_msgs/Sentence.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <geographic_msgs/GeoPointStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <hdl_graph_slam/FloorCoeffs.h>

#include <hdl_graph_slam/SaveMap.h>
#include <hdl_graph_slam/DumpGraph.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

namespace hdl_graph_slam {

class LidarMergeNodelet : public nodelet::Nodelet {
  public:
    typedef pcl::PointXYZINormal PointT;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> ApproxSyncPolicy;

    LidarMergeNodelet() {}

    virtual ~LidarMergeNodelet() {}

    virtual void onInit() {
        nh = getNodeHandle();
        mt_nh = getMTNodeHandle();
        private_nh = getPrivateNodeHandle();

        // init parameters
        agent_no = private_nh.param<std::string>("agent_no", "101");

        points_topic = private_nh.param<std::string>("points_topic", "/robotcar_" + agent_no + "/lidar/merged");
        left_points_topic = private_nh.param<std::string>("left_points_topic", "/robotcar_" + agent_no + "/lidar/left");
        right_points_topic = private_nh.param<std::string>("right_points_topic", "/robotcar_" + agent_no + "/lidar/right");

        left_lidar_frame = private_nh.param<std::string>("left_lidar_frame", "/robotcar_" + agent_no + "/velodyne_left");
        right_lidar_frame = private_nh.param<std::string>("right_lidar_frame", "/robotcar_" + agent_no + "/velodyne_right");

        // subscribers
        left_cloud_sub.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(mt_nh, left_points_topic, 1));
        right_cloud_sub.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(mt_nh, right_points_topic, 1));
        sync.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(5), *left_cloud_sub, *right_cloud_sub));
        sync->registerCallback(boost::bind(&LidarMergeNodelet::cloud_callback, this, _1, _2));

        double downsample_resolution = 0.1;
        std::cout << "downsample: VOXELGRID " << downsample_resolution << std::endl;
        auto voxelgrid = new pcl::VoxelGrid<PointT>();
        voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
        downsample_filter.reset(voxelgrid);

        // publishers
        points_pub = mt_nh.advertise<sensor_msgs::PointCloud2>(points_topic, 1, true);
    }

  private:
    /**
     * @brief merge left and right point clouds, then publish
     * @param left_cloud_msg
     * @param right_cloud_msg
     */
    void cloud_callback(const sensor_msgs::PointCloud2::ConstPtr &left_cloud_msg, const sensor_msgs::PointCloud2::ConstPtr &right_cloud_msg) {
        double diff = left_cloud_msg->header.stamp.toSec() - right_cloud_msg->header.stamp.toSec();
        if(diff > 0.1) {
            ROS_WARN("Diff between left and right PC timestamps too large: %f", diff);
        }

        const ros::Time &left_stamp = left_cloud_msg->header.stamp;
        const ros::Time &right_stamp = right_cloud_msg->header.stamp;

        pcl::PointCloud<PointT>::Ptr left_cloud(new pcl::PointCloud<PointT>());
        pcl::fromROSMsg(*left_cloud_msg, *left_cloud);

        pcl::PointCloud<PointT>::Ptr right_cloud(new pcl::PointCloud<PointT>());
        pcl::fromROSMsg(*right_cloud_msg, *right_cloud);

        // apply filter
        // TODO: fix this whole transformation and filtering stuff. It is super
        // heavy, causing a high load and the scan matcher to lose tracking.
        pcl::PointCloud<PointT>::Ptr left_filtered = left_cloud;    // side_filter(left_cloud, "left");
        pcl::PointCloud<PointT>::Ptr right_filtered = right_cloud;  // side_filter(right_cloud, "right");

        // if (!tf_listener.canTransform(left_lidar_frame, left_stamp, right_lidar_frame, right_stamp, "world")) {
        //     NODELET_ERROR_STREAM("failed to find transform between " << left_cloud_msg->header.frame_id << " and " <<
        //     right_cloud_msg->header.frame_id); return;
        // }

        tf::StampedTransform transform;
        // tf_listener.waitForTransform(left_lidar_frame, left_stamp, right_lidar_frame, right_stamp, "world",
        // ros::Duration(1.0)); tf_listener.lookupTransform(left_lidar_frame, left_stamp, right_lidar_frame,
        // right_stamp, "world", transform); transform.setIdentity(); transform.setOrigin(tf::Vector3(0.0, 0.1, 0.0));

        // pcl::PointCloud<PointT>::Ptr right_cloud_in_left_frame(new pcl::PointCloud<PointT>());

        // pcl_ros::transformPointCloud(, *right_cloud_in_left_frame, transform);

        // sensor_msgs::PointCloud2Ptr test_cloud_msg(new sensor_msgs::PointCloud2());
        // pcl::toROSMsg(*right_cloud_in_left_frame, *test_cloud_msg);
        // points_pub.publish(*test_cloud_msg);
        // return;

        pcl::PointCloud<PointT>::Ptr merged_cloud(new pcl::PointCloud<PointT>());

        pcl::concatenate(*left_cloud, *right_cloud, *merged_cloud);

        merged_cloud->header.frame_id = left_cloud->header.frame_id;
        merged_cloud->header.stamp = left_cloud->header.stamp;

        // NODELET_INFO_STREAM("merge nodelet: left cloud of size " << left_cloud->size());
        // NODELET_INFO_STREAM("merge nodelet: right cloud of size " << right_cloud->size());
        // NODELET_INFO_STREAM("merge nodelet: transformed cloud of size " << right_cloud_in_left_frame->size());
        // NODELET_INFO_STREAM("merge nodelet: merged cloud of size " << merged_cloud->size());

        sensor_msgs::PointCloud2Ptr merged_cloud_msg(new sensor_msgs::PointCloud2());
        pcl::toROSMsg(*merged_cloud, *merged_cloud_msg);
        points_pub.publish(*merged_cloud_msg);
    }

    pcl::PointCloud<PointT>::Ptr side_filter(const pcl::PointCloud<PointT>::Ptr &cloud, std::string right_or_left) const {
        pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
        filtered->reserve(cloud->size());

        std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points), [&](const PointT &p) {
            auto v = p.getVector3fMap();
            bool use_pt = true;  // v.x() > 0.0;
            if(right_or_left == "left") {
                use_pt = use_pt && v.y() < 0.0;
            } else {
                use_pt = use_pt && v.y() > 0.0;
            }
            return use_pt;
        });

        filtered->width = filtered->size();
        filtered->height = 1;
        filtered->is_dense = false;

        filtered->header = cloud->header;

        return filtered;
    }

    pcl::PointCloud<PointT>::Ptr downsample(const pcl::PointCloud<PointT>::Ptr &cloud) const {
        if(!downsample_filter) {
            return cloud;
        }

        pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());

        downsample_filter->setInputCloud(cloud);
        downsample_filter->filter(*filtered);
        filtered->header = cloud->header;

        return filtered;
    }

    // private vars
    ros::NodeHandle nh;
    ros::NodeHandle mt_nh;
    ros::NodeHandle private_nh;

    std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> left_cloud_sub;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> right_cloud_sub;
    std::unique_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync;

    std::string points_topic;
    std::string right_points_topic;
    std::string left_points_topic;
    std::string left_lidar_frame;
    std::string right_lidar_frame;
    std::string agent_no;

    tf::TransformListener tf_listener;

    pcl::Filter<PointT>::Ptr downsample_filter;

    ros::Publisher points_pub;
};

}  // namespace hdl_graph_slam

PLUGINLIB_EXPORT_CLASS(hdl_graph_slam::LidarMergeNodelet, nodelet::Nodelet)
