// SPDX-License-Identifier: BSD-2-Clause

#include <memory>
#include <chrono>
#include <iostream>
#include <hdl_graph_slam/custom_point_types.hpp>

#include <eigen_conversions/eigen_msg.h>
#include <ros/ros.h>
#include <ros/time.h>
#include <ros/duration.h>
#include <pcl_ros/point_cloud.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <std_msgs/Time.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

// TODO: remove
#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <hdl_graph_slam/ros_utils.hpp>
#include <hdl_graph_slam/registrations.hpp>
#include <hdl_graph_slam/ScanMatchingStatus.h>

namespace hdl_graph_slam {

class ScanMatchingOdometryNodelet : public nodelet::Nodelet {
  public:
    typedef pcl::PointXYZINormal PointT;
    typedef message_filters::sync_policies::ApproximateTime<geometry_msgs::PoseStamped, sensor_msgs::PointCloud2> ApproxSyncPolicy;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ScanMatchingOdometryNodelet() {}

    virtual ~ScanMatchingOdometryNodelet() {}

    virtual void onInit() {
        NODELET_DEBUG("initializing scan_matching_odometry_nodelet...");
        nh = getNodeHandle();
        private_nh = getPrivateNodeHandle();

        initialize_params();

        agent_no = std::to_string(private_nh.param<int>("agent_no", 42));
        points_topic = private_nh.param<std::string>("points_topic", "/velodyne_points_" + agent_no);
        odom_topic = private_nh.param<std::string>("odom_topic", "odom_" + agent_no);
        pose_topic = private_nh.param<std::string>("pose_topic", "/pose_" + agent_no);
        aligned_points_topic = private_nh.param<std::string>("aligned_points_topic", "/aligned_points_" + agent_no);
        filtered_points_topic = private_nh.param<std::string>("filtered_points_topic", "/filtered_points_" + agent_no);
        scan_matching_odometry_topic = private_nh.param<std::string>("scan_matching_odometry_topic", "/scan_matching_odometry_" + agent_no);

        if(private_nh.param<bool>("enable_imu_frontend", false)) {
            msf_pose_sub = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/msf_core/pose", 1, boost::bind(&ScanMatchingOdometryNodelet::msf_pose_callback, this, _1, false));
            msf_pose_after_update_sub = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/msf_core/pose_after_update", 1, boost::bind(&ScanMatchingOdometryNodelet::msf_pose_callback, this, _1, true));
        }



        if(private_nh.param<bool>("use_gps_ins_odom", false)) {
            ROS_INFO("Will use GPS/INS pose solution for odom");
            initial_pose.setIdentity();
            initial_pose_set = false;

            pose_sub = nh.subscribe(pose_topic, 32, &ScanMatchingOdometryNodelet::pose_callback, this);

        } else {
            ROS_INFO("Will use scan matching solution for odom");
            points_sub = nh.subscribe(filtered_points_topic, 1, &ScanMatchingOdometryNodelet::cloud_callback, this);
        }

        read_until_pub = nh.advertise<std_msgs::Header>(scan_matching_odometry_topic + "/read_until", 32);
        odom_pub = nh.advertise<nav_msgs::Odometry>(odom_topic, 32);
        trans_pub = nh.advertise<geometry_msgs::TransformStamped>(scan_matching_odometry_topic + "/transform", 32);
        status_pub = private_nh.advertise<ScanMatchingStatus>(scan_matching_odometry_topic + "/status", 8);
        aligned_points_pub = nh.advertise<sensor_msgs::PointCloud2>(aligned_points_topic, 32);

    }

  private:
    /**
     * @brief initialize parameters
     */
    void initialize_params() {
        auto &pnh = private_nh;

        odom_frame_id = pnh.param<std::string>("odom_frame_id", "odom_" + agent_no);
        base_link_frame_id = pnh.param<std::string>("base_link_frame_id", "robotcar_" + agent_no + "/base_link");

        robot_odom_frame_id = pnh.param<std::string>("robot_odom_frame_id", "robot_odom");

        // The minimum tranlational distance and rotation angle between keyframes.
        // If this value is zero, frames are always compared with the previous frame
        keyframe_delta_trans = pnh.param<double>("keyframe_delta_trans", 0.25);
        keyframe_delta_angle = pnh.param<double>("keyframe_delta_angle", 0.15);
        keyframe_delta_time = pnh.param<double>("keyframe_delta_time", 1.0);

        // Registration validation by thresholding
        transform_thresholding = pnh.param<bool>("transform_thresholding", false);
        max_acceptable_trans = pnh.param<double>("max_acceptable_trans", 1.0);
        max_acceptable_angle = pnh.param<double>("max_acceptable_angle", 1.0);

        // select a downsample method (VOXELGRID, APPROX_VOXELGRID, NONE)
        std::string downsample_method = pnh.param<std::string>("downsample_method", "VOXELGRID");
        double downsample_resolution = pnh.param<double>("downsample_resolution", 0.1);
        if(downsample_method == "VOXELGRID") {
            std::cout << "downsample: VOXELGRID " << downsample_resolution << std::endl;
            auto voxelgrid = new pcl::VoxelGrid<PointT>();
            voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
            downsample_filter.reset(voxelgrid);
        } else if(downsample_method == "APPROX_VOXELGRID") {
            std::cout << "downsample: APPROX_VOXELGRID " << downsample_resolution << std::endl;
            pcl::ApproximateVoxelGrid<PointT>::Ptr approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointT>());
            approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
            downsample_filter = approx_voxelgrid;
        } else {
            if(downsample_method != "NONE") {
                std::cerr << "warning: unknown downsampling type (" << downsample_method << ")" << std::endl;
                std::cerr << "       : use passthrough filter" << std::endl;
            }
            std::cout << "downsample: NONE" << std::endl;
            pcl::PassThrough<PointT>::Ptr passthrough(new pcl::PassThrough<PointT>());
            downsample_filter = passthrough;
        }

        registration = select_registration_method(pnh);
    }

    void pose_callback(const geometry_msgs::PoseStampedConstPtr &pose_msg) {
        if(!ros::ok()) {
            return;
        }

        if (!initial_pose_set) {
            tf::poseMsgToTF(pose_msg->pose, initial_pose);
            initial_pose_set = true;
        }

        tf::Pose pose_in_initial_frame;
        tf::Pose pose_in_world_frame;
        tf::poseMsgToTF(pose_msg->pose, pose_in_world_frame);
        pose_in_initial_frame = initial_pose.inverseTimes(pose_in_world_frame);

        tf::StampedTransform stamped_transform_in_initial_frame;
        stamped_transform_in_initial_frame.setData(pose_in_initial_frame);
        geometry_msgs::TransformStamped odom_trans;
        tf::transformStampedTFToMsg(stamped_transform_in_initial_frame, odom_trans);

        odom_trans.header.stamp = pose_msg->header.stamp;
        odom_trans.header.frame_id = odom_frame_id;
        odom_trans.child_frame_id = base_link_frame_id;

        trans_pub.publish(odom_trans);

        // broadcast the transform over tf
        odom_broadcaster.sendTransform(odom_trans);

        // publish the transform
        nav_msgs::Odometry odom;
        odom.header.stamp = pose_msg->header.stamp;
        odom.header.frame_id = odom_frame_id;

        tf::poseTFToMsg(pose_in_initial_frame, odom.pose.pose);

        odom_trans.child_frame_id = base_link_frame_id;
        odom.twist.twist.linear.x = 0.0;
        odom.twist.twist.linear.y = 0.0;
        odom.twist.twist.angular.z = 0.0;

        odom_pub.publish(odom);
    }

    /**
     * @brief callback for point clouds
     * @param cloud_msg  point cloud msg
     */
    void cloud_callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
        if(!ros::ok()) {
            return;
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
        pcl::fromROSMsg(*cloud_msg, *cloud);

        Eigen::Matrix4f pose = matching(cloud_msg->header.stamp, cloud);

        publish_odometry(cloud_msg->header.stamp, cloud_msg->header.frame_id, pose, agent_no);

        auto t2 = std::chrono::high_resolution_clock::now();
        auto runtime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        ROS_DEBUG_STREAM_THROTTLE(2.0, "Scan matching took " << runtime_ms.count() << " ms");

        // In offline estimation, point clouds until the published time will be supplied
        std_msgs::HeaderPtr read_until(new std_msgs::Header());
        read_until->frame_id = points_topic;
        read_until->stamp = cloud_msg->header.stamp + ros::Duration(1, 0);
        read_until_pub.publish(read_until);

        read_until->frame_id = filtered_points_topic;
        read_until_pub.publish(read_until);
    }

    void msf_pose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr &pose_msg, bool after_update) {
        if(after_update) {
            msf_pose_after_update = pose_msg;
        } else {
            msf_pose = pose_msg;
        }
    }

    /**
     * @brief downsample a point cloud
     * @param cloud  input cloud
     * @return downsampled point cloud
     */
    pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr &cloud) const {
        if(!downsample_filter) {
            return cloud;
        }

        pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());

        downsample_filter->setInputCloud(cloud);
        downsample_filter->filter(*filtered);

        return filtered;
    }

    /**
     * @brief estimate the relative pose between an input cloud and a keyframe cloud
     * @param stamp  the timestamp of the input cloud
     * @param cloud  the input cloud
     * @return the relative pose between the input cloud and the keyframe cloud
     */
    Eigen::Matrix4f matching(const ros::Time &stamp, const pcl::PointCloud<PointT>::ConstPtr &cloud) {
        if(!keyframe) {
            prev_time = ros::Time();
            prev_trans.setIdentity();
            keyframe_pose.setIdentity();
            keyframe_stamp = stamp;
            keyframe = downsample(cloud);
            registration->setInputTarget(keyframe);
            return Eigen::Matrix4f::Identity();
        }

        auto filtered = downsample(cloud);
        registration->setInputSource(filtered);

        std::string msf_source;
        Eigen::Isometry3f msf_delta = Eigen::Isometry3f::Identity();

        if(private_nh.param<bool>("enable_imu_frontend", false)) {
            if(msf_pose && msf_pose->header.stamp > keyframe_stamp && msf_pose_after_update &&
               msf_pose_after_update->header.stamp > keyframe_stamp) {
                Eigen::Isometry3d pose0 = pose2isometry(msf_pose_after_update->pose.pose);
                Eigen::Isometry3d pose1 = pose2isometry(msf_pose->pose.pose);
                Eigen::Isometry3d delta = pose0.inverse() * pose1;

                msf_source = "imu";
                msf_delta = delta.cast<float>();
            } else {
                std::cerr << "msf data is too old" << std::endl;
            }
        } else if(private_nh.param<bool>("enable_robot_odometry_init_guess", false) && !prev_time.isZero()) {
            tf::StampedTransform transform;
            if(tf_listener.waitForTransform(cloud->header.frame_id, stamp, cloud->header.frame_id, prev_time, robot_odom_frame_id, ros::Duration(0))) {
                tf_listener.lookupTransform(cloud->header.frame_id, stamp, cloud->header.frame_id, prev_time, robot_odom_frame_id, transform);
            } else if(tf_listener.waitForTransform(cloud->header.frame_id, ros::Time(0), cloud->header.frame_id, prev_time, robot_odom_frame_id, ros::Duration(0))) {
                tf_listener.lookupTransform(cloud->header.frame_id, ros::Time(0), cloud->header.frame_id, prev_time, robot_odom_frame_id, transform);
            }

            if(transform.stamp_.isZero()) {
                NODELET_WARN_STREAM("failed to look up transform between " << cloud->header.frame_id << " and " << robot_odom_frame_id);
            } else {
                msf_source = "odometry";
                msf_delta = tf2isometry(transform).cast<float>();
            }
        }

        // NODELET_INFO_STREAM_THROTTLE(2.0, "msf_source: " << msf_source);
        // NODELET_INFO_STREAM_THROTTLE(2.0, "msf_delta: " << std::endl << msf_delta.matrix());
        // NODELET_INFO_STREAM_THROTTLE(2.0, "delta t: " << (stamp - keyframe_stamp).toSec());

        pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
        registration->align(*aligned, prev_trans * msf_delta.matrix());

        publish_scan_matching_status(stamp, cloud->header.frame_id, aligned, msf_source, msf_delta);

        if(!registration->hasConverged()) {
            NODELET_INFO_STREAM("scan matching has not converged!!");
            NODELET_INFO_STREAM("ignore this frame(" << stamp << ")");
            return keyframe_pose * prev_trans;
        }

        Eigen::Matrix4f trans = registration->getFinalTransformation();
        // NODELET_INFO_STREAM_THROTTLE(2.0, "scan matching trans: " << std::endl << trans.matrix());
        // NODELET_INFO_STREAM_THROTTLE(2.0, "scan matching guess: " << std::endl << (prev_trans * msf_delta).matrix());
        Eigen::Matrix4f odom = keyframe_pose * trans;

        if(transform_thresholding) {
            Eigen::Matrix4f delta = prev_trans.inverse() * trans;
            double dx = delta.block<3, 1>(0, 3).norm();
            double da = std::acos(Eigen::Quaternionf(delta.block<3, 3>(0, 0)).w());

            if(dx > max_acceptable_trans || da > max_acceptable_angle) {
                NODELET_INFO_STREAM("too large transform!!  " << dx << "[m] " << da << "[rad]");
                NODELET_INFO_STREAM("ignore this frame(" << stamp << ")");
                return keyframe_pose * prev_trans;
            }
        }

        prev_time = stamp;
        prev_trans = trans;

        double delta_trans = trans.block<3, 1>(0, 3).norm();
        double delta_angle = std::acos(Eigen::Quaternionf(trans.block<3, 3>(0, 0)).w());
        double delta_time = (stamp - keyframe_stamp).toSec();
        if(delta_trans > keyframe_delta_trans || delta_angle > keyframe_delta_angle || delta_time > keyframe_delta_time) {
            ROS_DEBUG_STREAM("New scan matching keyframe: " << delta_trans << ", " << delta_angle << ", " << delta_time);
            keyframe = filtered;
            registration->setInputTarget(keyframe);

            keyframe_pose = odom;
            keyframe_stamp = stamp;
            prev_time = stamp;
            prev_trans.setIdentity();
        }

        if(aligned_points_pub.getNumSubscribers() > 0) {
            pcl::transformPointCloud(*cloud, *aligned, odom);
            aligned->header.frame_id = odom_frame_id;
            aligned_points_pub.publish(*aligned);
        }

        return odom;
    }

    /**
     * @brief publish odometry
     * @param stamp  timestamp
     * @param pose   odometry pose to be published
     */
    void publish_odometry(const ros::Time &stamp, const std::string &base_frame_id, const Eigen::Matrix4f &pose, const std::string agent_no) {
        // publish transform stamped for IMU integration
        geometry_msgs::TransformStamped odom_trans = matrix2transform(stamp, pose, odom_frame_id, base_link_frame_id);
        trans_pub.publish(odom_trans);

        // broadcast the transform over tf
        odom_broadcaster.sendTransform(odom_trans);

        // publish the transform
        nav_msgs::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = odom_frame_id;

        odom.pose.pose.position.x = pose(0, 3);
        odom.pose.pose.position.y = pose(1, 3);
        odom.pose.pose.position.z = pose(2, 3);
        odom.pose.pose.orientation = odom_trans.transform.rotation;

        odom.child_frame_id = base_frame_id;
        odom.twist.twist.linear.x = 0.0;
        odom.twist.twist.linear.y = 0.0;
        odom.twist.twist.angular.z = 0.0;

        odom_pub.publish(odom);
    }

    /**
     * @brief publish scan matching status
     */
    void publish_scan_matching_status(const ros::Time &stamp, const std::string &frame_id, pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr aligned, const std::string &msf_source, const Eigen::Isometry3f &msf_delta) {
        if(!status_pub.getNumSubscribers()) {
            return;
        }

        ScanMatchingStatus status;
        status.header.frame_id = frame_id;
        status.header.stamp = stamp;
        status.has_converged = registration->hasConverged();
        status.matching_error = registration->getFitnessScore();

        const double max_correspondence_dist = 0.5;

        // this determines inliers by checking distance between corresponding points is <0.5m.
        // TODO: try to use this to filter 'dynamic' points.
        int num_inliers = 0;
        std::vector<int> k_indices;
        std::vector<float> k_sq_dists;
        for(int i = 0; i < aligned->size(); i++) {
            const auto &pt = aligned->at(i);
            registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
            if(k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
                num_inliers++;
            }
        }
        status.inlier_fraction = static_cast<float>(num_inliers) / aligned->size();

        status.relative_pose = isometry2pose(Eigen::Isometry3f(registration->getFinalTransformation()).cast<double>());

        if(!msf_source.empty()) {
            status.prediction_labels.resize(1);
            status.prediction_labels[0].data = msf_source;

            status.prediction_errors.resize(1);
            Eigen::Isometry3f error = Eigen::Isometry3f(registration->getFinalTransformation()).inverse() * msf_delta;
            status.prediction_errors[0] = isometry2pose(error.cast<double>());
        }

        status_pub.publish(status);
    }

  private:
    // ROS topics
    ros::NodeHandle nh;
    ros::NodeHandle private_nh;

    tf::Pose initial_pose;
    bool initial_pose_set;

    ros::Subscriber points_sub;
    ros::Subscriber pose_sub;
    ros::Subscriber msf_pose_sub;
    ros::Subscriber msf_pose_after_update_sub;

    ros::Publisher odom_pub;
    ros::Publisher trans_pub;
    ros::Publisher aligned_points_pub;
    ros::Publisher status_pub;
    tf::TransformListener tf_listener;
    tf::TransformBroadcaster odom_broadcaster;
    tf::TransformBroadcaster keyframe_broadcaster;

    std::string points_topic;
    std::string pose_topic;
    std::string odom_topic;
    std::string filtered_points_topic;
    std::string aligned_points_topic;
    std::string agent_no;
    std::string scan_matching_odometry_topic;

    std::string odom_frame_id;
    std::string base_link_frame_id;
    std::string robot_odom_frame_id;
    ros::Publisher read_until_pub;

    // keyframe parameters
    double keyframe_delta_trans;  // minimum distance between keyframes
    double keyframe_delta_angle;  //
    double keyframe_delta_time;   //

    // registration validation by thresholding
    bool transform_thresholding;  //
    double max_acceptable_trans;  //
    double max_acceptable_angle;

    // odometry calculation
    geometry_msgs::PoseWithCovarianceStampedConstPtr msf_pose;
    geometry_msgs::PoseWithCovarianceStampedConstPtr msf_pose_after_update;

    ros::Time prev_time;
    Eigen::Matrix4f prev_trans;                  // previous estimated transform from keyframe
    Eigen::Matrix4f keyframe_pose;               // keyframe pose
    ros::Time keyframe_stamp;                    // keyframe time
    pcl::PointCloud<PointT>::ConstPtr keyframe;  // keyframe point cloud

    //
    pcl::Filter<PointT>::Ptr downsample_filter;
    pcl::Registration<PointT, PointT>::Ptr registration;
};  // namespace hdl_graph_slam

}  // namespace hdl_graph_slam

PLUGINLIB_EXPORT_CLASS(hdl_graph_slam::ScanMatchingOdometryNodelet, nodelet::Nodelet)
