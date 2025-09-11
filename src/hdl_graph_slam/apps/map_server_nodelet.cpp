// SPDX-License-Identifier: BSD-2-Clause

#include <atomic>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>

#include <eigen_conversions/eigen_msg.h>
#include <Eigen/Dense>

#include <g2o/core/robust_kernel_factory.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <nmea_msgs/Sentence.h>
#include <pcl/PolygonMesh.h>
#include <pcl/Vertices.h>
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/filters/random_sample.h>
#include <pcl/surface/convex_hull.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/impl/point_types.hpp>
#include <ros/console.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Header.h>
#include <std_msgs/String.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <visualization_msgs/MarkerArray.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <hdl_graph_slam/DumpGraph.h>
#include <hdl_graph_slam/FloorCoeffs.h>
#include <hdl_graph_slam/KeyframeArray_msg.h>
#include <hdl_graph_slam/Keyframe_msg.h>
#include <hdl_graph_slam/SaveMap.h>
#include <curb_projection/TrackedObjectObs.h>
#include <curb_projection/TrackedObjectObsArray.h>
#include <hdl_graph_slam/custom_point_types.hpp>
#include <hdl_graph_slam/graph_slam.hpp>
#include <hdl_graph_slam/information_matrix_calculator.hpp>
#include <hdl_graph_slam/keyframe.hpp>
#include <hdl_graph_slam/loop_detector.hpp>
#include <hdl_graph_slam/map_cloud_generator.hpp>
#include <hdl_graph_slam/ros_time_hash.hpp>
#include <hdl_graph_slam/ros_utils.hpp>
#include "hdl_graph_slam/dynamic_observation.hpp"

#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/edge_se3_plane.hpp>
#include <g2o/edge_se3_priorquat.hpp>
#include <g2o/edge_se3_priorvec.hpp>
#include <g2o/edge_se3_priorxy.hpp>
#include <g2o/edge_se3_priorxyz.hpp>
#include <vector>

template<typename T>
std::string type_name();

namespace hdl_graph_slam {

class MapServerNodelet : public nodelet::Nodelet {
  public:
    typedef pcl::PointXYZINormal PointT;
    typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::PointCloud2> ApproxSyncPolicy;

    MapServerNodelet() {}

    virtual ~MapServerNodelet() {}

    virtual void onInit() {
        ms_nh = getNodeHandle();
        ms_mt_nh = getMTNodeHandle();
        ms_private_nh = getPrivateNodeHandle();

        // init parameters
        map_server_topic = ms_private_nh.param<std::string>("map_server_topic", "/map_server");
        map_frame_id = ms_private_nh.param<std::string>("map_frame_id", "world");

        map_cloud_resolution = ms_private_nh.param<double>("map_cloud_resolution", 0.6);
        map_cloud_time_shading = ms_private_nh.param<bool>("map_cloud_time_shading", false);

        loop_detection = ms_private_nh.param<bool>("loop_detection", true);
        remove_dyn_pts = ms_private_nh.param<bool>("remove_dyn_pts", true);

        agent_keyframe_topic = ms_private_nh.param<std::string>("agent_keyframe_topic", "/agent_keyframes");
        dynamic_observation_topic = ms_private_nh.param<std::string>("dynamic_observation_topic", "/dynamic_observation");

        height_offset_keyframes = ms_private_nh.param<double>("height_offset_keyframes", 10.0);

        loop_closure_edge_robust_kernel = ms_private_nh.param<std::string>("loop_closure_edge_robust_kernel", "NONE");
        loop_closure_edge_robust_kernel_size = ms_private_nh.param<double>("loop_closure_edge_robust_kernel_size", 1.0);

        graph_slam.reset(new GraphSLAM(ms_private_nh.param<std::string>("g2o_solver_type", "lm_var")));
        loop_detector.reset(new hdl_graph_slam::LoopDetector(ms_private_nh));
        map_cloud_generator.reset(new hdl_graph_slam::MapCloudGenerator());
        inf_calclator.reset(new hdl_graph_slam::InformationMatrixCalculator(ms_private_nh));

        // subscribers
        keyframe_msg_sub = ms_nh.subscribe(agent_keyframe_topic, 1024, &hdl_graph_slam::MapServerNodelet::keyframe_msg_callback, this);
        tracked_object_sub = ms_nh.subscribe(dynamic_observation_topic, 1024, &hdl_graph_slam::MapServerNodelet::tracked_object_obs_callback, this);
        ms_command_sub = ms_nh.subscribe(map_server_topic + "/command", 1024, &hdl_graph_slam::MapServerNodelet::ms_command_callback, this);

        // publishers
        markers_pub = ms_mt_nh.advertise<visualization_msgs::MarkerArray>(map_server_topic + "/markers", 16);
        map_points_pub = ms_mt_nh.advertise<sensor_msgs::PointCloud2>(map_server_topic + "/map_points", 1, true);
        optimized_keyframes_pub = ms_mt_nh.advertise<hdl_graph_slam::KeyframeArray_msg>("/optimized_keyframes", 32, true);
        graph_changed_pub = ms_mt_nh.advertise<std_msgs::Bool>("/graph_changed", 1, true);

        graph_updated = false;
        double graph_update_interval = ms_private_nh.param<double>("graph_update_interval", 3.0);
        double map_cloud_update_interval = ms_private_nh.param<double>("map_cloud_update_interval", 10.0);
        ms_optimization_timer = ms_mt_nh.createWallTimer(ros::WallDuration(graph_update_interval), &hdl_graph_slam::MapServerNodelet::ms_optimization_timer_callback, this);
        ms_map_publish_timer = ms_mt_nh.createWallTimer(ros::WallDuration(map_cloud_update_interval), &hdl_graph_slam::MapServerNodelet::ms_map_points_publish_timer_callback, this);

        keyframe_metric_pub_timer = ms_mt_nh.createWallTimer(ros::WallDuration(10.0), &hdl_graph_slam::MapServerNodelet::publish_keyframes_for_metric_timer_callback, this);

        for(int i = 0; i < 10; i++) {
            setInitialPosition[i] = true;
            initial_pose[i] = Eigen::Isometry3d::Identity();
        }

        large_loop_detected = false;

        g2o::RobustKernel* test_kernel = graph_slam->robust_kernel_factory->construct(loop_closure_edge_robust_kernel);
        if(test_kernel == nullptr) {
            ROS_ERROR("... invalid robust kernel type: %s", loop_closure_edge_robust_kernel.c_str());
            EXIT_FAILURE;
        }
        ROS_INFO("Loop closure robust kernel: %s, size: %f", loop_closure_edge_robust_kernel.c_str(), loop_closure_edge_robust_kernel_size);

        ROS_INFO("Init done.");
    }

  private:
    /**
     * @brief received point clouds are pushed to #keyframe_queue
     * @param odom_msg
     * @param cloud_msg
     */
    void keyframe_msg_callback(const hdl_graph_slam::Keyframe_msg& keyframe_msg) {
        const ros::Time& stamp = keyframe_msg.header.stamp;

        ROS_DEBUG("New keyframe: ID=%d stamp=%f", keyframe_msg.id, stamp.toSec());

        if(std::find(registeredAgents.begin(), registeredAgents.end(), keyframe_msg.agent_no) == registeredAgents.end()) {
            registeredAgents.push_back(keyframe_msg.agent_no);
            latest_dyn_obs[keyframe_msg.agent_no] = -1;
        }

        pcl::PointCloud<PointT>::Ptr cloud_from_msg(new pcl::PointCloud<PointT>());

        pcl::MsgFieldMap field_map;
        pcl::createMapping<PointT>(keyframe_msg.cloud.fields, field_map);
        pcl::fromROSMsg(keyframe_msg.cloud, *cloud_from_msg);

        Eigen::Isometry3d kf_odom =
            Eigen::Translation3d(keyframe_msg.odom.position.x, keyframe_msg.odom.position.y, keyframe_msg.odom.position.z) *
            Eigen::Quaterniond(keyframe_msg.odom.orientation.w, keyframe_msg.odom.orientation.x,
                               keyframe_msg.odom.orientation.y, keyframe_msg.odom.orientation.z);

        if(isIdentity(kf_odom)) {
            ROS_WARN("  ##########   incoming odom is identity   ##########  ");
            ROS_WARN_STREAM("Agent: " << keyframe_msg.agent_no << " id: " << keyframe_msg.id);
            ROS_WARN("  ##########  ");
        }

        // if(keyframes[keyframe_msg.agent_no].empty() && new_keyframes[keyframe_msg.agent_no].empty() &&
        //    keyframe_queue[keyframe_msg.agent_no].empty()) {
        if(keyframe_msg.id % 1000000 == 0) {
            // this is the first keyframe received for this agent, set initial pose
            tf::StampedTransform pose;
            tf::poseMsgToTF(keyframe_msg.initial, pose);
            tf::poseMsgToEigen(keyframe_msg.initial, initial_pose[keyframe_msg.agent_no]);
            // tf_listener.waitForTransform("world", "robotcar_" + std::to_string(keyframe_msg.agent_no) + "/initial", ros::Time(0), ros::Duration(10.0));
            // tf_listener.lookupTransform("world", "robotcar_" + std::to_string(keyframe_msg.agent_no) + "/initial", ros::Time(0), pose);
            // tf::poseTFToEigen(pose, initial_pose[keyframe_msg.agent_no]);

            ROS_INFO("HDL Map Server: Agent %d Initial Pose received and set: trans=(%f, %f, %f) rot=(%f, %f, %f, %f)",
                     keyframe_msg.agent_no, pose.getOrigin().getX(), pose.getOrigin().getY(), pose.getOrigin().getZ(),
                     pose.getRotation().getX(), pose.getRotation().getY(), pose.getRotation().getZ(), pose.getRotation().getW());
        }

        KeyFrame::Ptr keyframe(new KeyFrame(stamp, keyframe_msg.id, kf_odom, keyframe_msg.accum_distance, cloud_from_msg));
        keyframe_queue[keyframe_msg.agent_no].push_back(keyframe);

        ROS_INFO("New Keyframe received - Agent %i - ID: %i - Queue: %li - Queued for LC check: %li, Processed KFs: %li",
                 keyframe_msg.agent_no, keyframe_msg.id, keyframe_queue[keyframe_msg.agent_no].size(),
                 new_keyframes[keyframe_msg.agent_no].size(), keyframes[keyframe_msg.agent_no].size());
    }

    hdl_graph_slam::KeyFrame::Ptr findKeyframeById(int id) const {
        for(int a = 0; a < registeredAgents.size(); a++) {
            for(int i = 0; i < keyframes[registeredAgents[a]].size(); i++) {
                if(keyframes[registeredAgents[a]][i]->keyframe_id == id) {
                    return keyframes[registeredAgents[a]][i];
                }
            }
        }
        return nullptr;
    }

    bool isIdentity(const Eigen::Isometry3d& transform) {
        Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
        return transform.isApprox(identity);
    }

    std::tuple<int, int> findKeyframeIndexById(int id) const {
        std::cout << "findKeyframeIndexById: " << id << std::endl;
        for(int a = 0; a < registeredAgents.size(); a++) {
            for(int i = 0; i < keyframes[registeredAgents[a]].size(); i++) {
                if(keyframes[registeredAgents[a]][i]->keyframe_id == id) {
                    return std::make_tuple(a, i);
                }
            }
        }
        std::cout << "no keyframe index found for the ID: " << id << std::endl;
        return std::make_tuple(0, 999999);
        ;
    }

    std::tuple<int, int> findKeyframeIndexByNodeId(int id) const {
        std::cout << "findKeyframeIndexByNodeId: " << id << std::endl;
        for(int a = 0; a < registeredAgents.size(); a++) {
            for(int i = 0; i < keyframes[registeredAgents[a]].size(); i++) {
                if(keyframes[registeredAgents[a]][i]->node->id() == id) {
                    std::cout << "Found! Agent: " << a << " Index: " << i << std::endl;
                    return std::make_tuple(a, i);
                }
            }
        }
        std::cout << "no keyframe index found for the node ID: " << id << std::endl;
        return std::make_tuple(0, 999999);
        ;
    }

    hdl_graph_slam::KeyFrame::Ptr findNewKeyframeById(int id) const {
        for(int a = 0; a < registeredAgents.size(); a++) {
            for(int i = 0; i < new_keyframes[a].size(); i++) {
                if(new_keyframes[a][i]->keyframe_id == id) {
                    return new_keyframes[a][i];
                }
            }
        }
        std::cout << "no new_keyframe found for the ID: " << id << std::endl;
        return nullptr;
    }

    /**
     * @brief received point clouds are pushed to #dynamic object queue
     */
    void tracked_object_obs_callback(curb_projection::TrackedObjectObsArrayConstPtr observations) {
        for(const curb_projection::TrackedObjectObs& obs_msg : observations->observations) {
            DynObservation::Ptr obs(new DynObservation(obs_msg));
            obs->keyframe = findKeyframeById(obs_msg.keyframe_id);
            dynamic_observations_q.push_back(obs);
        }
    }

    void remove_dynamic_points() {
        if(!remove_dyn_pts) {
            ROS_WARN_THROTTLE(20.0, "Dynamic point removal off");
            return;
        }

        // Iterate over the deque, try to find keyframes, remove the dynamic
        // points, and remove successful observations
        for(auto it = dynamic_observations_q.begin(); it != dynamic_observations_q.end();) {
            DynObservation* obs = it->get();

            hdl_graph_slam::KeyFrame::Ptr keyframe = findNewKeyframeById(obs->keyframe_id);
            if(!keyframe) {
                ROS_WARN_THROTTLE(1.0, "mapserver dynpt: keyframe %d not found in new keyframes", obs->keyframe_id);
                it++;
                continue;
            }

            pcl::CropBox<PointT> box_filter;
            obs->getBoxFilter(box_filter);

            box_filter.setInputCloud(keyframe->cloud);
            pcl::PointCloud<PointT> filtered_cloud;
            box_filter.filter(filtered_cloud);
            keyframe->cloud = filtered_cloud.makeShared();

            it = dynamic_observations_q.erase(it);

            latest_dyn_obs[obs->observing_agent_id] = obs->keyframe_id;
        }
    }

    void ms_command_callback(const std_msgs::String::ConstPtr& msg) {
        std::string command = msg->data.c_str();
        ROS_INFO("Received Command: [%s]", msg->data.c_str());
        if(!command.compare("load_graph")) {
            ROS_DEBUG("Loading graph..");

            ROS_DEBUG("Successful");
            ms_optimization_timer_callback_func();
            ms_map_points_publish_timer_callback_func();
        } else if(!command.compare("pub_graph")) {
            ROS_INFO("Publishing graph");
            graph_slam->save("graph_agent");
        } else if(!command.compare("pub_graph")) {
            ROS_INFO("Publishing graph");
            graph_slam->save("graph_agent");
        }
    }

    /**
     * @brief this method adds all the keyframes in #keyframe_queue to the pose graph (odometry edges)
     * @return if true, at least one keyframe was added to the pose graph
     */
    bool flush_keyframe_queue() {
        ROS_DEBUG("flush keyframe");
        bool graph_changed = false;

        for(int a = 0; a < registeredAgents.size(); a++) {
            if(keyframe_queue[registeredAgents[a]].empty()) {
                continue;
            }

            for(int i = 0; i < keyframe_queue[registeredAgents[a]].size(); i++) {
                const auto& keyframe = keyframe_queue[registeredAgents[a]][i];

                // calculate pose in the frame of the first node
                Eigen::Isometry3d odom_init(initial_pose[registeredAgents[a]].matrix().cast<double>());
                Eigen::Isometry3d odom = odom_init * keyframe->odom;

                // if this is the first time visiting this place,
                // set the first node to the initial pose and fix this one.
                if(setInitialPosition[registeredAgents[a]]) {
                    // add first node for this agent
                    keyframe->node = graph_slam->add_se3_node(initial_pose[registeredAgents[a]]);
                    // only fix the first agent's node
                    if(keyframe->keyframe_id / 1000000 == 0) {
                        keyframe->node->setFixed(true);
                        agent_maps_merged.insert(0);
                    }
                    setInitialPosition[registeredAgents[a]] = false;
                    // no need to add edges, can continue
                    graph_changed = true;
                    continue;
                }

                // add regular node
                keyframe->node = graph_slam->add_se3_node(odom);
                graph_changed = true;

                // add edge between consecutive keyframes
                // first find previous keyframe for this agent, need to check three queues:
                // keyframe_queue (newest), new_keyframes (new), keyframes (fully processed)
                KeyFrame::Ptr prev_keyframe;
                if(i == 0) {
                    // first keyframe in this queue, find most recent keyframe either in new_keyframes or in keyframes queue
                    if(!new_keyframes[registeredAgents[a]].empty())
                        prev_keyframe = new_keyframes[registeredAgents[a]].back();
                    else
                        prev_keyframe = keyframes[registeredAgents[a]].back();
                } else {
                    // use the previous element in keyframe_queue
                    prev_keyframe = keyframe_queue[registeredAgents[a]][i - 1];
                }

                // now we have the immediate predecessor, add an odometry edge
                Eigen::Isometry3d relative_pose = keyframe->odom.inverse() * prev_keyframe->odom;
                Eigen::MatrixXd information = inf_calclator->calc_information_matrix(keyframe->cloud, prev_keyframe->cloud, relative_pose);
                graph_slam->add_se3_edge(keyframe->node, prev_keyframe->node, relative_pose, information);
            }

            // move these keyframes to new_keyframes queue for loop closure checking
            for(auto& keyframe : keyframe_queue[registeredAgents[a]]) {
                // new_keyframes will be tested later for loop closure
                new_keyframes[registeredAgents[a]].push_back(keyframe);
            }

            // all received keyframes are added, clear the queue
            keyframe_queue[registeredAgents[a]].clear();
        }
        return graph_changed;
    }

    void ms_map_points_publish_timer_callback(const ros::WallTimerEvent& event) {
        ms_map_points_publish_timer_callback_func();
    }

    /**
     * @brief generate map point cloud and publish it
     * @param event
     */
    void ms_map_points_publish_timer_callback_func() {
        ROS_INFO("MAP POINTS PUBLISH TIMER CALLBACK");
        if(!map_points_pub.getNumSubscribers()) {
            ROS_INFO("return: No subscribers");
            return;
        }

        if(!graph_updated) {
            ROS_INFO("return: Graph not updated");
            return;
        }

        std::vector<hdl_graph_slam::KeyFrameSnapshot::Ptr> snapshot;

        keyframes_snapshot_mutex.lock();
        snapshot = keyframes_snapshot;
        keyframes_snapshot_mutex.unlock();

        std::vector<int> reduce_classes = {7, 8, 9};

        ROS_INFO("Map generation...");
        auto cloud = map_cloud_generator->generate(snapshot, map_cloud_resolution, map_cloud_time_shading);
        if(!cloud) {
            ROS_INFO("return: No cloud generated.");
            return;
        } else {
            ROS_INFO("Map generation done.");
        }

        cloud->header.frame_id = map_frame_id;
        cloud->header.stamp = snapshot.back()->cloud->header.stamp;

        sensor_msgs::PointCloud2Ptr cloud_msg(new sensor_msgs::PointCloud2());
        pcl::toROSMsg(*cloud, *cloud_msg);

        map_points_pub.publish(cloud_msg);
        ROS_DEBUG("Map Points published.");
    }

    void ms_optimization_timer_callback(const ros::WallTimerEvent& event) {
        ms_optimization_timer_callback_func();
    }

    bool translationDifferenceExceedsOne(const Eigen::Isometry3d& iso1, const Eigen::Isometry3d& iso2) {
        // Subtract the translation vectors
        Eigen::Vector3d difference = iso1.translation() - iso2.translation();

        // Check if the absolute difference in x and y components is greater than 1
        return (std::abs(difference.x()) > 1.0) || (std::abs(difference.y()) > 1.0);
    }

    void publish_keyframe_tfs() {
        std::vector<KeyFrame::Ptr> all_keyframes;
        for(int a = 0; a < registeredAgents.size(); a++) {
            for(int i = 0; i < keyframes[registeredAgents[a]].size(); i++) {
                all_keyframes.push_back(keyframes[registeredAgents[a]][i]);
            }
        }

        for(auto kf : all_keyframes) {
            const Eigen::Isometry3d keyframe_tf = kf->node->estimate();
            tf::StampedTransform stamped_tf;
            tf::transformEigenToTF(keyframe_tf, stamped_tf);
            stamped_tf.stamp_ = ros::Time::now();
            stamped_tf.frame_id_ = "world";
            stamped_tf.child_frame_id_ = "keyframe/" + std::to_string(kf->keyframe_id);
            tf_broadcaster.sendTransform(stamped_tf);
        }
    }

    void publish_keyframes_for_metric_timer_callback(const ros::WallTimerEvent& event) {
        publish_keyframes_for_metric();
    }

    void test() {}

    void publish_keyframes_for_metric() {
        // this function publishes all _keyframe_ poses, unlike
        // the keyframe_metric_pub, which publishes observations

        std::vector<KeyFrame::Ptr> all_keyframes;
        for(int a = 0; a < registeredAgents.size(); a++) {
            for(int i = 0; i < keyframes[registeredAgents[a]].size(); i++) {
                all_keyframes.push_back(keyframes[registeredAgents[a]][i]);
            }
        }

        hdl_graph_slam::KeyframeArray_msg keyframe_array;
        keyframe_array.header.frame_id = "world";
        keyframe_array.header.stamp = ros::Time::now();

        for(auto kf : all_keyframes) {
            hdl_graph_slam::Keyframe_msg keyframe_msg;
            keyframe_msg.header.frame_id = "world";
            keyframe_msg.header.stamp = kf->stamp;

            keyframe_msg.accum_distance = kf->accum_distance;
            keyframe_msg.agent_no = static_cast<int>(floor(kf->keyframe_id / 1000000));
            keyframe_msg.id = kf->keyframe_id;

            tf::poseEigenToMsg(initial_pose[keyframe_msg.agent_no], keyframe_msg.initial);

            geometry_msgs::Pose kf_odom_msg;
            tf::poseEigenToMsg(kf->estimate(), kf_odom_msg);
            keyframe_msg.odom = kf_odom_msg;

            // don't send the point cloud for better performance
            // keyframe_msg.cloud = kf->cloud;

            keyframe_array.keyframes.push_back(keyframe_msg);
        }

        optimized_keyframes_pub.publish(keyframe_array);
    }

    /**
     * @brief this methods adds all the data in the queues to the pose graph, and then optimizes the pose graph
     * @param event
     */
    void ms_optimization_timer_callback_func() {
        ROS_INFO("OPTIMIZATION TIMER CALLBACK");

        std::lock_guard<std::mutex> lock(main_thread_mutex);

        // add keyframes and floor coeffs in the queues to the pose graph
        bool keyframe_updated = flush_keyframe_queue();

        if(!keyframe_updated) {
            ROS_INFO("Optimization Callback: Return: Keyframes not updated.");
            return;
        }

        remove_dynamic_points();

        // loop detection
        std::vector<KeyFrame::Ptr> all_keyframes;

        for(int a = 0; a < registeredAgents.size(); a++) {
            for(int i = 0; i < keyframes[registeredAgents[a]].size(); i++) {
                all_keyframes.push_back(keyframes[registeredAgents[a]][i]);
            }
        }

        int num_iterations1 = ms_private_nh.param<int>("g2o_solver_num_iterations", 1024);
        graph_slam->optimize(num_iterations1);
        if(large_loop_detected) {
            // large loop was detected last round and now the graph has been
            // optimized, tell the other nodes
            pub_graph_changed();
            large_loop_detected = false;
        }

        for(int a = 0; a < registeredAgents.size(); a++) {
            int latest_dyn_obs_keyframe = latest_dyn_obs[a];

            // iterate over new_keyframes queue from front, removing oldest keyframes first
            while(!new_keyframes[registeredAgents[a]].empty()) {
                KeyFrame::Ptr new_keyframe = new_keyframes[registeredAgents[a]].front();

                // check if dynamic object removal has reached this keyframe yet
                if(remove_dyn_pts && new_keyframe->keyframe_id > latest_dyn_obs_keyframe) {
                    break;
                }

                // now we know that this keyframe either has no dynamic objects
                // or they have been removed, so we can check for loops
                if(loop_detection) {
                    loop_detector->check_new_keyframes(all_keyframes, new_keyframe, *graph_slam, registeredAgents[a]);
                } else {
                    ROS_WARN_THROTTLE(10.0, "LOOP DETECTION DISABLED!");
                }

                // copy to list of keyframes for this agent and remove this keyframe from the queue
                keyframes[registeredAgents[a]].push_back(new_keyframe);
                new_keyframes[registeredAgents[a]].pop_front();
            }
        }

        // obtain new confirmed loop closures
        std::vector<Loop> new_loops = loop_detector->get_detected_loops();

        for(const auto& loop : new_loops) {
            ROS_INFO("\n ----- LOOP FOUND (%d) ----- ", loop.source);
            Eigen::Isometry3d relpose(loop.relative_pose.cast<double>());

            int first_agent_id = static_cast<int>(floor(loop.src_kf->keyframe_id / 1000000));
            int second_agent_id = static_cast<int>(floor(loop.target_kf->keyframe_id / 1000000));
            bool first_loop_closure = true;
            if(first_agent_id != second_agent_id && (first_agent_id == 0 || second_agent_id == 0)) {
                // inter-agent lc with the main agent, check if this is the first one
                if(first_agent_id == 0 && agent_maps_merged.find(second_agent_id) == agent_maps_merged.end()) {
                    ROS_INFO("First loop closure between agent 0 and agent %d", second_agent_id);
                    agent_maps_merged.insert(second_agent_id);
                    first_loop_closure = false;
                } else if (second_agent_id == 0 && agent_maps_merged.find(first_agent_id) == agent_maps_merged.end()) {
                    ROS_INFO("First loop closure between agent 0 and agent %d", first_agent_id);
                    agent_maps_merged.insert(first_agent_id);
                    first_loop_closure = false;
                }
            }

            // compute relative transform
            Eigen::Isometry3d src_pose = loop.src_kf->node->estimate();
            Eigen::Isometry3d tgt_pose = loop.target_kf->node->estimate();
            Eigen::Isometry3d current_relpose = tgt_pose.inverse() * src_pose;
            Eigen::Affine3d diff_relposes((current_relpose.inverse() * relpose).matrix());
            tf::Transform t_tf;
            tf::transformEigenToTF(diff_relposes, t_tf);

            if (!first_loop_closure) {
                // check if the rotation angle is consistent with the previous estimate
                tf::Quaternion q = t_tf.getRotation();
                double rot_angle = q.getAngle();
                if(rot_angle > 1.2) {
                    ROS_WARN("Proposed loop rotation part is inconsistent with current estimate, angle diff: %f", rot_angle);
                    continue;
                }
            }

            double rel_dist = t_tf.getOrigin().length();
            if(rel_dist > 0.5) {
                large_loop_detected = true;
            }

            ROS_INFO_STREAM("Keyframe " << loop.src_kf->keyframe_id << " -> " << loop.target_kf->keyframe_id << std::endl
                                        << relpose.matrix() << std::endl);
            Eigen::MatrixXd information_matrix =
                inf_calclator->calc_information_matrix(loop.target_kf->cloud, loop.src_kf->cloud, relpose);

            // this amplifies the loop closure in the optimization?
            // information_matrix.topLeftCorner(3, 3).array() *= 100;


            auto edge = graph_slam->add_se3_edge(loop.target_kf->node, loop.src_kf->node, relpose, information_matrix);
            if(first_loop_closure) {
                // this is not the first inter-agent loop closure, use the robust kernel
                graph_slam->add_robust_kernel(edge, loop_closure_edge_robust_kernel, loop_closure_edge_robust_kernel_size);
            }
            detected_loop_closures.push_back(loop);
        }

        std::vector<KeyFrameSnapshot::Ptr> snapshot(all_keyframes.size());
        std::transform(all_keyframes.begin(), all_keyframes.end(), snapshot.begin(), [=](const KeyFrame::Ptr& k) {
            auto s = std::make_shared<KeyFrameSnapshot>(k);
            return s;
        });

        keyframes_snapshot_mutex.lock();
        keyframes_snapshot.swap(snapshot);
        keyframes_snapshot_mutex.unlock();
        graph_updated = true;

        publish_keyframe_tfs();
        pub_marker_array();
    }

    void pub_graph_changed() {
        ROS_INFO("Publishing graph changed");
        std_msgs::Bool msg;
        msg.data = true;
        graph_changed_pub.publish(msg);
    }

    /**
     * create and publish array containg visualization markers
     */
    void pub_marker_array() const {
        float z_offset = ms_private_nh.param<float>("marker_z_offset", 0.0);
        float scale_factor = ms_private_nh.param<float>("marker_scale_factor", 1.0);

        visualization_msgs::MarkerArray markers;
        ros::Time stamp = ros::Time::now();
        int marker_id = 0;

        // node markers
        visualization_msgs::Marker traj_marker;
        traj_marker.header.frame_id = map_frame_id;
        traj_marker.header.stamp = stamp;
        traj_marker.ns = "node_markers";
        traj_marker.id = marker_id++;
        traj_marker.type = visualization_msgs::Marker::SPHERE_LIST;

        traj_marker.pose.orientation.w = 1.0;
        traj_marker.scale.x = traj_marker.scale.y = traj_marker.scale.z = 2.0 * scale_factor;

        for(int a = 0; a < registeredAgents.size(); a++) {
            for(KeyFrame::Ptr kf : keyframes[registeredAgents[a]]) {
                Eigen::Vector3d pos = kf->node->estimate().translation();
                geometry_msgs::Point p;
                tf::pointEigenToMsg(pos, p);
                p.z += z_offset;
                traj_marker.points.push_back(p);
                std_msgs::ColorRGBA color;
                color.a = 1.0;
                color.r = 0.4;
                color.g = 0.8;
                color.b = 0.0;
                traj_marker.colors.push_back(color);
            }
        }

        markers.markers.push_back(traj_marker);

        // node id labels
        for(int a = 0; a < registeredAgents.size(); a++) {
            for(KeyFrame::Ptr kf : keyframes[registeredAgents[a]]) {
                visualization_msgs::Marker label;
                label.header = traj_marker.header;
                label.ns = "kf_id_markers";
                label.action = visualization_msgs::Marker::ADD;
                label.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
                label.color.a = 1.0;
                label.color.r = 0.6;
                label.color.g = 0.6;
                label.color.b = 0.6;
                label.scale.z = 1.5 * scale_factor;
                label.id = kf->keyframe_id;
                auto p = kf->node->estimate().translation();
                label.pose.position.x = p.x();
                label.pose.position.y = p.y();
                label.pose.position.z = p.z() - 5.0 + z_offset;
                label.text = std::to_string(kf->keyframe_id);
                markers.markers.push_back(label);
            }
        }

        // edge markers
        visualization_msgs::Marker edge_marker;
        edge_marker.header.frame_id = map_frame_id;
        edge_marker.header.stamp = stamp;
        edge_marker.ns = "edge_markers";
        edge_marker.id = 2;
        edge_marker.type = visualization_msgs::Marker::LINE_LIST;

        edge_marker.pose.orientation.w = 1.0;
        edge_marker.scale.x = 0.3 * scale_factor;
        edge_marker.pose.position.z = z_offset;

        auto edge_itr = graph_slam->graph->edges().begin();
        for(int i = 0; edge_itr != graph_slam->graph->edges().end(); edge_itr++, i++) {
            g2o::HyperGraph::Edge* edge = *edge_itr;
            g2o::EdgeSE3* edge_se3 = dynamic_cast<g2o::EdgeSE3*>(edge);

            g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_se3->vertices()[0]);
            g2o::VertexSE3* v2 = dynamic_cast<g2o::VertexSE3*>(edge_se3->vertices()[1]);
            Eigen::Vector3d pt1 = v1->estimate().translation();
            Eigen::Vector3d pt2 = v2->estimate().translation();

            geometry_msgs::Point pt1_msg, pt2_msg;

            pt1_msg.z += height_offset_keyframes;
            pt2_msg.z += height_offset_keyframes;

            tf::pointEigenToMsg(pt1, pt1_msg);
            tf::pointEigenToMsg(pt2, pt2_msg);

            edge_marker.points.push_back(pt1_msg);
            edge_marker.points.push_back(pt2_msg);

            std_msgs::ColorRGBA edge_color;
            edge_color.a = 1.0;
            edge_color.r = 0.3;
            edge_color.g = 0.7;
            edge_marker.colors.push_back(edge_color);
            edge_marker.colors.push_back(edge_color);
        }

        markers.markers.push_back(edge_marker);

        markers_pub.publish(markers);
    }

    double angleBetweenVectors(const Eigen::Vector3d& ab, const Eigen::Vector3d& ac) {
        double cosTheta = ab.dot(ac) / (ab.norm() * ac.norm());
        double theta = std::acos(cosTheta);
        double degrees = theta * 180 / M_PI;

        if(std::abs(degrees) > std::abs(degrees - 180)) {
            degrees = abs(degrees - 180);
        }
        return degrees;
    }

    struct VehicleObservations {
        int vehicle_id;
        std::vector<DynObservation::Ptr> observations;
    };

  private:
    // ROS
    ros::NodeHandle ms_nh;
    ros::NodeHandle ms_mt_nh;
    ros::NodeHandle ms_private_nh;
    ros::WallTimer ms_optimization_timer;
    ros::WallTimer ms_map_publish_timer;
    ros::WallTimer keyframe_metric_pub_timer;

    std::unique_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_sub;
    std::unique_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync;
    ros::Subscriber ms_command_sub;
    ros::Subscriber keyframe_msg_sub;
    ros::Subscriber tracked_object_sub;

    double height_offset_keyframes;

    bool loop_detection;
    bool remove_dyn_pts;

    bool large_loop_detected;
    bool setInitialPosition[10];
    Eigen::Isometry3d initial_pose[10];
    std::deque<int> registeredAgents;

    ros::Publisher markers_pub;
    ros::Publisher graph_changed_pub;

    std::mutex trans_odom2map_mutex;
    std::vector<Eigen::Matrix4f> trans_odom2map;
    ros::Publisher odom2map_pub;

    std::string loop_closure_edge_robust_kernel;
    double loop_closure_edge_robust_kernel_size;
    std::set<int> agent_maps_merged;  // agent ids (0,1,..) of agents that have been merged into the first agents map

    std::string map_server_topic;
    std::string map_frame_id;

    std::string agent_keyframe_topic;
    std::string dynamic_observation_topic;

    ros::Publisher map_points_pub;
    ros::Publisher optimized_keyframes_pub;

    tf::TransformBroadcaster tf_broadcaster;
    tf::TransformListener tf_listener;

    ros::ServiceServer dump_service_server;
    ros::ServiceServer save_map_service_server;

    // for map cloud generation
    std::atomic_bool graph_updated;
    double map_cloud_resolution;
    bool map_cloud_time_shading;

    std::mutex keyframes_snapshot_mutex;
    std::vector<hdl_graph_slam::KeyFrameSnapshot::Ptr> keyframes_snapshot;
    std::unique_ptr<hdl_graph_slam::MapCloudGenerator> map_cloud_generator;
    std::map<int, int> latest_dyn_obs;  // map agent_id -> latest keyframe_id where dyn. obs were removed

    // graph slam
    // all the below members must be accessed after locking main_thread_mutex
    std::mutex main_thread_mutex;

    std::deque<hdl_graph_slam::KeyFrame::Ptr> new_keyframes[10];
    std::deque<hdl_graph_slam::Loop> detected_loop_closures;
    std::deque<hdl_graph_slam::KeyFrame::Ptr> keyframe_queue[10];

    std::vector<hdl_graph_slam::KeyFrame::Ptr> keyframes[10];
    std::deque<DynObservation::Ptr> dynamic_observations_q;

    std::unique_ptr<hdl_graph_slam::GraphSLAM> graph_slam;
    std::unique_ptr<hdl_graph_slam::LoopDetector> loop_detector;

    std::unique_ptr<hdl_graph_slam::InformationMatrixCalculator> inf_calclator;
};

}  // namespace hdl_graph_slam

PLUGINLIB_EXPORT_CLASS(hdl_graph_slam::MapServerNodelet, nodelet::Nodelet)
