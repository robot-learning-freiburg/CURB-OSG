// SPDX-License-Identifier: BSD-2-Clause

#ifndef LOOP_DETECTOR_HPP
#define LOOP_DETECTOR_HPP

#include <boost/format.hpp>
#include <fstream>
#include <hdl_graph_slam/graph_slam.hpp>
#include <hdl_graph_slam/keyframe.hpp>
#include <hdl_graph_slam/registrations.hpp>

#include <Eigen/src/Geometry/Transform.h>
#include <g2o/types/slam3d/vertex_se3.h>

#include <hdl_graph_slam/custom_point_types.hpp>
#include <thread>
#include "hdl_graph_slam/safe_queue.hpp"
#include "scancontext/Scancontext.h"

#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/passthrough.h>
#include <ros/console.h>
#include <ros/duration.h>
#include <ros/publisher.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <visualization_msgs/MarkerArray.h>

namespace hdl_graph_slam {

enum LoopSource { DIST, SCANCONTEXT };

// this describes a loop closure or a loop closure candidate
struct Loop {
  public:
    // this macro is required because the struct contains an Eigen matrix
    // see https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Ptr = std::shared_ptr<Loop>;

    // constructor with pose
    Loop(const KeyFrame::Ptr &src_kf, const KeyFrame::Ptr &target_kf, const LoopSource source)
        : src_kf(src_kf), target_kf(target_kf), source(source) {}

    KeyFrame::Ptr src_kf;
    KeyFrame::Ptr target_kf;
    Eigen::Matrix4f relative_pose;
    Eigen::Matrix4f init_guess;

    // whether relative pose is set or not
    bool pose_valid;

    // where this loop closure (candidate) came from
    enum LoopSource source;

    // scan context stuff - only valid if source is SCANCONTEXT
    double sc_yaw, sc_dist;

    void set_relative_pose(Eigen::Matrix4f &pose) {
        relative_pose = pose;
        pose_valid = true;
    }

    void write_debug_output(std::string target_dir) {
        // verify that the output directory exists
        boost::filesystem::path p(target_dir);
        if(!boost::filesystem::exists(p)) {
            ROS_ERROR("Output directory does not exist: %s", target_dir.c_str());
            return;
        }

        // write the loop closure to a file
        std::ofstream init_guess_f(target_dir + "/init_guess.txt");
        init_guess_f << init_guess << std::endl;
        init_guess_f.close();

        std::ofstream relative_pose_f(target_dir + "/relative_pose.txt");
        relative_pose_f << relative_pose << std::endl;
        relative_pose_f.close();

        std::ofstream metadata_f(target_dir + "/metadata.txt");
        metadata_f << "source: " << src_kf->keyframe_id << std::endl;
        metadata_f << "target: " << target_kf->keyframe_id << std::endl;
        switch (source) {
            case LoopSource::DIST:
                metadata_f << "source: DIST" << std::endl;
                break;
            case LoopSource::SCANCONTEXT:
                metadata_f << "source: SCANCONTEXT" << std::endl;
                break;
        }
        metadata_f.close();

        // serialize keyframe point clouds
        pcl::io::savePCDFileBinary(target_dir + "/src.pcd", *src_kf->cloud);
        pcl::io::savePCDFileBinary(target_dir + "/target.pcd", *target_kf->cloud);

        ROS_INFO("Wrote loop closure debug output to %s", target_dir.c_str());
    }
};

/**
 * @brief this class finds loops by scan matching and adds them to the pose graph
 */
class LoopDetector {
  public:
    typedef pcl::PointXYZINormal PointT;

    /**
     * @brief constructor
     * @param pnh
     */
    LoopDetector(ros::NodeHandle &pnh) {
        distance_thresh = pnh.param<double>("distance_thresh", 5.0);
        accum_distance_thresh = pnh.param<double>("accum_distance_thresh", 8.0);
        distance_from_last_edge_thresh = pnh.param<double>("min_edge_interval", 5.0);

        debug_output = pnh.param<std::string>("loop_debug_output", "");

        fitness_score_max_range = pnh.param<double>("fitness_score_max_range", std::numeric_limits<double>::max());
        fitness_score_thresh = pnh.param<double>("fitness_score_thresh", 0.5);

        // scan context stuff:
        scan_context_matching = pnh.param<bool>("scan_context_matching", true);
        sc_descriptor_distance_thres = pnh.param<double>("sc_descriptor_distance", 0.7);  // threshold for sc descriptor distance metric
        sc_max_radius = pnh.param<double>("sc_max_radius", 80);  // scans are limited to this radius for sc
        distance_thresh_sc = pnh.param<double>("distance_thresh_sc", 100);  // only validate sc candidates below this 2d distance threshold

        sc_manager.setMaximumRadius(sc_max_radius);
        sc_manager.setSCdistThres(sc_descriptor_distance_thres);

        registration = select_registration_method(pnh);

        for(int i = 0; i < 10; i++) {
            last_edge_accum_distance[i] = 0.0;
        }

        loop_checker_thread = std::thread(&LoopDetector::validate_candidates, this);

        // visualization
        loop_marker_pub = pnh.advertise<visualization_msgs::MarkerArray>("/map_server/loop_closure_markers", 16);
    }

    // Destructor
    ~LoopDetector() {
        candidates_to_check.wakeup();
        detected_loops_output.wakeup();
        loop_checker_thread.join();
    }

    /**
     * @brief detect loop candidates by distance and add them to the queue
     * @param keyframes       keyframes
     * @param new_keyframes   newly registered keyframes
     * @param graph_slam      pose graph
     */
    void check_new_keyframes(const std::vector<KeyFrame::Ptr> &keyframes, const KeyFrame::Ptr &new_keyframe, hdl_graph_slam::GraphSLAM &graph_slam, int agent_no) {
        find_candidates_by_dist(keyframes, new_keyframe, agent_no);

        // compute scan context, store scan context in SC Manager, detect loop closures
        if(scan_context_matching) {
            find_candidates_by_sc(new_keyframe);
        } else {
            ROS_WARN_THROTTLE(20.0, "Scan context off");
        }
    }

    // get all detected loops, emptying the output queue
    std::vector<Loop> get_detected_loops() {
        std::vector<Loop> output;
        while(detected_loops_output.size() > 0) {
            output.push_back(detected_loops_output.dequeue());
        }
        return output;
    }

    double get_distance_thresh() {
        return distance_thresh;
    }

  private:
    void write_debug_output(Loop &loop) {
        if (debug_output == "") return;

        // verify that the output directory exists
        boost::filesystem::path p(debug_output);
        if (!boost::filesystem::exists(p)) {
            ROS_ERROR("Output directory does not exist: %s", debug_output.c_str());
            return;
        }


        // target dir format: debug_output/src_kf-target_kf/
        std::string target_dir = (boost::format("%s/%d-%d") % debug_output % loop.src_kf->keyframe_id % loop.target_kf->keyframe_id).str();
        // make sure target dir does not exist
        if(boost::filesystem::exists(target_dir)) {
            ROS_ERROR("Target directory already exists: %s", target_dir.c_str());
            return;
        }
        boost::filesystem::create_directories(target_dir);

        loop.write_debug_output(target_dir);
    }

    void visualize_loop(Loop &loop, bool is_candidate) {
        visualization_msgs::MarkerArray ma;

        auto a = loop.src_kf->node->estimate().translation();
        auto b = loop.target_kf->node->estimate().translation();

        geometry_msgs::Point p_start;
        p_start.x = a.x();
        p_start.y = a.y();
        p_start.z = a.z();

        geometry_msgs::Point p_end;
        p_end.x = b.x();
        p_end.y = b.y();
        p_end.z = b.z();

        visualization_msgs::Marker m;
        m.header.frame_id = "world";
        m.header.stamp = ros::Time::now();
        m.id = loop.target_kf->keyframe_id * (loop.source + 1);
        m.ns = "loop_closures";
        m.type = visualization_msgs::Marker::ARROW;
        m.action = visualization_msgs::Marker::ADD;
        m.lifetime = ros::Duration(20.0);
        m.pose.orientation.w = 1.0;
        m.points.push_back(p_start);
        m.points.push_back(p_end);

        m.scale.x = 1.0;
        m.scale.y = 3.0;
        m.scale.z = 5.0;

        m.color.a = 0.6;  // Don't forget to set the alpha!

        if(is_candidate && loop.source == LoopSource::SCANCONTEXT) {
            // sc candidates: green-red, depending on score
            m.color.r = (loop.sc_dist / sc_descriptor_distance_thres);
            m.color.g = 1.0 - (loop.sc_dist / sc_descriptor_distance_thres);
            m.color.b = 0.06;
        } else if(is_candidate && loop.source == LoopSource::DIST) {
            // distance candidates: purplish
            m.color.r = 0.8;
            m.color.g = 0.0;
            m.color.b = 0.8;
        } else if(!is_candidate && loop.source == LoopSource::DIST) {
            // found loop & dist -> blue-green
            m.lifetime = ros::Duration(40.0);
            m.color.r = 0.0;
            m.color.g = 1.0;
            m.color.b = 1.0;
        } else if(!is_candidate && loop.source == LoopSource::SCANCONTEXT) {
            // found loop & SC -> pink
            m.lifetime = ros::Duration(40.0);
            m.color.r = 0.8;
            m.color.g = 0.15;
            m.color.b = 0.88;
        }

        ma.markers.push_back(m);

        loop_marker_pub.publish(ma);
    }

    /**
     * @brief find loop candidates by estimated distance. add them to the queue candidates_to_check
     * @param all_keyframes  candidate keyframes of loop start
     * @param new_keyframe   loop end keyframe
     */
    void find_candidates_by_dist(const std::vector<KeyFrame::Ptr> &all_keyframes, const KeyFrame::Ptr &new_keyframe, int agent_no) {
        for(const auto &k : all_keyframes) {
            if(k->keyframe_id / 1000000 == agent_no) {
                // keyframes are from the same agent, check min loop distance thresh
                if(new_keyframe->accum_distance - k->accum_distance < accum_distance_thresh) {
                    continue;
                }
            }

            const auto &pos1 = k->node->estimate().translation();
            const auto &pos2 = new_keyframe->node->estimate().translation();

            // check distance radius
            double dist = (pos1.head(2) - pos2.head(2)).norm();
            // double dist3d = (pos1.head(3) - pos2.head(3)).norm();
            if(dist > distance_thresh) {
                continue;
            }

            // add to queue of candidates
            Loop loop_candidate(new_keyframe, k, LoopSource::DIST);
            visualize_loop(loop_candidate, true);
            candidates_to_check.enqueue(loop_candidate);
        }
    }

    /**
     * @brief find loop candidates by scan context. add them to the queue candidates_to_check
     * @param all_keyframes  candidate keyframes of loop start
     * @param new_keyframe   loop end keyframe
     */
    void find_candidates_by_sc(const KeyFrame::Ptr &new_keyframe) {
        sc_manager.makeAndSaveScancontextAndKeys(new_keyframe);
        // detect_result is a tuple (target_frame, distance, yaw)
        auto detect_result = sc_manager.detectLoopClosureID(new_keyframe);

        KeyFrame::Ptr target_kf = std::get<0>(detect_result);

        // guard - if sc_manager returned nullptr, no loop was found
        if(!target_kf) return;

        if(target_kf->keyframe_id == new_keyframe->keyframe_id) {
            ROS_ERROR("Scan context matched keyframe with self (should not happen)");
            return;
        }

        if(target_kf->keyframe_id / 1000000 == new_keyframe->keyframe_id / 1000000) {
            // keyframes are from the same agent, check 1km loop distance thresh (todo: make parameter)
            double dist = new_keyframe->accum_distance - target_kf->accum_distance;
            if(dist < 1000.0) return;
        }

        float sc_dist = std::get<1>(detect_result);
        // yaw: how much to rotate target until it matches src
        float yaw = std::get<2>(detect_result);

        // check if the proposed loop closure is too long
        auto target_tf = target_kf->node->estimate();
        auto src_tf = new_keyframe->node->estimate();
        auto rel_tf = target_tf.inverse() * src_tf;
        Eigen::Vector2f transl(rel_tf.translation().x(), rel_tf.translation().y());
        if(transl.norm() > distance_thresh_sc) {
            return;
        }
        ROS_INFO("[SC] Found loop between %d and %d (dist=%f, yaw=%f)", new_keyframe->keyframe_id, target_kf->keyframe_id, sc_dist, yaw);

        // store new loop candidate - without checking sc dist, as it is thresholded in SC Manager
        Loop loop_candidate(new_keyframe, target_kf, LoopSource::SCANCONTEXT);
        loop_candidate.sc_dist = sc_dist;
        loop_candidate.sc_yaw = yaw;

        visualize_loop(loop_candidate, true);

        candidates_to_check.enqueue(loop_candidate);
    }

    /**
     * @brief wait for new candidates in queue candidates_to_check and check
     * them. If they are valid, add them to the queue detected_loops.
     * this function is supposed to run in its own thread.
     */
    void validate_candidates() {
        while(ros::ok()) {
            Loop loop_candidate = candidates_to_check.dequeue();
            if(check_loop(loop_candidate)) {
                detected_loops_output.enqueue(loop_candidate);
                visualize_loop(loop_candidate, false);
            }
        }
    }

    /**
     * @brief To validate a loop candidate this function applies a scan matching
     * between keyframes consisting the loop. If they are matched well, the loop
     * is added to the queue of detected loops
     * @param  candidate     candidate loop to check. If the loop is valid, will set the relative_pose
     * @return true if loop is valid, false otherwise
     */
    bool check_loop(Loop &candidate) {
        if(candidate.src_kf->accum_distance - last_edge_accum_distance[candidate.src_kf->keyframe_id / 1000000] < distance_from_last_edge_thresh) {
            // too close to the last registered loop edge
            ROS_DEBUG("loop %d->%d check return: distance from last edge thresh", candidate.src_kf->keyframe_id,
                      candidate.target_kf->keyframe_id);
            return false;
        }

        // check if there is already a loop towards this target keyframe
        int target_kf_id = candidate.target_kf->keyframe_id;
        bool kf_is_already_detected = std::find_if(all_loop_targets.begin(), all_loop_targets.end(), [&](int existing_target_id) {
                                          return (target_kf_id == existing_target_id);
                                      }) != all_loop_targets.end();
        if(kf_is_already_detected) {
            ROS_DEBUG("loop %d->%d check return: already a loop toward this target", candidate.src_kf->keyframe_id,
                      candidate.target_kf->keyframe_id);
            return false;
        }

        registration->setInputTarget(candidate.target_kf->cloud);

        // get initial guess
        candidate.init_guess.setIdentity();

        std::string source;
        if(candidate.source == LoopSource::DIST) {
            source = "DST";
            Eigen::Isometry3d source_pose = candidate.src_kf->node->estimate();
            source_pose.linear() = Eigen::Quaterniond(source_pose.linear()).normalized().toRotationMatrix();
            Eigen::Isometry3d target_pose = candidate.target_kf->node->estimate();
            target_pose.linear() = Eigen::Quaterniond(target_pose.linear()).normalized().toRotationMatrix();
            // guess is the pose of the source in the frame of target
            candidate.init_guess = (target_pose.inverse() * source_pose).matrix().cast<float>();
            // prior: same z-level
            candidate.init_guess(2, 3) = 0.0;
        } else {
            source = "SC";
            // loop comes from scan context - only set yaw and assume translation is 0
            tf::Quaternion q = tf::createQuaternionFromYaw(candidate.sc_yaw);
            tf::Transform t(q);
            Eigen::Isometry3d source_in_target;
            tf::transformTFToEigen(t, source_in_target);
            // inverse because SC yaw is to rotate target - we need to rotate the source cloud the opposite way
            candidate.init_guess = source_in_target.matrix().inverse().cast<float>();
        }
        ROS_INFO("[%s] Checking loop between %d and %d ", source.c_str(), candidate.src_kf->keyframe_id,
                 candidate.target_kf->keyframe_id);
        ROS_INFO_STREAM("initial guess: \n" << candidate.init_guess << std::endl);

        // scan matching
        pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
        registration->setInputSource(candidate.src_kf->cloud);
        registration->align(*aligned, candidate.init_guess);
        double score = registration->getFitnessScore(fitness_score_max_range);

        if(!registration->hasConverged()) {
            ROS_INFO("Not converged");
            return false;
        }

        double current_score_thresh = fitness_score_thresh;
        if(candidate.source == LoopSource::SCANCONTEXT) {
            // dynamic threshold depending on scan context certainty: less certain (higher sc dist) -> lower threshold
            current_score_thresh -= (candidate.sc_dist / sc_descriptor_distance_thres) * fitness_score_thresh;
            current_score_thresh += fitness_score_thresh / 6;
        }
        // "score" is not really a score but a distance measure -> need to use greater-than
        if(score > current_score_thresh) {
            ROS_INFO("Score too high: %f > %f", score, current_score_thresh);
            return false;
        }

        Eigen::Matrix4f relative_pose = registration->getFinalTransformation();
        ROS_INFO("Loop confirmed with distance score %f / %f ", score, current_score_thresh);
        ROS_INFO_STREAM("relpose: \n" << relative_pose << std::endl);

        Eigen::Affine3f t_eig(relative_pose);
        tf::Transform t_tf;
        tf::transformEigenToTF(t_eig.cast<double>(), t_tf);
        tf::Quaternion q = t_tf.getRotation();
        double rot_angle = q.getAngle();
        ROS_INFO("relpose total angle: %f", rot_angle);

        last_edge_accum_distance[candidate.src_kf->keyframe_id / 1000000] = candidate.src_kf->accum_distance;

        candidate.set_relative_pose(relative_pose);

        write_debug_output(candidate);

        return true;
    }

  private:
    double distance_thresh;  // estimated distance between keyframes consisting a loop must be less than this distance
    double accum_distance_thresh;           // traveled distance between ...
    double distance_from_last_edge_thresh;  // a new loop edge must far from the last one at least this distance
    std::string debug_output;  // path to save debug output, will be written if != ""

    double fitness_score_max_range;  // maximum allowable distance between corresponding points
    double fitness_score_thresh;     // threshold for scan matching

    double last_edge_accum_distance[10];

    std::vector<int> all_loop_targets;  // history of all detected loop target kf ids to prevent double loops

    ros::Publisher loop_marker_pub;

    // scan context stuff
    bool scan_context_matching;
    SCManager sc_manager;
    float sc_descriptor_distance_thres;
    float distance_thresh_sc;
    float sc_max_radius;

    pcl::Registration<PointT, PointT>::Ptr registration;

    std::thread loop_checker_thread;

    SafeQueue<Loop> candidates_to_check;    // fifo for candidates that will be checked in a seperate thread
    SafeQueue<Loop> detected_loops_output;  // fifo for confirmed loops to be added to graph
};

}  // namespace hdl_graph_slam

#endif  // LOOP_DETECTOR_HPP
