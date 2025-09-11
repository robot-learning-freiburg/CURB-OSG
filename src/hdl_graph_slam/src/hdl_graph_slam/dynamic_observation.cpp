// SPDX-License-Identifier: BSD-2-Clause

#include <hdl_graph_slam/dynamic_observation.hpp>

#include <boost/filesystem.hpp>
#include <pcl/impl/point_types.hpp>
#include <eigen_conversions/eigen_msg.h>
#include <pcl/io/pcd_io.h>
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
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>

namespace hdl_graph_slam {

DynObservation::DynObservation(const curb_projection::TrackedObjectObs &msg) {
    // read object points
    pcl::MsgFieldMap field_map;
    pcl::createMapping<pcl::PointXYZ>(msg.cloud.fields, field_map);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_from_msg(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(msg.cloud, *cloud_from_msg);
    cloud = cloud_from_msg->makeShared();

    keyframe_id = msg.keyframe_id;
    stamp = msg.header.stamp;
    instance_id = msg.instance_id;
    observing_agent_id = msg.observing_agent_id;
    centroid = msg.centroid.point;
}

DynObservation::~DynObservation() {}

bool DynObservation::getBoxFilter(pcl::CropBox<DynObservation::PointT> &output_filter) {
    if(cloud->size() == 0) {
        ROS_WARN_THROTTLE(1.0, "mapserver dynpt: zero points in cloud");
        return false;
    }

    if(cloud->points.size() == 1) {
        // only one point -> make 2x2x2m box
        Eigen::Vector4f box_point_a(-1.0, -1.0, -1.0, 1.0);
        Eigen::Vector4f box_point_b(+1.0, +1.0, +1.0, 1.0);
        output_filter.setMin(box_point_a);
        output_filter.setMax(box_point_b);
        Eigen::Vector3d centroid_;
        tf::pointMsgToEigen(centroid, centroid_);
        output_filter.setTranslation(centroid_.cast<float>());
    } else {
        // enough points to build bbox
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);
        output_filter.setMin(min_pt.getArray4fMap());
        output_filter.setMax(max_pt.getArray4fMap());
    }

    // negative makes it so the filter _excludes_ stuff in the box
    output_filter.setNegative(true);

    return true;
}

}  // namespace hdl_graph_slam
