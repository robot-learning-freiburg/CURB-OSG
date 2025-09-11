// SPDX-License-Identifier: BSD-2-Clause
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/crop_box.h>
#include <boost/optional.hpp>
#include "geometry_msgs/Point.h"
#include "std_msgs/ColorRGBA.h"
#include <hdl_graph_slam/custom_point_types.hpp>
#include <hdl_graph_slam/keyframe.hpp>
#include <curb_projection/TrackedObjectObs.h>

namespace g2o {
class VertexSE3;

class HyperGraph;

class SparseOptimizer;
}  // namespace g2o

namespace hdl_graph_slam {

/**
 * @brief Dynamic Observation
 */
struct DynObservation {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using PointT = pcl::PointXYZINormal;
    using Ptr = std::shared_ptr<DynObservation>;

    DynObservation(const curb_projection::TrackedObjectObs &msg);
    virtual ~DynObservation();

    bool getBoxFilter(pcl::CropBox<PointT> &output_filter);

    ros::Time stamp;
    int keyframe_id;
    int instance_id;
    int observing_agent_id;
    geometry_msgs::Point centroid;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud;
    hdl_graph_slam::KeyFrame::Ptr keyframe;
};

}  // namespace hdl_graph_slam
