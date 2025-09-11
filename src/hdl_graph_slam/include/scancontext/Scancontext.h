#pragma once

#include <ctime>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <iostream>

#include <Eigen/Dense>

// #include <opencv2/opencv.hpp>
// #include <opencv2/core/eigen.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <cv_bridge/cv_bridge.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include "hdl_graph_slam/keyframe.hpp"
#include "scancontext/nanoflann.hpp"
#include "scancontext/KDTreeVectorOfVectorsAdaptor.h"
#include "scancontext/tic_toc.h"

using namespace Eigen;
using namespace nanoflann;

using std::cout;
using std::endl;
using std::make_pair;

using std::atan2;
using std::cos;
using std::sin;

using SCPointType = pcl::PointXYZINormal;  // CURB-SG point type (TODO: could change to XYZ only)
using KeyMat = std::vector<std::vector<float> >;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor<KeyMat, float>;

// namespace SC2
// {

void coreImportTest(void);

// sc param-independent helper functions
float xy2theta(const float &_x, const float &_y);
MatrixXd circshift(MatrixXd &_mat, int _num_shift);
std::vector<float> eig2stdvec(MatrixXd _eigmat);

class SCManager {
  public:
    SCManager() = default;  // reserving data space (of std::vector) could be considered. but the descriptor is lightweight so don't care.

    Eigen::MatrixXd makeScancontext(const pcl::PointCloud<SCPointType> &_scan_down);
    Eigen::MatrixXd makeRingkeyFromScancontext(Eigen::MatrixXd &_desc);
    Eigen::MatrixXd makeSectorkeyFromScancontext(Eigen::MatrixXd &_desc);

    int fastAlignUsingVkey(MatrixXd &_vkey1, MatrixXd &_vkey2);
    double distDirectSC(MatrixXd &_sc1, MatrixXd &_sc2);  // "d" (eq 5) in the original paper (IROS 18)
    std::pair<double, int> distanceBtnScanContext(MatrixXd &_sc1, MatrixXd &_sc2);  // "D" (eq 6) in the original paper (IROS 18)

    int find_id_mapping(int kf_id);

    // User-side API
    void makeAndSaveScancontextAndKeys(const hdl_graph_slam::KeyFrame::Ptr keyframe);
    std::tuple<hdl_graph_slam::KeyFrame::Ptr, float, float> detectLoopClosureID(const hdl_graph_slam::KeyFrame::Ptr current_kf);

    // for ltslam
    // User-side API for multi-session
    void saveScancontextAndKeys(Eigen::MatrixXd _scd);

    const Eigen::MatrixXd &getConstRefRecentSCD(void);

  public:
    // hyper parameters ()
    const double LIDAR_HEIGHT = 0.0;  // lidar height : add this for simply directly using lidar scan in the lidar local coord (not robot base coord) / if you use robot-coord-transformed lidar scans, just set this as 0.

    const int PC_NUM_RING = 20;    // 20 in the original paper (IROS 18)
    const int PC_NUM_SECTOR = 60;  // 60 in the original paper (IROS 18)
    double PC_MAX_RADIUS = 80.0;   // 80 meter max in the original paper (IROS 18)
    const double PC_UNIT_SECTORANGLE = 360.0 / double(PC_NUM_SECTOR);
    const double PC_UNIT_RINGGAP = PC_MAX_RADIUS / double(PC_NUM_RING);

    // tree
    const int NUM_EXCLUDE_RECENT = 10;  // simply just keyframe gap (related with loopClosureFrequency in yaml), but node position distance-based exclusion is ok.
    const int NUM_CANDIDATES_FROM_TREE = 10;  // 10 is enough. (refer the IROS 18 paper)

    // loop thres
    const double SEARCH_RATIO = 0.1;  // for fast comparison, no Brute-force, but search 10 % is okay. // not was in the original conf paper, but improved ver.
    // const double SC_DIST_THRES = 0.13; // empirically 0.1-0.2 is fine (rare false-alarms) for 20x60 polar context (but for 0.15 <, DCS or ICP fit score check (e.g., in LeGO-LOAM) should be required for robustness)

    double SC_DIST_THRES = 0.2;  // 0.4-0.6 is good choice for using with robust kernel (e.g., Cauchy, DCS) + icp fitness threshold / if not, recommend 0.1-0.15
    // const double SC_DIST_THRES = 0.7; // 0.4-0.6 is good choice for using with robust kernel (e.g., Cauchy, DCS) + icp fitness threshold / if not, recommend 0.1-0.15

    // config
    const int TREE_MAKING_PERIOD_ = 30;  // i.e., remaking tree frequency, to avoid non-mandatory every remaking, to save time cost / in the LeGO-LOAM integration, it is synchronized with the loop detection callback (which is 1Hz) so it means the tree is updated evrey 10 sec. But you can use the smaller value because it is enough fast ~ 5-50ms wrt N.
    int tree_making_period_conter = 0;

    // setter
    void setSCdistThres(double _new_thres);
    void setMaximumRadius(double _max_r);

    // data
    std::vector<double> polarcontexts_timestamp_;  // optional.
    std::vector<Eigen::MatrixXd> polarcontexts_;
    std::vector<Eigen::MatrixXd> polarcontext_invkeys_;
    std::vector<Eigen::MatrixXd> polarcontext_vkeys_;

    std::vector<hdl_graph_slam::KeyFrame::Ptr> keyframes;  // map scManager ids to HDL keyframes for multi-agent setup

    KeyMat polarcontext_invkeys_mat_;
    KeyMat polarcontext_invkeys_to_search_;
    std::unique_ptr<InvKeyTree> polarcontext_tree_;

    bool is_tree_batch_made = false;
    std::unique_ptr<InvKeyTree> polarcontext_tree_batch_;

};  // SCManager

// } // namespace SC2
