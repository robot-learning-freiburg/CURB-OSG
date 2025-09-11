from typing import Tuple
import open3d as o3d
import numpy as np
from numpy.typing import NDArray
import os
import rospy
import json
from tf.transformations import compose_matrix, euler_from_matrix, translation_from_matrix
import cv2
import matplotlib.pyplot as plt
from curb_projection import CameraModel
from scipy.interpolate import griddata
from tqdm import tqdm

import sys
sys.path.append("/workspaces/collaborative-scene-graphs/src/")
from robotcar_dataset_sdk.python.velodyne import load_velodyne_binary


MODELS_DIR = "/workspaces/collaborative-scene-graphs/src/curb_projection/intrinsics"
LIDAR_DIR = "/workspaces/collaborative-scene-graphs/data/radar-robotcar/2019-01-14-14-15-12-radar-oxford-10k/velodyne_left"

model = CameraModel(MODELS_DIR, "stereo/left")

trans_lidar2cam = [-0.592, -0.338, 0.292]
rpy_lidar2cam = [-0.005, -0.042, 3.134]
T_lidar2cam = compose_matrix(translate=trans_lidar2cam, angles=rpy_lidar2cam)
T_lidar2cam = np.linalg.inv(T_lidar2cam)

lidar_files = os.listdir(LIDAR_DIR)
lidar_files.sort()
lidar_stamps = [int(file.split('.')[0]) for file in lidar_files]

def find_closest_stamp(stamp, lidar_stamps):
    idx = np.argmin(np.abs(np.array(lidar_stamps) - stamp))
    return lidar_stamps[idx]

def show_img(img):
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def compute_occlusion(lm_mask, lm_info, depth_img):
    lm_depth = float(lm_info['depth'])

    infront = depth_img[depth_img < (lm_depth - 4.0)]
    occluded = lm_mask * (depth_img < (lm_depth - 4.0)).astype(np.uint8)
    occlusion_ratio = occluded.sum() / lm_mask.sum()
    infront_depths = depth_img[occluded == 1]
    if len(infront_depths) > 0:
        depth_diff = lm_depth - infront_depths.mean()
    else:
        depth_diff = -1
    return occlusion_ratio, depth_diff
    
def process_prediction(pred, masks_dir):
    if pred["segments_info"] == []:
        return pred

    pred_stamp = int(pred['image_id'])
    vis_file_name = os.path.join(masks_dir, pred['image_id'] + ".vis.png")
    file_name = os.path.join(masks_dir, pred['file_name'])

    vis_img = cv2.imread(vis_file_name)
    masks = cv2.imread(file_name)

    lidar_stamp = find_closest_stamp(pred_stamp, lidar_stamps)

    lidar_file = os.path.join(LIDAR_DIR, str(lidar_stamp) + ".bin")
    scan = load_velodyne_binary(lidar_file)
    # voxelgrid downsample @ 10cm
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(scan[:3].T)
    # pcd = pcd.voxel_down_sample(voxel_size=0.1)
    # scan = np.asarray(pcd.points).T
    xyz = scan[:3]
    xyz = np.dot(T_lidar2cam, np.vstack([xyz, np.ones((1, xyz.shape[1]))]))[:3]

    # project into image
    pts_uv, uv_depths, _ = model.project(xyz, masks.shape[:2])
    # map_pts_uv.shape = (N,2)
    pts_uv = pts_uv.astype(int).T

    X, Y = masks.shape[:2]
    grid_x, grid_y = np.mgrid[0:masks.shape[0], 0:masks.shape[1]]
    depth_img = griddata(pts_uv, uv_depths, (grid_y, grid_x), method='nearest')

    lidar_mask = np.zeros(masks.shape[:2], dtype=np.uint8)
    for uv, depth in zip(pts_uv, uv_depths):
        u, v = uv
        # h, w = model.get_projected_size(0.20, 0.20, depth)
        h, w = (25,25)
        # draw rectagle around point
        h2 = int(h//2)
        w2 = int(w//2)
        cv2.rectangle(lidar_mask, (u-w2, v-h2), (u+w2, v+h2), 1, -1)
    
    depth_img *= lidar_mask

    depth_vis = ((depth_img / depth_img.max()) * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    depth_vis[depth_img == 0] = 0
    max_lm_depth = max([lm['depth'] for lm in pred['segments_info']])
    depth_vis[depth_img > (max_lm_depth - 4.0)] = 0

    depth_img[depth_img == 0] = np.inf

    overlay = np.zeros(masks.shape, dtype=np.uint8)
    for lm_info in pred['segments_info']:

        lm_mask = np.zeros(masks.shape[:2], dtype=np.uint8)
        lm_mask[masks[:,:,2] == lm_info['id']] = 1

        occlusion_ratio, depth_diff = compute_occlusion(lm_mask, lm_info, depth_img)
        lm_info['occlusion_ratio'] = occlusion_ratio
        lm_info['depth_diff'] = depth_diff

        if occlusion_ratio > 0.5:
            overlay[lm_mask == 1] = np.array([0, 0, 255])
    
    occluded_vis_fname = os.path.join(masks_dir, pred['image_id'] + ".visoccl.png")
    occluded_vis = cv2.addWeighted(vis_img, 1.0, overlay, 0.5, 0)
    cv2.imwrite(occluded_vis_fname, occluded_vis)

    cv2.addWeighted(vis_img, 0.6, depth_vis, 0.4, 0, depth_vis)
    depth_vis_fname = os.path.join(masks_dir, pred['image_id'] + ".visdepth.png")
    cv2.imwrite(depth_vis_fname, depth_vis)
    
    return pred
    

def main(json_file: str):
    masks_dir = json_file.split('.json')[0]
    assert os.path.exists(json_file), f"File {json_file} does not exist"
    assert os.path.exists(masks_dir), f"Directory {masks_dir} does not exist"

    with open(json_file, 'r') as f:
        predictions = json.load(f)["annotations"]
    
    for pred in tqdm(predictions):
        pred = process_prediction(pred, masks_dir)
    
    with open(json_file, 'w') as f:
        json.dump({"annotations": predictions}, f, indent=4)

#%%
if __name__ == "__main__":
    target_dir = sys.argv[1]
    runs = os.listdir(target_dir)
    for run in runs:
        json_file = os.path.join(target_dir, run, "semseg_preds.json")
        assert os.path.exists(json_file), f"File {json_file} does not exist"
        print(f"Processing {json_file}..")
        main(json_file)








