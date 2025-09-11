from bisect import bisect_left
import os
from typing import List, Optional
import cv2
import numpy as np
from numpy.typing import NDArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


def plot_intersections(
    intersections,
    pub,
    ns="gt_intsersections",
    size=12.0,
    color=(1.0, 0.0, 0.0, 1.0),
    z_offset=0.0,
):
    ma = MarkerArray()
    ma.markers = []
    for i, intersection in enumerate(intersections):
        m = Marker()
        m.header.frame_id = "world"
        m.ns = ns
        m.id = i
        m.type = m.SPHERE
        m.pose.position.x = intersection[0]
        m.pose.position.y = intersection[1]
        m.pose.position.z = z_offset
        m.pose.orientation.w = 1.0
        m.scale.x = size
        m.scale.y = size
        m.scale.z = size
        m.color.r = color[0]
        m.color.g = color[1]
        m.color.b = color[2]
        m.color.a = color[3]
        ma.markers.append(m)
    return ma


def plot_ways(ways, pub, size=1.6, color=(0.3, 0.3, 0.3, 1.0), z_offset=0.0):
    ma = MarkerArray()
    ma.markers = []
    for i, way in enumerate(ways):
        m = Marker()
        m.header.frame_id = "world"
        m.ns = "gt_ways"
        m.id = i + 1000
        m.type = m.LINE_STRIP
        m.scale.x = size
        m.pose.orientation.w = 1.0
        m.pose.position.z = z_offset
        m.color.r = color[0]
        m.color.g = color[1]
        m.color.b = color[2]
        m.color.a = color[3]
        m.points = []
        for node in way.values():
            m.points.append(Point(node[0], node[1], 0.0))
        ma.markers.append(m)
    return ma


class StampedImg:
    def __init__(self, timestamp: float, filename: str):
        self.timestamp = timestamp
        self.filename = filename


def read_stamps(dataset_path: str) -> List[StampedImg]:
    files = os.listdir(dataset_path)
    files = [f for f in files if f.endswith(".png")]
    files.sort()

    timestamps_str = [f.split(".")[0] for f in files]
    timestamps = [
        StampedImg(int(stamp) / 1e6, os.path.join(dataset_path, files[i]))
        for i, stamp in enumerate(timestamps_str)
    ]
    return timestamps


def find_closest_sample(target_stamp: float, imgs: List[StampedImg]) -> str:
    """
    Returns file name of image file closest to the given timestamp.
    Timestamp must be in seconds and imgs list must be sorted ascending by stamp.
    """
    all_stamps = [img.timestamp for img in imgs]
    pos = bisect_left(all_stamps, target_stamp)
    if pos == 0:
        return imgs[0].filename
    if pos == len(all_stamps):
        return imgs[pos - 1].filename
    before = all_stamps[pos - 1]
    after = all_stamps[pos]
    if (after - target_stamp) < (target_stamp - before):
        return imgs[pos].filename
    else:
        return imgs[pos - 1].filename


def draw_landmark(
    img: NDArray, center_uv: NDArray, pts_uv: NDArray, depths: NDArray, rgb: List[float], occl_ratio: Optional[float] = None
) -> NDArray:
    """
    Draws a landmark on an image.
    """
    pts_uv = pts_uv.astype(int).T

    overlay = np.zeros(img.shape, dtype=np.uint8)

    color = (
        (np.array(rgb) * 255).astype(int).tolist()
    )  # Convert color to a list of integers
    color = color[::-1]  # Convert RGB to BGR
    # draw points
    for i, pt in enumerate(pts_uv):
        d = depths[i]
        size = int(30 * 1 / d)

        cv2.circle(overlay, tuple(pt), size, color, -1)

    # draw center
    center_uv = center_uv.squeeze().astype(int)
    cv2.rectangle(
        overlay,
        (center_uv[0] - 3, center_uv[1] - 3),
        (center_uv[0] + 3, center_uv[1] + 3),
        color,
        -1,
    )

    # write occl_dist and occl_ratio if available
    if occl_ratio is not None:
        cv2.putText(
            overlay,
            f"occl_ratio: {occl_ratio:.2f}",
            (center_uv[0] + 10, center_uv[1] - 40 + np.random.randint(-40, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    # draw hull around object points
    hull = cv2.convexHull(pts_uv)
    cv2.polylines(overlay, [hull], True, color, 2)

    return overlay


def draw_mask(img: NDArray, pts_uv: NDArray) -> NDArray:
    """
    Draws a binary mask of the convex hull of the pts.
    """
    pts_uv = pts_uv.astype(int).T

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # draw hull around object points
    hull = cv2.convexHull(pts_uv)
    
    # fill hull with ones
    cv2.fillPoly(mask, [hull], 1)

    return mask
