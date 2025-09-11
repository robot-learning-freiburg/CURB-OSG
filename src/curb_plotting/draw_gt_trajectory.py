import csv
from os import path

import numpy as np
from geometry_msgs.msg import Quaternion as Quaternion_msg
from tf.transformations import quaternion_from_euler
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

DATASET = "/workspaces/collaborative-scene-graphs/data/radar-robotcar/2019-01-11-13-24-51-radar-oxford-10k"

# "/workspaces/collaborative-scene-graphs/data/radar-robotcar/2019-01-14-14-15-12-radar-oxford-10k",
# "/workspaces/collaborative-scene-graphs/data/radar-robotcar/2019-01-15-13-06-37-radar-oxford-10k"
CSV = "gps/ins.csv"

# end_timestamps = ["0001547214125480917", "0001547476127067551", "0001547558500508021"]
# end_timestamps = ["9999999999999999999", "9999999999999999999", "9999999999999999999"]
END_STAMP = "9999999999999999999"


def line2loc(line):
    loc = np.array(
        [
            float(line["northing"]) - 5735848.36,
            float(line["easting"]) - 620057.41,
            float(line["down"]) + 114.7,
            float(line["yaw"]),
        ]
    )
    return loc


def read_gt_track(dataset):
    track = []
    timestamps = []

    with open(path.join(dataset, CSV)) as f:
        reader = csv.DictReader(f)
        lastloc = None
        for line in reader:
            if int(line["timestamp"]) > int(END_STAMP):
                break

            if lastloc is None:
                lastloc = line2loc(line)
                track.append(lastloc)
                timestamps.append(line["timestamp"])
                continue

            loc = line2loc(line)
            dist = loc - lastloc
            if np.linalg.norm(dist) > 5:
                track.append(loc)
                timestamps.append(line["timestamp"])
                lastloc = line2loc(line)

    return track, timestamps


if __name__ == "__main__":
    rospy.init_node("gt_trajectory_publisher", anonymous=True)
    marker_pub = rospy.Publisher("gps_viz2", MarkerArray, latch=True, queue_size=1)

    track = read_gt_track(DATASET)
    ma = MarkerArray()
    ma.markers = []
    stamp = rospy.Time.now()

    id = 0
    m = Marker()
    m.header.frame_id = "world"
    m.header.stamp = stamp
    m.id = id
    id += 1
    m.type = m.LINE_STRIP
    m.scale.x = 5.0  # Thickness of the line
    m.color.a = 1.0
    m.color.r = 0.0
    m.color.g = 0.0
    m.color.b = 0.0

    m.points = []
    for p in track:
        pt = Point()
        pt.x = p[0]
        pt.y = p[1]
        pt.z = 0.0
        m.points.append(pt)

    ma.markers.append(m)

    ## uncomment to show timestamps
    #
    # mt = Marker()
    # mt.header.frame_id = "world"
    # mt.header.stamp = stamp
    # mt.id = id
    # id += 1
    # mt.type = m.TEXT_VIEW_FACING
    # mt.pose.position.x = p[0] + 2.0
    # mt.pose.position.y = p[1]
    # mt.pose.position.z = p[2]
    # mt.scale.z = 1.0
    # mt.color.a = 1.0
    # mt.color.r = 0.0 + (i / 2)
    # mt.color.g = 0.0
    # mt.color.b = 1.0 - (i / 2)
    # mt.text = f"{i}: {timestamps[i][j]}"
    # mt.lifetime = rospy.Duration(100000)

    # ma.markers.append(mt)

    len(ma.markers)

    marker_pub.publish(ma)

    rospy.spin()
