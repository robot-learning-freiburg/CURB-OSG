import csv
from os import path

import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray

datasets = [
  "/workspaces/collaborative-scene-graphs/data/radar-robotcar/2019-01-11-13-24-51-radar-oxford-10k",
  "/workspaces/collaborative-scene-graphs/data/radar-robotcar/2019-01-14-14-15-12-radar-oxford-10k",
  "/workspaces/collaborative-scene-graphs/data/radar-robotcar/2019-01-15-13-06-37-radar-oxford-10k"
]
csv_file = "gps/ins.csv"

#end_timestamps = ["0001547214125480917", "0001547476127067551", "0001547558500508021"]
end_timestamps = ["9999999999999999999", "9999999999999999999", "9999999999999999999"]

tracks = [[], [], []]
timestamps = [[], [], []]

def line2loc(line):
  loc = np.array([float(line["northing"]) - 5735848.36,
                  float(line["easting"]) - 620057.41,
                  float(line["down"]) + 114.7,
                  float(line["yaw"])
                  ])
  return loc


for i,d in enumerate(datasets):
  cumdist = 0.0
  f = open(path.join(d, csv_file))
  reader = csv.DictReader(f)
  lastloc = None
  for line in reader:
    if (int(line["timestamp"]) > int(end_timestamps[i])):
      break

    if lastloc is None:
      lastloc = line2loc(line)
      tracks[i].append(lastloc)
      timestamps[i].append(line["timestamp"])
      continue

    loc = line2loc(line)
    dist = loc - lastloc
    cumdist += np.linalg.norm(dist)
    if np.linalg.norm(dist) > 5:
      tracks[i].append(loc)
      timestamps[i].append(line["timestamp"])
      lastloc = line2loc(line)

  print(f"cumulative distance {i}: {cumdist}")

  f.close()


rospy.init_node("asdf", anonymous=True)
marker_pub = rospy.Publisher("gps_viz", MarkerArray, latch=True, queue_size=1)

from geometry_msgs.msg import Quaternion as Quaternion_msg
from tf.transformations import quaternion_from_euler

ma = MarkerArray()
stamp = rospy.Time.now()

id = 0
for i,track in enumerate(tracks):
  for j,p in enumerate(track):
    m = Marker()
    m.header.frame_id = "world"
    m.header.stamp = stamp
    m.id = id
    id += 1
    m.type = m.ARROW
    m.pose.position.x = p[0]
    m.pose.position.y = p[1]
    m.pose.position.z = p[2]
    q = quaternion_from_euler(0.0, 0.0, p[3])
    m.pose.orientation = Quaternion_msg(*q)
    m.scale.x = 3.0
    m.scale.y = 0.8
    m.scale.z = 0.8
    m.color.a = 1.0
    m.color.r = 0.0 + (i / 2)
    m.color.g = 0.0
    m.color.b = 1.0 - (i / 2)
    m.lifetime = rospy.Duration(100000)

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
print('hi')

rospy.spin()
