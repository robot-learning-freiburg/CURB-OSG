from functools import partial

import rospy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

from hdl_graph_slam.msg import Keyframe_msg


def create_marker(x,y,z,stamp,color, agent_no):
  m = Marker()
  m.header.frame_id = f"robotcar_{agent_no}/initial"
  m.header.stamp = stamp
  m.type = m.SPHERE
  m.scale.x = 5.0
  m.scale.y = 5.0
  m.scale.z = 5.0
  m.color.a = 0.5
  m.color.r = 1.0 - color
  m.color.g = 1.0
  m.color.b = 1.0
  m.pose.position.x = x
  m.pose.position.y = y
  m.pose.position.z = z
  m.pose.orientation.w = 1.0
  m.lifetime = rospy.Duration(1.0)
  m.id = rospy.Time.now().nsecs
  return m


def keyframe_callback(keyframe_msg: Keyframe_msg):
  p = keyframe_msg.odom.position
  m = create_marker(p.x, p.y, p.z, keyframe_msg.header.stamp, 1.0, keyframe_msg.agent_no)
  marker_pub.publish(m)


def odom_callback(agent_id, odom_msg: Odometry):
  p = odom_msg.pose.pose.position
  m = create_marker(p.x, p.y, p.z, odom_msg.header.stamp, 0.0, agent_id)
  marker_pub.publish(m)


rospy.init_node("asdf", anonymous=True)
marker_pub = rospy.Publisher("odom_markers", Marker, queue_size=256, latch=True)
keyframe_sub = rospy.Subscriber("/agent_keyframes", Keyframe_msg, keyframe_callback)
odom_sub = rospy.Subscriber("/robotcar_0/odom_0", Odometry, partial(odom_callback, 0))
odom_sub = rospy.Subscriber("/robotcar_1/odom_1", Odometry, partial(odom_callback, 1))
odom_sub = rospy.Subscriber("/robotcar_2/odom_2", Odometry, partial(odom_callback, 2))

rospy.spin()
