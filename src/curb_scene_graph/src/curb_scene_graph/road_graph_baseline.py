import rospy
from hdl_graph_slam.msg import KeyframeArray_msg
from .road_graph import RoadGraph
from .base_classes import SGLayer, SGNode

class RoadGraphBaseline(RoadGraph):
    """ Road Graph baselline based on the intersections of agent trajectories
    with tracked objects trajectories.
    """
    def __init__(self):
        self.keyframes_sub = rospy.Subscriber(
            "keyframes", KeyframeArray_msg, self.keyframes_callback, queue_size=1
        )

    def reload_params(self):
        pass
    
    def keyframes_callback(self, msg):
        for keyframe_msg in msg.keyframes:
            keyframe = SGNode(keyframe_msg)
            self.add_node(keyframe)
            self.add_edge(keyframe)