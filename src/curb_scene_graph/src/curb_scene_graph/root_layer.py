from geometry_msgs.msg import Point
from .base_classes import SGLayer, SGNode
from typing import List
import rospy
from visualization_msgs.msg import Marker


class RootNode(SGNode):
    def __init__(self):
        self.pos = Point()
        self.reload_params()
    
    def reload_params(self):
        self.rgba: List[float] = rospy.get_param("~curb/scene_graph/rgba")  # type: ignore
        self.xyz: List[float] = rospy.get_param("~curb/scene_graph/root_pos")  # type: ignore
        self.z_offset: float = rospy.get_param("~curb/scene_graph/root_layer_offset")  # type: ignore
        self.pos.x = self.xyz[0]
        self.pos.y = self.xyz[1]
        self.pos.z = self.xyz[2] + self.z_offset

        

    def get_center_point(self) -> Point:
        return self.pos

    def get_marker(self):
        m = Marker()
        m.action = m.ADD
        m.id = 0
        m.ns = "root_layer"
        m.header.frame_id = "world"
        m.type = m.CUBE

        m.color.r = self.rgba[0]
        m.color.g = self.rgba[1]
        m.color.b = self.rgba[2]
        m.color.a = self.rgba[3]

        m.pose.orientation.w = 1.0

        m.pose.position = self.pos

        m.scale.x = m.scale.y = m.scale.z = 5.0

        return m


class RootLayer(SGLayer):
    def __init__(self):
        super().__init__()
        self.root_node = RootNode()
        self.reload_params()

    def get_nodes(self) -> List[SGNode]:
        return [self.root_node]
    
    def reload_params(self):
        self.root_node.reload_params()
