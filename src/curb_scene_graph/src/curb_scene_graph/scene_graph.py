from typing import List, Tuple
import os
from geometry_msgs.msg import Point
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool
from tf import TransformListener
import pickle

from .base_classes import SGLayer
from .static_objects import LandMarkNode, LandMarkLayer
from .road_graph import RoadGraph
from .road_graph_baseline import RoadGraphBaseline
from .road_graph_opengraph import Intersection, Road, RoadGraphOG
from .dynamic_objects import DynamicObjectsLayer, TrackedObject
from .root_layer import RootLayer, RootNode


# SceneGraph is a class instantiated once by the node code
class SceneGraph:
    def __init__(self):
        # single transform listener with long cache that is passed to layers
        self.tl = TransformListener(cache_time=rospy.Duration(120), interpolate=True)

        self.root_layer = RootLayer()

        self.road_graph_method: str = rospy.get_param("~curb/scene_graph/road_graph_method")  # type: ignore
        self.road_graph: RoadGraph

        if self.road_graph_method == "opengraph":
            rospy.loginfo("Using OpenGraph road graph method")
            self.road_graph = RoadGraphOG()
        elif self.road_graph_method == "baseline":
            rospy.loginfo("Using baseline road graph method")
            self.road_graph = RoadGraphBaseline()
        else:
            raise ValueError(f"Unknown road graph method: {self.road_graph_method}")

        self.dyn_obj_layer = DynamicObjectsLayer(self.tl)
        self.landmark_layer = LandMarkLayer(self.tl)

        self.layers = [
            self.root_layer,
            self.road_graph,
            self.dyn_obj_layer,
            self.landmark_layer,
        ]

        self.edge_id_cnt = 0

        self.pub_timer = rospy.Timer(rospy.Duration(4), self.render)
        self.node_marker_pub = rospy.Publisher("sg_nodes", MarkerArray, queue_size=2)
        self.edge_marker_pub = rospy.Publisher("sg_edges", MarkerArray, queue_size=2)

        self.sg_dir: str = rospy.get_param("~scenegraphs_dir")  # type: ignore
        self.dump_sub = rospy.Subscriber("~dump_sg", Bool, self.dump_sg)
        
        # loading not working yet
        # self.load_sg()

        self.render()
        rospy.loginfo("Scene graph node ready")

    def dump_sg(self, _=None):
        rospy.logwarn("Dumping scene graph")
        all_nodes = []
        for layer in self.layers:
            with layer.lock:
                nodes = layer.get_nodes()
                all_nodes.append(nodes)
        with open(f"{self.sg_dir}/sg_dump.pkl", "wb") as f:
            pickle.dump(all_nodes, f)
        rospy.logwarn(f"Dumped {len(self.layers)} layers")
    
    def load_sg(self, _=None):
        raise NotImplementedError
        # if not os.path.exists(f"{self.sg_dir}/sg_dump.pkl"):
        #     rospy.loginfo("No scene graph dump found")
        #     return
        # rospy.logwarn(f"Loading scene graph from {self.sg_dir}/sg_dump.pkl")

        # with open(f"{self.sg_dir}/sg_dump.pkl", "rb") as f:
        #     self.layers = pickle.load(f)
        
        # rospy.logwarn(f"Loaded {len(self.layers)} layers")
        
    def reset(self):
        self.road_graph.reset()

    def get_eraser_marker(self) -> Marker:
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = rospy.Time.now()
        m.action = m.DELETEALL
        return m

    def render(self, _=None):
        edge_ma = MarkerArray()
        edge_ma.markers = [self.get_eraser_marker()]
        self.edge_id_cnt = 0

        node_ma = MarkerArray()
        node_ma.markers = [self.get_eraser_marker()]

        edge_markers, node_markers = self.get_all_markers()
        node_ma.markers += node_markers
        edge_ma.markers += edge_markers

        rospy.loginfo_throttle(30.0, f"SG: pub {len(node_markers)} nodes and {len(edge_markers)} edges")
        self.node_marker_pub.publish(node_ma)
        self.edge_marker_pub.publish(edge_ma)

    def get_all_markers(self) -> Tuple[List[Marker], List[Marker]]:
        edge_markers = []
        node_markers = []

        for layer in self.layers:
            layer: SGLayer
            with layer.lock:
                layer.reload_params()
                for obj in layer.get_nodes():
                    m = obj.get_marker()
                    # marker can be None
                    if m is None:
                        continue
                    if type(m) == Marker:
                        node_markers.append(m)
                    if type(m) == MarkerArray and m.markers:
                        node_markers += m.markers
                edge_markers += self.edge_markers(layer)

        return (edge_markers, node_markers)

    def edge_markers(self, layer: SGLayer) -> List[Marker]:
        """compute incoming edge markers for the given layer (connections from
        the next higher layer). assumes the layer to be locked.
        """
        rgba: List[float] = rospy.get_param("~curb/scene_graph/edge_rgba")
        edge_width: float = rospy.get_param("~curb/scene_graph/edge_width")
        markers = []
        for node in layer.get_nodes():
            if isinstance(node, (TrackedObject, LandMarkNode)):
                node_pos = node.get_center_point()
                parent_pos = self.road_graph.find_kf_pos(node.get_kf_id())
                if node_pos and parent_pos:
                    edge_marker = self.edge_marker(parent_pos, node_pos, rgba, edge_width)
                    edge_marker.ns = str(type(node).__name__)
                    markers.append(edge_marker)
            elif isinstance(node, (Road, Intersection)):
                if isinstance(node, Road):
                    # edges for Roads disabled for now (bug)
                    continue
                parent_pos = self.root_layer.root_node.get_center_point()
                node_pos = node.get_center_point()
                if node_pos:
                    edge_marker = self.edge_marker(parent_pos, node_pos, rgba, edge_width)
                    edge_marker.ns = str(type(node).__name__)
                    markers.append(edge_marker)
            elif isinstance(node, RootNode):
                continue  # no incoming edge for root node
            else:
                raise NotImplementedError

        return markers

    def edge_marker(self, a: Point, b: Point, rgba: List[float], edge_width: float) -> Marker:
        """Creates a marker that draws a fine white line between a and b."""
        m = Marker()
        m.action = m.ADD
        m.id = self.edge_id_cnt
        self.edge_id_cnt += 1
        m.ns = "sg_edges"
        m.header.frame_id = "world"

        m.color.a = rgba[3]
        m.color.r = rgba[0]
        m.color.g = rgba[1]
        m.color.b = rgba[2]

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        m.points = [a, b]

        m.type = m.ARROW
        m.scale.x = edge_width
        m.scale.y = edge_width
        m.scale.z = 0.1

        return m
