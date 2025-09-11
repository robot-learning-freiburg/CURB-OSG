from abc import ABC, abstractmethod
from typing import List, Union
from threading import Semaphore

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


# Define some abstract base classes for layers/nodes to inherit
class SGNode(ABC):
    def __init__(self, parent=None):
        self.parent = parent

    @abstractmethod
    def get_marker(self) -> Union[Marker, MarkerArray]:
        pass

    @abstractmethod
    def get_center_point(self) -> Point:
        pass

    def get_parent(self):
        return self.parent

    @abstractmethod
    def reload_params(self):
        pass


class SGLayer(ABC):
    def __init__(self):
        self.lock = Semaphore()

    @abstractmethod
    def get_nodes(self) -> List[SGNode]:
        pass
    
    @abstractmethod
    def reload_params(self):
        pass
