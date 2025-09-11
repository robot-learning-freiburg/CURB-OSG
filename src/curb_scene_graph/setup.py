#!/usr/bin/env python

# Try catkin install
from distutils.core import setup
from pathlib import Path

from catkin_pkg.python_setup import generate_distutils_setup

# with open(f'{Path(__file__).parent}/requirements.txt', 'r') as f:
#    pip_dependencies = f.readlines()


d = generate_distutils_setup(
    scripts=["bin/curb_scene_graph_node"],
    packages=["curb_scene_graph", "curb_sg_eval"],
    package_dir={"": "src"},
)

# d.update({"install_requires": pip_dependencies})

setup(**d)
