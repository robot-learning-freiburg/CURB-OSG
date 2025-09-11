#!/usr/bin/env python

# Try catkin install
from distutils.core import setup
from pathlib import Path

from catkin_pkg.python_setup import generate_distutils_setup

# with open(f'{Path(__file__).parent}/requirements.txt', 'r') as f:
#    pip_dependencies = f.readlines()


d = generate_distutils_setup(
    scripts=["bin/semantic_mapping_node", "bin/masa_node", "bin/offline_masa_processor", "bin/masa_node_precomputed"],
    packages=["curb_projection", "masa", "projects", "open_graph"],
    package_dir={"": "src"},
)

# d.update({"install_requires": pip_dependencies})

setup(**d)
