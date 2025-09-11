#!/bin/bash
# Do not run manually, this will be called from devcontainer.json

sudo chown -R curb /commandhistory
# install rosdeps
rosdep install --from-paths -i -y /workspaces/collaborative-scene-graphs/src/

# install correct np version
pip install numpy==1.23.5

# set git user + email
git config --global user.name 'Tim Steinke'
git config --global user.email 'tim.steinke@pluto.uni-freiburg.de'

source "$HOME/.profile"
source /opt/ros/noetic/setup.bash

# install open3d here (if put in requirements.txt it causes an error)
pip install open3d

# featup local setup
echo 'installing featup'
cd /workspaces/collaborative-scene-graphs/src/FeatUp/
pip install -e .

# setup grounding dino
echo 'installing grounding dino'
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

cd /workspaces/collaborative-scene-graphs
# try to do this in dockerfile
# pip install -r requirements.txt

echo '# set PATH so it includes users private bin if it exists
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi
source /opt/ros/noetic/setup.bash
export PYTHONPATH="$PYTHONPATH:/workspaces/collaborative-scene-graphs/"' >> $HOME/.bashrc

echo 'source /workspaces/collaborative-scene-graphs/devel/setup.bash' >> $HOME/.bashrc
echo 'export ROS_MASTER_URI=http://localhost:11312' >> $HOME/.bashrc

# auto completion in python debugger
echo 'import rlcompleter
pdb.Pdb.complete=rlcompleter.Completer(locals()).complete' > $HOME/.pdbrc
