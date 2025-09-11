# FROM nvidia/
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# get manpages
RUN yes | unminimize

# Set the locale & timezone
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update \
    && apt-get install -y --no-install-recommends locales tzdata \
    && rm -rf /var/lib/atp/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    clangd \
    wget \
    curl \
    openssh-client \
    htop \
    fzf \
    tree \
    git \
    man \
    sudo \
    less \
    lsb-release \
    x11-apps \
    moreutils \
    gdb \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# install ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    ros-noetic-geodesy \
    ros-noetic-pcl-ros \
    ros-noetic-ros-numpy \
    ros-noetic-nmea-msgs \
    ros-noetic-vision-msgs \
    ros-noetic-jsk-recognition-msgs \
    ros-noetic-jsk-rviz-plugins \
    ros-noetic-libg2o \
    python3-rospy \
    python3-catkin-tools \
    python3-pip \
    python3-urllib3 \
    python-is-python3 \
    cppcheck \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# create non-root user & grant sudo rights
RUN useradd -ms /bin/bash curb && echo "curb:curb" | chpasswd && adduser curb sudo \
    && sed -i /etc/sudoers -re 's/^%sudo.*/%sudo ALL=(ALL:ALL) NOPASSWD: ALL/g' \
    && mkdir -p /home/curb/.local/bin && chown -R curb /home/curb/
ENV PATH="/home/curb/.local/bin:${PATH}"

# persistent BASH history
# https://code.visualstudio.com/remote/advancedcontainers/persist-bash-history
RUN SNIPPET="export HISTFILE=/commandhistory/.bash_history" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R curb /commandhistory \
    && echo "$SNIPPET" >> "/home/curb/.bashrc"

# Setup rosdep
# RUN apt-get install -y python3-rosdep
# RUN rosdep init
USER curb
# RUN rosdep update

# install torch
#RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -i https://download.pytorch.org/whl/cu111
# RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html \
RUN pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu111 \
    && rm -r /home/curb/.cache/pip

COPY ../requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && rm -r /home/curb/.cache/pip

# setup MASA deps
RUN mim install mmengine \
    && pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html \
    && python3  -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('punkt_tab')" \
    && rm -r /home/curb/.cache/pip


# install ros deps
# RUN rosdep install --from-paths -i -y /workspaces/collaborative-scene-graphs/src/

CMD bash
