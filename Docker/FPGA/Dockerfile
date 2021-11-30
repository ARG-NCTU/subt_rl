FROM arm64v8/ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO melodic

###################################### user #####################################

ENV SHELL=/bin/bash \
    USER=argsubt \
    UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/${USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${UID} \
    ${USER} 

RUN echo "root:root" | chpasswd
RUN echo "${USER}:111111" | chpasswd

###################################### basic tools #####################################

RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -yq dist-upgrade \
    && apt-get -o Acquire::ForceIPv4=true install -yq --no-install-recommends \
    locales \
    cmake \
    make \
    git \
    vim \
    gedit \
    wget \
    sudo \
    lsb-release \
    build-essential \
    net-tools \
    dirmngr \
    gnupg2 \
    mercurial \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libpcap-dev \
    python-gtk2 \
    python-gobject \
    python-tk \
    python3-pip \
    python3-setuptools \
    python3-opencv \
    python3-numpy \
    python3-empy  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

###################################### ROS #####################################

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -o Acquire::ForceIPv4=true install --no-install-recommends -y \
    ros-$ROS_DISTRO-desktop-full \
    ros-$ROS_DISTRO-serial \
    ros-$ROS_DISTRO-soem \
    ros-$ROS_DISTRO-openslam-gmapping \
    ros-$ROS_DISTRO-geodesy \
    ros-$ROS_DISTRO-cartographer-* \
    ros-$ROS_DISTRO-ddynamic-reconfigure \
    ros-$ROS_DISTRO-perception \
    ros-$ROS_DISTRO-rgbd-launch \
    ros-$ROS_DISTRO-joystick-drivers \
    ros-$ROS_DISTRO-pointcloud-to-laserscan \
    ros-$ROS_DISTRO-robot-localization \
    ros-$ROS_DISTRO-spacenav-node \
    ros-$ROS_DISTRO-tf2-sensor-msgs \
    ros-$ROS_DISTRO-controller-manager \
    ros-$ROS_DISTRO-twist-mux \
    ros-$ROS_DISTRO-velodyne-simulator \
    ros-$ROS_DISTRO-gazebo-ros \
    ros-$ROS_DISTRO-move-base-msgs \
    ros-$ROS_DISTRO-lms1xx \
    ros-$ROS_DISTRO-interactive-marker-twist-server \
    ros-$ROS_DISTRO-ecl-threads \
    ros-$ROS_DISTRO-teleop-twist-joy \
    ros-$ROS_DISTRO-rosserial \
    python-rosdep \
    python-rosinstall \
    python-rosinstall-generator \
    python-wstool \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init \
    && rosdep update

##################################### PIP3 ######################################

RUN pip3 install --upgrade pip setuptools

RUN pip3 install \
    pyyaml \
    rospkg \
    catkin-tools\
    cython \
    image \
    pandas \
    pypozyx \
    gdown

# Pytorch for CPU
RUN wget https://github.com/KumaTea/pytorch-aarch64/releases/download/v1.9.0/torch-1.9.0-cp36-cp36m-linux_aarch64.whl -O \
    torch-1.9.0-cp36-cp36m-linux_aarch64.whl \
    && pip3 install torch-1.9.0-cp36-cp36m-linux_aarch64.whl torchvision \
    && rm -rf  torch-1.9.0-cp36-cp36m-linux_aarch64.whl

##################################### setting ###################################################
RUN chown -R ${USER}:${USER} ${HOME}/
RUN echo "${USER} ALL=(ALL)  ALL" >> /etc/sudoers
RUN echo "export OPENBLAS_CORETYPE=ARMV8" >> /etc/profile

# Support of nvidia-docker 2.0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# setup entrypoint
COPY ./ros_entrypoint.sh /

ENTRYPOINT ["/ros_entrypoint.sh"]

WORKDIR ${HOME}
