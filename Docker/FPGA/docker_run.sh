#!/usr/bin/env bash

ARGS=("$@")

# Make sure processes in the container can connect to the x server
# Necessary so gazebo can create a context for OpenGL rendering (even headless)
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]; then
    xauth_list=$(xauth nlist $DISPLAY)
    xauth_list=$(sed -e 's/^..../ffff/' <<<"$xauth_list")
    if [ ! -z "$xauth_list" ]; then
        echo "$xauth_list" | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

# Prevent executing "docker run" when xauth failed.
if [ ! -f $XAUTH ]; then
    echo "[$XAUTH] was not properly created. Exiting..."
    exit 1
fi

if [ ! -z "$1" ]; then
    ROS_MASTER_URI=http://$1:11311
    echo "ROS_MASTER $1"
fi

if [ ! -z "$2" ]; then
    ROS_IP=$2
    echo "ROS_IP $2"
fi

BASH_OPTION=bash

xhost +

docker run \
    -it \
    --rm \
    -e DISPLAY=unix$DISPLAY \
    -e XAUTHORITY=$XAUTH \
    -e ROS_MASTER_URI=$ROS_MASTER_URI \
    -e ROS_IP=$ROS_IP \
    -v "$XAUTH:$XAUTH" \
    -v "/dev:/dev" \
    -v "/etc/localtime:/etc/localtime:ro" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -v "/home/$USER/catkin_ws/subt_rl:/home/argsubt/catkin_ws/subt_rl" \
    --user "root:root" \
    --name argsubt \
    --network host \
    --privileged \
    argnctu/subt:FPGA \
    $BASH_OPTION