#!/bin/bash

docker pull ailabsummerschool2025/summerschool-2025:public

if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
fi

# X 서버 연결 테스트
if ! xset q &>/dev/null; then
    echo "X 서버에 연결할 수 없습니다. GUI 세션에서 실행하세요."
    exit 1
fi

xhost +local:docker

XAUTH_FILE=/tmp/.docker.xauth
rm -f $XAUTH_FILE
touch $XAUTH_FILE
xauth nlist $DISPLAY | sed -e 's/^..////ffff/' | xauth -f $XAUTH_FILE nmerge -
chmod 644 $XAUTH_FILE


echo "Finished Setting X11..."
echo "DISPLAY=$DISPLAY"
echo "XAUTH_FILE=$XAUTH_FILE"

docker run -it --rm \
    --name isaac-lab-container \
    --gpus all \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=/tmp/.docker.xauth \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $XAUTH_FILE:/tmp/.docker.xauth:rw  \
    -v $(pwd):/workspace/host \
    --network=host \
    --shm-size=8g \
    ailabsummerschool2025/summerschool-2025:public
