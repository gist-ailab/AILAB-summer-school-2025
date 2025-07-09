podman pull docker.io/ailabsummerschool2025/summerschool-2025:public

export DISPLAY=:0
xhost +local:root

# Isaac Lab 
sudo podman run -it --rm \
    --name isaac-lab-container \
    --device nvidia.com/gpu=all \
    --security-opt label=disable \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --network=host \
    ailabsummerschool2025/summerschool-2025:public
