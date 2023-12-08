#!/bin/bash

IMAGE_NAME="iiyama/hierarchical-wm"
CONTAINER_NAME="iiyama-hierarchical-wm"
GPU_NUMBER=""

CMD="/bin/bash"


# Parse the arguments
while getopts ":g:" opt; do
  case ${opt} in
    g )
      GPU_NUMBER="${OPTARG}"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    : )
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Check if the GPU number is provided
if [ -z "$GPU_NUMBER" ]; then
  echo "Please provide the GPU number using the -g option."
  exit 1
fi

# Set the container name
CONTAINER_NAME="$CONTAINER_NAME$GPU_NUMBER"

# Allow access to the X server
xhost +local:docker

# Check if a container with the same name is already running
if [ $(docker ps -aq -f name=$CONTAINER_NAME) ]; then
    echo "Container with the name $CONTAINER_NAME already exists."
    if [ $(docker ps -q -f name=$CONTAINER_NAME) ]; then
        echo "Entering the existing running container..."
        docker exec -it $CONTAINER_NAME $CMD
    else
        echo "Starting the existing stopped container..."
        docker start -ai $CONTAINER_NAME
    fi
    exit
fi


docker run -it --rm \
    --runtime=nvidia \
    --gpus device=$GPU_NUMBER \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/src:/workspace \
    -v ~/mujoco210:/root/.mujoco/mujoco210 \
    --ipc=host \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    --name $CONTAINER_NAME $IMAGE_NAME \
    $CMD


# Allow access to the X server
xhost -local:docker