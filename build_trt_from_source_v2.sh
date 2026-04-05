#!/bin/bash
# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/devel?version=1.3.0rc10
set -euo pipefail

TAG="1.3.0rc10"
PULL=0
IMAGE="nvcr.io/nvidia/tensorrt-llm/devel:${TAG}"
WORKDIR=`pwd`
NVIDIA_VISIBLE_DEVICES=${SLURM_STEP_GPUS}
USER_IMAGE_TAG=${IMAGE}-$USER

# 1) Pull base image
step_pull() {
  docker pull ${IMAGE}
  make -C docker ngc-devel_run LOCAL_USER=1 DOCKER_PULL=${PULL} IMAGE_TAG=${TAG}
}

# 2) Build user image
step_build_image() {
  . make-env.sh
  source .env

  DOCKER_PROGRESS=auto
  docker build \
      --progress $DOCKER_PROGRESS \
      --build-arg BASE_IMAGE_WITH_TAG=$IMAGE \
      --build-arg USER_ID=$USER_ID \
      --build-arg USER_NAME=$USER_NAME \
      --build-arg GROUP_ID=$GROUP_ID \
      --build-arg GROUP_NAME=$GROUP_NAME \
      -f docker/Dockerfile.user \
      --tag $USER_IMAGE_TAG \
      .
}

# 3) Run container
step_run_container() {
  if [ -n "${NVIDIA_VISIBLE_DEVICES:-}" ]; then
    echo "Running with ${NVIDIA_VISIBLE_DEVICES}"
  else
    echo "No visible devices, make sure running within slurm alloc"
    exit 1
  fi

  docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
             --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
             --env "CCACHE_DIR=${WORKDIR}/cpp/.ccache" \
             --env "CCACHE_BASEDIR=${WORKDIR}" \
             --env "CONAN_HOME=${WORKDIR}/cpp/.conan" \
             --workdir ${WORKDIR} \
             --tmpfs /tmp:exec \
             --volume ${WORKDIR}:${WORKDIR} \
             $USER_IMAGE_TAG
}

# 4) Within container, build wheel from source
step_build_wheel() {
  ./scripts/build_wheel.py --clean --use_ccache --cuda_architectures=90-real 2>&1 | tee _whl_build.log
}

# 5) Install wheel
step_install_wheel() {
  BUILD_DIR="build"
  export TRTLLM_PRECOMPILED_LOCATION=$(ls ${BUILD_DIR}/tensorrt_llm*.whl | head -1)
  if [[ -n $TRTLLM_PRECOMPILED_LOCATION ]]; then
      echo "Installing precompiled wheel ${TRTLLM_PRECOMPILED_LOCATION} in editable mode..."
  else
      echo "No precompiled wheel found, exiting"
      exit 1
  fi

  TRTLLM_USE_PRECOMPILED=1 python3 -m pip install -v -e .[devel] 2>&1 | tee _editable.install.log
}

main() {
  step_pull
  step_build_image
  step_run_container
  # step_build_wheel and step_install_wheel are meant to run inside the container
}

# Allow calling individual steps: ./build_trt_from_source_v2.sh step_build_image
if [[ $# -gt 0 ]]; then
  "$@"
else
  main
fi
