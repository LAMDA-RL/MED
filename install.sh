#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# should match env name from YAML
ENV_NAME=med
PYTHON_VER=3.8

# setup conda
CONDA_DIR="$(conda info --base)"
source "${CONDA_DIR}/etc/profile.d/conda.sh"

# deactivate the env, if it is active
ACTIVE_ENV_NAME="$(basename ${CONDA_PREFIX})"
if [ "${ENV_NAME}" = "${ACTIVE_ENV_NAME}" ]; then
    conda deactivate
fi

# !!! this removes existing version of the env
conda remove -y -n "${ENV_NAME}" --all

# create conda env
conda create -p "${CONDA_PREFIX}/envs/${ENV_NAME}" python="${PYTHON_VER}" -y
if [ $? -ne 0 ]; then
    echo "*** Failed to create env"
    exit 1
fi

# Let conda set LD_LIBRARY_PATH
pushd "${CONDA_PREFIX}/envs/${ENV_NAME}"

# create files
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh

# set activate variables
echo "export OLD_LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}" >>./etc/conda/activate.d/env_vars.sh
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${CONDA_PREFIX}/envs/${ENV_NAME}/lib" >>./etc/conda/activate.d/env_vars.sh

# set deactivate variables
echo "export LD_LIBRARY_PATH=\${OLD_LD_LIBRARY_PATH}" >>./etc/conda/deactivate.d/env_vars.sh
echo "unset OLD_LD_LIBRARY_PATH" >>./etc/conda/deactivate.d/env_vars.sh

popd

# activate env
conda activate "${ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "*** Failed to activate env"
    exit 1
fi

# double check that the correct env is active
ACTIVE_ENV_NAME="$(basename ${CONDA_PREFIX})"
if [ "${ENV_NAME}" != "${ACTIVE_ENV_NAME}" ]; then
    echo "*** Env is not active, aborting"
    exit 1
fi

pip install -e .
pip install setuptools==59.5.0 wheel==0.38.4
pip install matplotlib tqdm ipdb tensorboard
pip install torch gym==0.21.0 wandb setproctitle
pip install protobuf==3.20.1 absl-py numpy pygame
pip install pettingzoo==1.22.2

# ========== Solve Conflicts ==========
pip install gym==0.21.0
pip install pyglet==1.5.0
pip install importlib-metadata==4.13.0
pip install numpy==1.23.1

# setup pre-commit
pre-commit install
