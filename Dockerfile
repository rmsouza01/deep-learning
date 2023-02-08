FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel AS pytorch-base
# Some definitions that we propogate to container's env below
ARG USERNAME=user
ARG WORKSPACE_DIR=/home/user/deep-learning

SHELL ["/bin/bash", "-c"]

# Use a non-root user
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

# Create the user
RUN groupadd --gid $USER_GID ${USERNAME} \
    && useradd --uid $USER_UID --gid $USER_GID -m ${USERNAME}

RUN mkdir ${WORKSPACE_DIR}/ && \
    chown -R $USER_GID:$USER_UID ${WORKSPACE_DIR}

# Some development helpers and add user as a sudoer
RUN apt-get update \
    && apt-get install -y git ssh tmux vim curl htop sudo

RUN echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Setup our users env
USER ${USERNAME}
ENV WORKSPACE_DIR=${WORKSPACE_DIR} \
    PATH="/home/${USERNAME}/.local/bin:${PATH}" \
    NVIDIA_DRIVER_CAPABILITIES="all"

# Add any extra libraries not included in base image
# and ensure we use the cuda11.6 binaries any torch packages to match base image
WORKDIR ${WORKSPACE_DIR}
COPY ./requirements.txt ./
RUN pip install --upgrade pip && \ 
    pip install -r ./requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
ENTRYPOINT ["/bin/bash"]
