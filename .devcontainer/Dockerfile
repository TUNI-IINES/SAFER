FROM osrf/ros:humble-desktop-full
ENV WORKSPACE=${WORKSPACE}
ENV USERNAME=ros
ENV DEBIAN_FRONTEND=noninteractive

# Add new user as a sudoer.
RUN useradd -m $USERNAME && \
        echo "$USERNAME:$USERNAME" | chpasswd && \
        usermod --shell /bin/bash $USERNAME && \
        usermod -aG sudo $USERNAME && \
        echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
        chmod 0440 /etc/sudoers.d/$USERNAME && \
        # Replace 1000 with your user/group id
        usermod  --uid 1000 $USERNAME && \
        groupmod --gid 1000 $USERNAME

RUN apt-get update \
   && apt-get -y install --no-install-recommends \
   unzip \
   nmap \
   iputils-ping \
   iproute2 \
   nano \
   git \
   wget \
   curl \
   ca-certificates \
   ffmpeg \
   build-essential \
   cmake \
   pkg-config \
   libopencv-dev \
   python3-dev \
   python3-pip \
   python3-venv \
   python3-scipy \
   python3-matplotlib \
   libjpeg-dev \
   libpng-dev \
   libtiff-dev \
   libatlas-base-dev \
   libgl1-mesa-glx \
   libglib2.0-0 \
   libcanberra-gtk-module

RUN apt-get autoremove -y \
   && apt-get clean -y \
   && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/humble/setup.bash" >> /home/ros/.bashrc
RUN echo '[ -f ${WORKSPACE}/install/setup.bash ] && source ${WORKSPACE}/install/setup.bash' >> /home/ros/.bashrc

# Install ROS CV utilities and image transport (safe if available in the base image)
RUN apt-get update && apt-get install -y --no-install-recommends \
   ros-humble-cv-bridge \
   ros-humble-image-transport

# Install Parrot Sphinx simulator
#RUN curl --fail --silent --show-error --location https://debian.parrot.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/debian.parrot.com.gpg > /dev/null
#RUN echo "deb [signed-by=/usr/share/keyrings/debian.parrot.com.gpg] https://debian.parrot.com/ $(lsb_release -cs) main generic" | sudo tee /etc/apt/sources.list.d/debian.parrot.com.list
#RUN echo "parrot-sphinx sphinx/firmwared_users string ros" | debconf-set-selections
#RUN echo "parrot-sphinx sphinx/license_approval boolean true" | debconf-set-selections
#RUN sudo apt update && sudo apt install -y parrot-sphinx parrot-ue4-empty

# Install CPU PyTorch first (use build arg or adjust for CUDA if desired)
RUN python3 -m pip install --upgrade pip setuptools wheel ultralytics opencv-python-headless \
   pyyaml opencv-contrib-python-headless cython tqdm parrot-olympe==7.5.0 \
   && python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# Install smaller deps after, to avoid large rebuild times.
RUN python3 -m pip install --no-cache-dir setuptools==58.2.0 "numpy<2.0"
