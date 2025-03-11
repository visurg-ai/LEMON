# @brief  Image with support for: CUDA runtime and pyenv.
# @author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
# @date   11 Aug 2023.

# Image info
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04
#LABEL maintainer="luiscarlos.gph@gmail.com"
#LABEL version="1.0"
#LABEL description="Image with support for lemon, contains CUDA, Intel MKL, faiss-gpu, and pyenv."

# Update package lists and install necessary tools
RUN apt-get update && apt-get install -y curl gnupg

# Create the keyrings directory for storing repository keys
RUN mkdir -p /etc/apt/keyrings

# Add NVIDIA CUDA repository key
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    | gpg --dearmor -o /etc/apt/keyrings/nvidia-cuda.gpg

# Add NVIDIA Machine Learning repository key
RUN curl -fsSL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    | gpg --dearmor -o /etc/apt/keyrings/nvidia-ml.gpg

# Add the NVIDIA CUDA repository to your APT sources
RUN echo "deb [signed-by=/etc/apt/keyrings/nvidia-cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" \
    > /etc/apt/sources.list.d/nvidia-cuda.list

# Add the NVIDIA Machine Learning repository to your APT sources
RUN echo "deb [signed-by=/etc/apt/keyrings/nvidia-ml.gpg] https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/ /" \
    > /etc/apt/sources.list.d/nvidia-ml.list

# Install NVIDIA repository keys
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 134
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install commonly used packages
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y apt-utils aptitude sudo vim zsh wget curl git net-tools tmux locales build-essential zlib1g-dev libbz2-dev libreadline-dev libssl-dev libsqlite3-dev libffi-dev liblzma-dev tk-dev ncurses-dev

# Install locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Set user environment variables and create user account
ARG USER
ARG UID
ARG GID
ENV HOME="/home/$USER"
RUN groupadd -g $UID $USER
RUN useradd -m -u $UID -g $UID -p pass123 -s /bin/zsh $USER
RUN usermod -a -G sudo $USER
RUN touch $HOME/.zshrc && chown $USER:$USER $HOME/.zshrc

# Intall oh-my-zsh so we have a nice terminal
RUN sudo -u $USER wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O $HOME/install.sh
RUN sudo -u $USER chmod +x $HOME/install.sh 
RUN sudo -u $USER $HOME/install.sh --unattended
RUN sudo -u $USER sed -i 's/robbyrussell/terminalparty/g' $HOME/.zshrc

# Install Python's pyenv environment
ENV PYTHON_VERSION="3.10.12"
RUN sudo -u $USER git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
RUN sudo -u $USER echo 'export PYENV_ROOT="$HOME/.pyenv"' >> $HOME/.zshrc
RUN sudo -u $USER echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> $HOME/.zshrc
RUN sudo -u $USER echo 'eval "$(pyenv init --path)"' >> $HOME/.zshrc
RUN sudo -u $USER /bin/zsh -c "source $HOME/.zshrc && pyenv install $PYTHON_VERSION"
RUN sudo -u $USER /bin/zsh -c "source $HOME/.zshrc && pyenv global $PYTHON_VERSION"

# Install ffmpeg
RUN DEBIAN_FRONTEND=noninteractive apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y ffmpeg

# Install swig
RUN DEBIAN_FRONTEND=noninteractive apt install -y swig

# Install Intel MKL
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && gpg --dearmor GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && mv GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB.gpg /usr/share/keyrings/oneapi-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y intel-basekit intel-aikit

# Point PATH and LD_LIBRARY_PATH to the CUDA directories
RUN echo 'export PATH=/usr/local/cuda/bin:${PATH}' >> $HOME/.zshrc
RUN echo 'export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}' >> $HOME/.zshrc
RUN echo '/opt/intel/oneapi/intelpython/latest/lib/libiomp5.so' >> /etc/ld.so.preload

# Install last version of CMake
ENV CMAKE_VERSION="v3.27.2"
RUN DEBIAN_FRONTEND=noninteractive apt remove --purge --auto-remove -y cmake
RUN cd /opt && git clone https://github.com/Kitware/CMake.git && cd CMake && git checkout $CMAKE_VERSION && ./bootstrap && make -j && make install

# Install numpy with MKL support
ENV NUMPY_VERSION="v1.25.2"
RUN cd /opt && git clone https://github.com/numpy/numpy.git
RUN cd /opt/numpy && git checkout $NUMPY_VERSION && git submodule update --init
RUN cd /opt/numpy && echo "[mkl]" >> site.cfg
RUN cd /opt/numpy && echo "library_dirs = /opt/intel/oneapi/mkl/latest/lib/intel64" >> site.cfg
RUN cd /opt/numpy && echo "include_dirs = /opt/intel/oneapi/mkl/latest/include" >> site.cfg
RUN cd /opt/numpy && echo "libraries = mkl_rt" >> site.cfg
RUN chown -R $USER:$USER /opt/numpy
RUN sudo -u $USER /bin/zsh -c "source $HOME/.zshrc && cd /opt/numpy && pip install ."

# Install faiss
ENV FAISS_VERSION="v1.7.4"
RUN cd /opt && git clone https://github.com/facebookresearch/faiss.git
RUN cd /opt/faiss && git checkout $FAISS_VERSION && mkdir build 
RUN /bin/zsh -c "source $HOME/.zshrc && cd /opt/faiss/build && cmake -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_C_API=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2 -DBLA_VENDOR=Intel10_64_dyn -DMKL_LIBRARIES='-L/opt/intel/oneapi/mkl/latest/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core' -DCUDAToolkit_ROOT=/usr/local/cuda -DPython_EXECUTABLE=$HOME/.pyenv/shims/python -DPython_INCLUDE_DIRS=$HOME/.pyenv/versions/$PYTHON_VERSION/include -DPython_LIBRARIES=$HOME/.pyenv/versions/$PYTHON_VERSION/lib .."
RUN cd /opt/faiss/build && make -j faiss
RUN cd /opt/faiss/build && make -j swigfaiss 
RUN cd /opt/faiss/build && make install
RUN chown -R $USER:$USER /opt/faiss
RUN sudo -u $USER /bin/zsh -c "source $HOME/.zshrc && cd /opt/faiss/build/faiss/python && python setup.py install"

# Install PyTorch
ENV PYTORCH_INSTALL_CMD="pip install torch==1.12.1+cu116 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116"
RUN sudo -u $USER /bin/zsh -c "source $HOME/.zshrc && $PYTORCH_INSTALL_CMD"

# Install other dependancies
RUN sudo -u $USER /bin/zsh -c "source $HOME/.zshrc && pip install pandas==1.3.5"
RUN sudo -u $USER /bin/zsh -c "source $HOME/.zshrc && pip install --no-cache-dir yt-dlp selenium ultralytics Pillow lmdb tensorboard"
RUN cd /opt && git clone https://github.com/luiscarlosgph/videosum.git
RUN chown -R $USER:$USER /opt/videosum
RUN sudo -u $USER /bin/zsh -c "source $HOME/.zshrc && cd /opt/videosum && python setup.py install" 


# Start services and run the user-specified command
RUN mkdir -p $HOME/lemon
COPY ../* $HOME/lemon/
WORKDIR $HOME/lemon

ADD files/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["sleep infinity"]
