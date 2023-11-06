ARG TF_VERSION=2.10.0
FROM tensorflow/tensorflow:${TF_VERSION}-gpu

# Create non-root user.
ARG USERNAME=vscode$USER
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    # Add user to sudoers.
    apt-get update && \
    apt-get install -y sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    # Change default shell to bash.
    usermod --shell /bin/bash $USERNAME

# Install git including LFS.
RUN apt-get update && \
    apt-get install -y git && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs

# To enable plotting.
RUN apt-get update && \
    apt-get install -y libcairo2-dev libgirepository1.0-dev libgtk-3-dev python3-gi-cairo ffmpeg && \
    pip install pycairo PyGObject

#Install necessities for making C++ files
RUN apt update && \
    apt install g++ 

# Using devtoolset with correct manylinux2014 libraries.
ARG PREFIX=/dt9/usr
ARG CC=gcc
ARG CXX="${PREFIX}/bin/g++"
ARG LIBDIR="${PREFIX}/lib"
ARG INCLUDEDIR="${PREFIX}/include"
ARG CFLAGS="-O3 -march=x86-64 -mtune=generic -fPIC"

# Install FFTW3.
RUN cd /opt && \
    curl -sL http://www.fftw.org/fftw-3.3.9.tar.gz | tar xz && \
    cd fftw-3.3.9 && \
    ./configure CC="${CC}" CFLAGS="${CFLAGS}" --enable-openmp --enable-float && \
    make && \
    make install && \
    ./configure CC="${CC}" CFLAGS="${CFLAGS}" --enable-openmp && \
    make && \
    make install

# Install spiral waveform.
RUN cd /opt && \
    git clone https://github.com/mrphys/spiral-waveform --branch v1.0.0 && \
    cd spiral-waveform && \
    make install

# Install TFMRI.
ARG TFMRI_VERSION=0.22.0
RUN apt-get update && \
    apt-get install -y libopenexr-dev && \
    cd /opt && \
    git clone -b wip1200 --single-branch https://github.com/mrphys/tensorflow-mri.git && \
    cd tensorflow-mri && \
    make lib CXX=g++ && \
    pip install .

# Install tensorflow utilities.
RUN pip install tensorflow-datasets tensorflow-addons tensorboard 

# Install additional packages.
RUN pip install ffmpeg-python scipy scikit-learn scikit-image ipykernel google-api-python-client pydicom jupyter

