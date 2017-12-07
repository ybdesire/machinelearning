# Install face_recognition at Ubuntu



## Steps


* (1) anaconda & python 3.6 environment


```
conda create --name envtf
source activate envtf
```


* (2) Install boost

```
sudo apt-get install libboost-all-dev
```


* (3) apt updage

```
apt-get -y update
```

* (4) install essential

```
apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*
```


* (5) get dlib


```
cd ~ && \
mkdir -p dlib && \
git clone -b 'v19.7' --single-branch https://github.com/davisking/dlib.git dlib/ 
```


* (6) modify `dlib/tools/python/CMakeLists.txt` before build dlib

```
Before compiling dlib, please edited dlib's tools/python/CMakeLists.txt file from:

set(USE_SSE4_INSTRUCTIONS ON CACHE BOOL "Use SSE4 instructions")

to:

set(USE_SSE2_INSTRUCTIONS ON CACHE BOOL "Use SSE2 instructions")
```


* (7) build dlib at `dlib` directory

```
mkdir build; cd build; cmake .. -DDLIB_USE_CUDA=0 -DUSE_AVX_INSTRUCTIONS=1; cmake --build .
```

Then, you can `import dlib` at python.


* (8) install dlib python at `dlib` directory

```
python setup.py install
```


* (9) install face_recognition by pip

```
pip install face_recognition
```


Then, you can try examples of face_recognition [here](https://github.com/ageitgey/face_recognition).

