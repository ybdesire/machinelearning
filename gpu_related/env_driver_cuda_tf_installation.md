GPU server environment setup and other configuration related knowledge.

# Server

* linux ubuntu 16.04
* 4 GeForce GTX 1080 Ti


# Environment setup

* Install nvidia 1080Ti graphics driver
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-396
```
Afterwards, you can check the Installation with the `nvidia-smi` command, which will report all your CUDA-capable devices in the system.


* Download and install Anaconda `Anaconda3-5.1.0-Linux-x86_64.sh` from [here](https://www.anaconda.com/download/).

* Download CUDA 9.0 `cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb` from [here](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal).

* Install CUDA as cmd below
```
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```

* Download cuDNN for CUDA 9.0 'cuDNN v7.0.5 Library for Linux' `cudnn-9.0-linux-x64-v7.solitairetheme8` from [here](https://developer.nvidia.com/rdp/cudnn-archive)

* Install cuDnn
```
tar zxvf cudnn-9.0-linux-x64-v7.solitairetheme8
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-9.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64
sudo chmod a+r /usr/local/cuda-9.0/lib64/libcudnn*
```

* Check if CUDA installation correctly as [here](http://xcat-docs.readthedocs.io/en/stable/advanced/gpu/nvidia/verify_cuda_install.html)

* Add environment path by `vim ~/.bash_profile`, and `source ~/.bash_profile`

```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda-9.0
export PATH=$PATH:/usr/local/cuda-9.0/bin:/home/ubuntu/anaconda3/bin
```

Then you can create and config your virtual environment.


# Virtual Environment

* my virtual environment

```
conda create --name env_gpu_py35 python=3.5
source activate env_gpu_py35
pip install tensorflow-gpu keras
conda install scikit-learn
conda install jupyter
```