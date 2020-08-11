
1. create env

```
conda create --name env_xxx_py35 python=3.5
```


2. delete env

```
conda env remove --name myenv
```

3. clone env
```
conda create --name <env_name> --clone base
```

4. set conda source
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```
