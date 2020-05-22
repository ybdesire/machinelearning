1. at old server

```
(base) [tmp]$ source activate env_xxx
(env_xxx) [tmp]$ conda env export > environment.yml
```

activate special env which you want to export.


2. at new server

```
conda env create --file environment.yml
```

create new environment based on `environment.yml`


