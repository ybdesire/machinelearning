# demo to build model with py files to elf/exe by pyinstaller

demo env: python3.8, linux

## steps

1. train and get model 

```shell
python train.py
```

2. build to elf/exe

```shell
pyinstaller main.spec
```

3. run the elf

```shell
cd dist/main/
./main
```



