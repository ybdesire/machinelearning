# what is the code

* based on this project 
   * https://github.com/chasembowers/poker-learn
   

# environment setup 

* windows-7 os
* (1) install Anaconda (64bit)
* (2) create conda environment "envtfpy2" with python 2.7

```
conda create --name envtfpy2 python=2.7
```


* (3) activate environment "envtfpy2"

```
activate envtfpy2
```

now we come to python 2.7 environment. To deactivate this environment, use: `deactivate envtfpy2`


* (4) install scipy

```
conda install scipy
```


* (5) install scikit-learn

```
conda install scikit-learn
```

* (6) install matplotlib

```
conda install matplotlib
```

   
# how to run the code

* (1) clone the project 

```
git clone https://github.com/chasembowers/poker-learn.git
```

* (2) run the demo code (bankroll_demo.py) at python2.7 environment

```
(envtfpy2) C:\tmp\poker-learn>python bankroll_demo.py
Beginning simulation of 1000 hands.
Players are training...
Complete.
Simulation complete.

Beginning simulation of 10000 hands.
Players are training...
Complete.
Players are training...
Complete.
```
