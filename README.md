Machine Learning
=================

My machine learning code written by Python.

# Environment Setup

1. Install Python 3.5 at Windows10.
2. Install IPython 4.0.3.
3. Install machine learning packages installer [Anaconda](https://www.continuum.io/downloads#_windows).
4. Run IPython and access "http://127.0.0.1:8888" at browser.

```
> ipython notebook
```


# ML libs/packages

**numpy**
* Matrix operations: add, subtraction, inverse.
* [code sample](https://github.com/ybdesire/machinelearning/blob/master/1_numpy/matrix_calc.py)

**matplotlib**
* Python 2D plotting library.
* code sample: samples should be opened by ipython.
   * [draw line](https://github.com/ybdesire/machinelearning/blob/master/3_matplotlib/hello.ipynb)
   * [draw image](https://github.com/ybdesire/machinelearning/blob/master/3_matplotlib/draw_image.ipynb)


# ML algorithms

Samples should be opened by ipython.

## Supervised Learning 

### Classification

**Decision Tree**
* [sklearn DecisionTreeClassifier](https://github.com/ybdesire/machinelearning/blob/master/4_decision_tree/1_DTs_predict.ipynb)
* [digit recognition by kaggle data(MNIST)](https://github.com/ybdesire/machinelearning/blob/master/4_decision_tree/DTs_Digit_Recognition/predict_and_generate_kaggle_result.ipynb)
* [shannon entropy calculation](https://github.com/ybdesire/machinelearning/blob/master/4_decision_tree/calc_shannon.ipynb)


**Random Forest**
* [Random Forest result](https://github.com/ybdesire/machinelearning/blob/master/5_random_forest/RF_digit_recognition.ipynb)
* [Random Forest predict probability](https://github.com/ybdesire/machinelearning/blob/master/5_random_forest/RF_digit_recognition_probability.ipynb)


**Neural Network**
* [Basic Three Layers Network](https://github.com/ybdesire/machinelearning/blob/master/6_NN/toy_example.ipynb)


**Gradient Boosting for classification**
* [GradientBoostingClassifier](https://github.com/ybdesire/machinelearning/blob/master/7_ensemble/sklearn.ensemble%20learn.ipynb)


**CNN (Deep Learning)**
* [CNN for kaggle MNIST](https://github.com/ybdesire/machinelearning/blob/master/6_NN/CNN_mnist_kaggle.ipynb)
* [Create diffirent CNN structure](https://github.com/ybdesire/machinelearning/tree/master/11_CNN)


**XGBoost**
* [Basic useage of XGB](https://github.com/ybdesire/machinelearning/blob/master/10_xgboost/easy_example/main.py)
* [XGB for kaggle digit recognition](https://github.com/ybdesire/machinelearning/blob/master/10_xgboost/xgboost_kaggle_digit_recognition.ipynb)

### Regression

**Linear Regression**
[sklearn LinearRegression and linear function parameters](https://github.com/ybdesire/machinelearning/blob/master/14_regression/Linear_Regression.ipynb)
[difference between  `np.linalg.lstsq` and `linear_model.LinearRegression`](https://github.com/ybdesire/machinelearning/blob/master/14_regression/Diff_np.linalg.lstsq_LinearRegression.ipynb)

## Un-Supervised Learning

[HMM basic](https://github.com/ybdesire/machinelearning/blob/master/15_HMM/basic_hmm.ipynb)



### Cluster

**k-Nearest Neighbor**
* [self-written kNN](https://github.com/ybdesire/machinelearning/blob/master/2_knn/knn.ipynb)
* [kNN by lib sklearn](https://github.com/ybdesire/machinelearning/blob/master/2_knn/KNeighborsClassifier.ipynb)
* [kNN cluster example](https://github.com/ybdesire/machinelearning/blob/master/12_cluster/KNN.ipynb)

### PCA
* [pca algorithm](https://github.com/ybdesire/machinelearning/blob/master/13_data_compression/PCA_demo.ipynb)

# Image process

1. [Image smooth, shift, rotate, zoom by scipy.ndimage](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/image_process_scikit-image.ipynb)
2. Keep gray image pixel value range [0,255] when room/shift/rotate by setting order=1.   commit-220ac520a0d008e74165fe3aace42b93844aedde


# Distributed ML

## Spark

### Spark Cluster Deployment

[Deployment cluster environment](https://github.com/ybdesire/machinelearning/blob/master/16_spark/)

## Jupyter Integrate with Spark

[get sc at jupyter](https://github.com/ybdesire/machinelearning/blob/master/16_spark/spark_local_run_jupyter)

### Spark One Node

[Run at One Node](https://github.com/ybdesire/machinelearning/blob/master/16_spark/spark_local_run_basic)


### Spark Cluster 

[Run at Cluster](https://github.com/ybdesire/machinelearning/blob/master/16_spark/spark_cluster_run_basic)


## Mlib

[recommandation system](https://github.com/ybdesire/machinelearning/blob/master/16_spark/recommendation_system/basic_recommendation_system.ipynb)

