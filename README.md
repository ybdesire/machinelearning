Machine Learning
=================

My machine learning code written by Python.

# 1. Environment Setup

* (1) Install Python 3.5 at Windows10.
* (2) Install IPython 4.0.3.
* (3) Install machine learning packages installer [Anaconda](https://www.continuum.io/downloads#_windows).
* (4) Run IPython and access "http://127.0.0.1:8888" at browser.

```
> ipython notebook
```


# 2. ML libs/packages


## 2.1 numpy

* Matrix operations: add, subtraction, inverse.
* [code sample](https://github.com/ybdesire/machinelearning/blob/master/1_numpy/matrix_calc.py)


## 2.2 matplotlib

* Python 2D plotting library.
* code sample: samples should be opened by ipython.
   * [draw line](https://github.com/ybdesire/machinelearning/blob/master/3_matplotlib/hello.ipynb)
   * [draw image](https://github.com/ybdesire/machinelearning/blob/master/3_matplotlib/draw_image.ipynb)


## 2.3 scipy

* [leastsq](https://github.com/ybdesire/machinelearning/blob/master/20_scipy/leastsq.ipynb)


# 3. ML algorithms

Samples should be opened by ipython.

## 3.1 Supervised Learning 

### 3.1.1 Classification

**Linear Model**
* [Linear Discriminant Analysis](https://github.com/ybdesire/machinelearning/blob/master/21_linear_model/lda.ipynb)


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

### 3.1.2 Regression

**Linear Regression**

* [sklearn LinearRegression and linear function parameters](https://github.com/ybdesire/machinelearning/blob/master/14_regression/Linear_Regression.ipynb)
* [difference between  `np.linalg.lstsq` and `linear_model.LinearRegression`](https://github.com/ybdesire/machinelearning/blob/master/14_regression/Diff_np.linalg.lstsq_LinearRegression.ipynb)

## 3.2 Un-Supervised Learning

[HMM basic](https://github.com/ybdesire/machinelearning/blob/master/15_HMM/basic_hmm.ipynb)

### 3.2.1 Cluster

**k-Nearest Neighbor**
* [self-written kNN](https://github.com/ybdesire/machinelearning/blob/master/2_knn/knn.ipynb)
* [kNN by lib sklearn](https://github.com/ybdesire/machinelearning/blob/master/2_knn/KNeighborsClassifier.ipynb)
* [kNN cluster example](https://github.com/ybdesire/machinelearning/blob/master/12_cluster/KNN.ipynb)

### 3.2.2 PCA
* [pca algorithm](https://github.com/ybdesire/machinelearning/blob/master/13_data_compression/PCA_demo.ipynb)


## 3.3 Model evaluation

* [KFold and StratifiedKFold](https://github.com/ybdesire/machinelearning/blob/master/19_model_evaluate_selection/KFold.ipynb)
* [accuracy score](https://github.com/ybdesire/machinelearning/blob/master/19_model_evaluate_selection/accuracy_score.ipynb)
* [confusion matrix](https://github.com/ybdesire/machinelearning/blob/master/19_model_evaluate_selection/confusion_matrix.ipynb)
* [P, R, F1 value](https://github.com/ybdesire/machinelearning/blob/master/19_model_evaluate_selection/P_R_F1.ipynb)
* [corss validation](https://github.com/ybdesire/machinelearning/blob/master/19_model_evaluate_selection/cross_validation_simplest.ipynb)

# 4. Feature Engineering

## 4.1 Working With Text Data

[Extract 3 types of text feature: bag of words, TF, TF-IDF](https://github.com/ybdesire/machinelearning/blob/master/18_text_feature/work_with_text_data_basic.ipynb)

## 4.2 String Hash

[Extract string hash value by FeatureHasher](https://github.com/ybdesire/machinelearning/blob/master/18_text_feature/FeatureHasher.ipynb)



# 5. Image process

## 5.1 OpenCV
* [Install opencv-python at Win](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/opencv/install_python_opencv_windows.md)
* [Image Read/Cut/Display](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/opencv/img_read_display/1_img_read.py)
* [Image Read/Cut/Display by Jupyter](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/opencv/img_read_display/2_img_basic.ipynb)


## 5.2 Useful features
* [Image smooth, shift, rotate, zoom by scipy.ndimage](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/image_process_scikit-image.ipynb)
* Keep gray image pixel value range [0,255] when room/shift/rotate by setting order=1.   commit-220ac520a0d008e74165fe3aace42b93844aedde


# 6. Distributed ML

## 6.1 Spark

### 6.1.1 Spark Cluster Deployment

* [Deployment cluster environment](https://github.com/ybdesire/machinelearning/blob/master/16_spark/)

### 6.1.2 Jupyter Integrate with Spark

* [get sc at jupyter](https://github.com/ybdesire/machinelearning/blob/master/16_spark/spark_local_run_jupyter)

### 6.1.3 Spark One Node

* [Run at One Node](https://github.com/ybdesire/machinelearning/blob/master/16_spark/spark_local_run_basic)


### 6.1.4 Spark Cluster 

* [Run at Cluster](https://github.com/ybdesire/machinelearning/blob/master/16_spark/spark_cluster_run_basic)
* [Distributed programming demo for feature extraction](https://github.com/ybdesire/machinelearning/blob/master/16_spark/file_process_distributed)
* [Get running slave current directory and IP address](https://github.com/ybdesire/machinelearning/blob/master/16_spark/get_running_slave_address)
* [Submit multiple .py files which have dependency](https://github.com/ybdesire/machinelearning/blob/master/16_spark/spark_submit_multi_py_files)

### 6.1.5 Mlib

* [recommandation system](https://github.com/ybdesire/machinelearning/blob/master/16_spark/recommendation_system/basic_recommendation_system.ipynb)
* [kmeans at cluster](https://github.com/ybdesire/machinelearning/blob/master/16_spark/spark_mllib/kmeans_run_cluster)

## 6.2 Hadoop

### 6.2.1 Environment Setup

* [Standalone Environment Setup](https://github.com/ybdesire/machinelearning/blob/master/17_hadoop/env_setup_standalone)
* [Single Node Cluster Environment Setup](https://github.com/ybdesire/machinelearning/blob/master/17_hadoop/env_setup_cluster_singlenode)

### 6.2.2 Run Hadoop self-example at Standalone mode

* [hadoop example](https://github.com/ybdesire/machinelearning/blob/master/17_hadoop/run_example_standalone)

### 6.2.3 HDFS

* [HDFS basic operation at single node cluster](https://github.com/ybdesire/machinelearning/tree/master/17_hadoop/env_setup_cluster_singlenode#9-hdfs-operation)


# 7. NLP

* [n-gram](https://github.com/ybdesire/machinelearning/blob/master/9_NLP/1_ngram/3grams.py)
* [word2vec basic](https://github.com/ybdesire/machinelearning/blob/master/9_NLP/2_word2vec/word2vec_intro.ipynb)


# 8. Small project/features

* [Human face image completion](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/image_completion_face.ipynb)






