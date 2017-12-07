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

* [Matrix operations: add, subtraction, inverse](https://github.com/ybdesire/machinelearning/blob/master/1_numpy/matrix_calc.py)
* [linspace](1_numpy/linspace.py)
* [zeros](1_numpy/zeros.py)
* [meshgrid](1_numpy/meshgrid.py)
* [sin/cos/pi](1_numpy/sin_cos_pi.py)

## 2.2 matplotlib

* [draw line](https://github.com/ybdesire/machinelearning/blob/master/3_matplotlib/hello.ipynb)
* [draw image](https://github.com/ybdesire/machinelearning/blob/master/3_matplotlib/draw_image.ipynb)
* [draw 2 lists to bar](https://github.com/ybdesire/machinelearning/blob/master/3_matplotlib/2list_to_bar.ipynb)
* [draw bar with x](https://github.com/ybdesire/machinelearning/blob/master/3_matplotlib/bar.ipynb)
* [draw image without axes](3_matplotlib/plot_image_without_axes.ipynb)
* [figure resize](3_matplotlib/figure_resize.ipynb)
* [plot 2 lines at one fiture](3_matplotlib/plot_2_lines_at_one_figure.ipynb)
* [plot point and line at one fiture](3_matplotlib/plot_line_and_point_at_one_figure.ipynb)
* [3D plot with pca](3_matplotlib/3d_plot_with_pca.ipynb)
* [plot 2 lines with different line width](3_matplotlib/linewidth.ipynb)
* [save to png image](3_matplotlib/save_image_by_matplotlib.ipynb)
* [plot x/y/title label with font_size](3_matplotlib/plot_label_with_size.ipynb)


## 2.3 scipy

* [leastsq](https://github.com/ybdesire/machinelearning/blob/master/20_scipy/leastsq.ipynb)
* [load matlab mat file](https://github.com/ybdesire/machinelearning/blob/master/20_scipy/load_matlab_mat_file.ipynb)
* [sparse matrix basic](20_scipy/sparse_matrix.ipynb)
* [gaussian filter](20_scipy/gaussian_filter.ipynb)
* [get data distribution probability](20_scipy/get_list_norm_distribution_probability.ipynb)


## 2.4 pandas

* [read csv and iterator row by row](26_pandas/read_csv.py)


## 2.5 seaborn

* [hotmap for confusion matrix](30_seaborn/confusion_matrix_heap_map.ipynb)


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


**SVM**
* [SVM with diffirent kernels](https://github.com/ybdesire/machinelearning/blob/master/22_svm/svm_kernel.ipynb)


**Neural Network**
* [Basic Three Layers Network](https://github.com/ybdesire/machinelearning/blob/master/6_NN/toy_example.ipynb)


**Gradient Boosting for classification**
* [GradientBoostingClassifier](https://github.com/ybdesire/machinelearning/blob/master/7_ensemble/sklearn.ensemble%20learn.ipynb)


**CNN (Deep Learning)**
* [CNN for kaggle MNIST](https://github.com/ybdesire/machinelearning/blob/master/6_NN/CNN_mnist_kaggle.ipynb)
* [Create diffirent CNN structure](https://github.com/ybdesire/machinelearning/tree/master/11_CNN)
* [CNN Visualization](25_keras/keras_basic_cnn_visualization.ipynb)


**XGBoost**
* [Basic useage of XGB](https://github.com/ybdesire/machinelearning/blob/master/10_xgboost/easy_example/main.py)
* [XGB for kaggle digit recognition](https://github.com/ybdesire/machinelearning/blob/master/10_xgboost/xgboost_kaggle_digit_recognition.ipynb)


**DBN**
* [Install DBN lib at win](27_DBN/install_win.md)
* [DBN code example for python](27_DBN/dbn_iris.py)


**RNN**
* [RNN for MNIST](28_RNN/RNN_MNIST.ipynb)


### 3.1.2 Regression

**Linear Regression**

* [sklearn LinearRegression and linear function parameters](https://github.com/ybdesire/machinelearning/blob/master/14_regression/Linear_Regression.ipynb)
* [difference between  `np.linalg.lstsq` and `linear_model.LinearRegression`](https://github.com/ybdesire/machinelearning/blob/master/14_regression/Diff_np.linalg.lstsq_LinearRegression.ipynb)
* [regression by CNN](https://github.com/ybdesire/machinelearning/tree/master/11_CNN/cnn_regression_example/cnn_regression_example.ipynb)


## 3.2 Un-Supervised Learning

* [HMM basic](https://github.com/ybdesire/machinelearning/blob/master/15_HMM/basic_hmm.ipynb)

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
* [MSE](https://github.com/ybdesire/machinelearning/blob/master/19_model_evaluate_selection/MSE.ipynb)
* [Logloss](19_model_evaluate_selection/onehot_logloss.ipynb)
* [classification report](19_model_evaluate_selection/classification_report.ipynb)
* [feature selection by feature importance](19_model_evaluate_selection/feature_selection_by_importance.ipynb)


## 3.4 Model selection

* [GridSearchCV](https://github.com/ybdesire/machinelearning/blob/master/19_model_evaluate_selection/grid_search.ipynb)
* [train & test split](19_model_evaluate_selection/train_test_split.ipynb)
* [Learning Curves](19_model_evaluate_selection/learning_curve.ipynb)
* [select k best feature](19_model_evaluate_selection/select_k_best_feature.ipynb)


## 3.5 Tensorflow

* [Install Tensorflow at windows](https://github.com/ybdesire/machinelearning/blob/master/23_tensorflow/install_tf_windows.md)
* ["hello world" for basic calculation](23_tensorflow/tensorflow_helloworld.ipynb)
* [basic linear model by tf-core low level api](23_tensorflow/tf_basic_basic_linear_model.ipynb)
* [basic linear model training by tf-core low level api](23_tensorflow/tf_basic_train.ipynb)
* [basic linear model by tf.estimator high level api](23_tensorflow/tf_basic_estimator.ipynb)
* [basic softmax regression at mnist dataset](23_tensorflow/tf_basic_mnist_softmax.ipynb)
* [basic CNN](23_tensorflow/tf_basic_cnn.ipynb)


## 3.6 Tensorboard
* [Tensorboard with keras](31_tensorboard/tensorboard_keras_basic.ipynb)
* [Tensorboard with keras CNN](31_tensorboard/tensorboard_keras_cnn.ipynb)
* [EMBEDDING: mnist data](31_tensorboard/embedding_mnist)
* [EMBEDDING: your own data](31_tensorboard/tensorboard_embedding_your_data.ipynb)



## 3.7 keras

* [Install keras at Windows](https://github.com/ybdesire/machinelearning/blob/master/25_keras/install_at_win_conda_tf)
* [Keras basic neural network](25_keras/keras_basic_nn.ipynb)
* [Keras 1D-CNN](25_keras/keras_basic_1dcnn_digit_sklearn.ipynb)
* [Keras 2D-CNN](25_keras/keras_basic_2dcnn_digit_sklearn.ipynb)
* [int to one-hot & one-hot to int](25_keras/onehot_argmax.ipynb)
* [RNN by keras](25_keras/keras_rnn.ipynb)
* [RESNET-50 by keras](25_keras/keras_resnet50.ipynb)


## 3.8 theano
* [Install theano at win](29_theano/install/install_theano_at_win.md)


## 3.9 Incremental learning

* [Incremental learning by SGDClassifier partial_fit](https://github.com/ybdesire/machinelearning/blob/master/24_incremental_learning/SGDClassifier_partial_fit.ipynb)


## 3.10 outlier detection

* [IsolationForest](33_outlier_detection/outlier_detection_IsolationForest.ipynb)


# 4. Feature Engineering

## 4.1 Working With Text Data

* [Extract 3 types of text feature: bag of words, TF, TF-IDF](https://github.com/ybdesire/machinelearning/blob/master/18_text_feature/work_with_text_data_basic.ipynb)

## 4.2 String Hash

* [Extract string hash value by FeatureHasher](https://github.com/ybdesire/machinelearning/blob/master/18_text_feature/FeatureHasher.ipynb)
* [HashingVectorizer](18_text_feature/HashingVectorizer.ipynb)


# 5. Image process

## 5.1 OpenCV

### 5.1.1 OpenCV Python

**Installation**
* [Install opencv-python at Win-64 by conda py2 env](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/opencv/install_opencv_windows_by_conda_env_py2.md)
* [Install opencv-python at Win-64 with conda(python3)](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/opencv/install_opencv_windows_by_conda_py3.md)
* [Install opencv-python at Win by py2](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/opencv/install_python_opencv_windows.md)

**Basic**
* [Image Read/Cut/Display](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/opencv/img_read_display/1_img_read.py)
* [Image Read/Cut/Display by Jupyter](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/opencv/img_read_display/2_img_basic.ipynb)

**Preprocess**

* [Image smooth by blurring, Gaussian, median and bilateral filtering](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/opencv/pre_process/smooth.ipynb)

**Projects**
* [defect detection](8_image_process/opencv/defect_detection/defect_detection.ipynb)


### 5.1.2 OpenCV CPP

opencv 2.4.9 & windows-7

* [Init opencv cpp project](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/CppProject/image_pre_process)


### 5.1.3 Features & Matcher
* [SIFT](8_image_process/opencv/SIFT/get_sift_feature.ipynb)
* [Images matching with average distance output](8_image_process/opencv/SIFT/match_by_2_images_navie_ave.ipynb)
* [Images matching with most-similar-average distance output](8_image_process/opencv/SIFT/match_by_2_images_sort_ave.ipynb)
* [Different size images match by ORB feature and BFMatcher](8_image_process/opencv/SIFT/match_diff_size.ipynb)
* [Different direction images match by ORB feature and BFMatcher](8_image_process/opencv/SIFT/match_rotate.ipynb)



## 5.2 Useful features
* [Image smooth, shift, rotate, zoom by scipy.ndimage](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/image_process_scikit-image.ipynb)
* Keep gray image pixel value range [0,255] when room/shift/rotate by setting order=1.   commit-220ac520a0d008e74165fe3aace42b93844aedde


## 5.3 OCR
* [tesseract basic feature](32_OCR/tesseract/basic_usage)


## 5.4 3D graph process
* [stl file parse](34_stl_file_3d/parse_stl_file.ipynb)


## 5.5 face_recognition
* [install face_recognition at ubuntu](35_face_recognition/install_at_ubuntu.md)



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

## 7.1 nltk
* [nltk basic](https://github.com/ybdesire/machinelearning/blob/master/9_NLP/3_nltk/nltk_intro.ipynb)


## 7.2 word2vec
* [word2vec basic](https://github.com/ybdesire/machinelearning/blob/master/9_NLP/2_word2vec/word2vec_intro.ipynb)
* [gensim word2vec basic](9_NLP/2_word2vec/gensim_basic.ipynb)


## 7.3 Others

* [n-gram](https://github.com/ybdesire/machinelearning/blob/master/9_NLP/1_ngram/3grams.py)
* [preprocess for en text](https://github.com/ybdesire/machinelearning/blob/master/9_NLP/preprocess_en_text/Basic_Preprocess_En.ipynb)
* [Bags of words feature](https://github.com/ybdesire/machinelearning/blob/master/18_text_feature/extract_mine_strings_feature.ipynb)
* [TF, TF-IDF feature](https://github.com/ybdesire/machinelearning/blob/master/18_text_feature/extract_mine_strings_feature.ipynb)


# 8. other machine learning related algorithm
* [correlation analysis](ml_related_alg/correlation_analysis.ipynb)
* [association analysis](ml_related_alg/association_analysis/association_analysis.ipynb)
* [jaccard similarity](19_model_evaluate_selection/jaccard_similarity.ipynb)
* [2 strings/list similarity (unequal length)](ml_related_alg/levenshtein_distance.ipynb)
* [peak detect]( ml_related_alg/peak_detect.ipynb)
* [panel data analysis: fixed/random effects model by linearmodels](ml_related_alg/panel_data_analysis/panel_data_fixed_random_model_by_linearmodels.ipynb)
* [panel data analysis: mixed effects model by statsmodels](ml_related_alg/panel_data_analysis/panel_data_mixed_effects_model_by_statsmodels.ipynb)



# 9. Small project/features

* [Human face image completion](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/image_completion_face.ipynb)
* [Human detection by opencv hug & svm](8_image_process/opencv/image_human_detection/human_detection.ipynb)
* [gibberish text detection](9_NLP/gibberish_detection/gibberish_sentence_detection.ipynb)




