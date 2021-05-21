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

* [Matrix operations: add, subtraction, inverse](https://github.com/ybdesire/machinelearning/blob/master/numpy/matrix_calc.py)
* [linspace](numpy/linspace.py)
* [zeros](numpy/zeros.py)
* [meshgrid](numpy/meshgrid.py)
* [sin/cos/pi](numpy/sin_cos_pi.py)
* [random](numpy/np.random.ipynb)
* [np.where to find index](numpy/npwhere.py)
* [batch replace array values](numpy/np_batch_replace.py)
* [np.split](numpy/npsplit.py)
* [np.arange](numpy/nparange.py)
* [multi-index for 1d array](numpy/index_array_1d.py)
* [polynomial fit](numpy/polynomial_fit.ipynb)


## 2.2 matplotlib

* [draw line](https://github.com/ybdesire/machinelearning/blob/master/matplotlib/hello.ipynb)
* [draw image](https://github.com/ybdesire/machinelearning/blob/master/matplotlib/draw_image.ipynb)
* [draw 2 lists to bar](https://github.com/ybdesire/machinelearning/blob/master/matplotlib/2list_to_bar.ipynb)
* [draw bar with x](https://github.com/ybdesire/machinelearning/blob/master/matplotlib/bar.ipynb)
* [draw image without axes](matplotlib/plot_image_without_axes.ipynb)
* [figure resize](matplotlib/figure_resize.ipynb)
* [plot 2 lines at one fiture](matplotlib/plot_2_lines_at_one_figure.ipynb)
* [plot point and line at one fiture](matplotlib/plot_line_and_point_at_one_figure.ipynb)
* [3D plot with pca](matplotlib/3d_plot_with_pca.ipynb)
* [plot 2 lines with different line width](matplotlib/linewidth.ipynb)
* [save to png image](matplotlib/save_image_by_matplotlib.ipynb)
* [plot x/y/title label with font_size](matplotlib/plot_label_with_size.ipynb)
* [sub-plot](matplotlib/sub_plot.ipynb)
* [set x/y axes limits](matplotlib/matplotlib_set_xy_axes_limits.ipynb)
* [add legend](matplotlib/add_legend.py)
* [plot multi-lines with labels for each line](matplotlib/plot_multi_lines_with_label.ipynb)
* [plot confusion matrix](matplotlib/plot_confusion_matrix.ipynb)


## 2.3 scipy

* [leastsq](https://github.com/ybdesire/machinelearning/blob/master/scipy/leastsq.ipynb)
* [load matlab mat file](https://github.com/ybdesire/machinelearning/blob/master/scipy/load_matlab_mat_file.ipynb)
* [sparse matrix basic](scipy/sparse_matrix.ipynb)
* [gaussian filter](scipy/gaussian_filter.ipynb)
* [get data distribution probability](scipy/get_list_norm_distribution_probability.ipynb)
* [cdist euclidean]( scipy/cdist_euclidean.py)
* [downsample/down-sample data](https://github.com/ybdesire/machinelearning/blob/master/scipy/re-sample%2Bor%2Bdown-sample.ipynb)
* [optimize with constraints equal](scipy/scipy_optimize_with_constraints_eq.ipynb)
* [load matlab mat file then fft](scipy/load_matlab_mat_file_then_fft.ipynb)


## 2.4 pandas

* [Series basic for pandas](pandas/basic_series.py)
* [DataFrame basic for pandas](pandas/basic_dataframe.py)
* [create DataFrame from list of data](pandas/create_dataframe_from_list.py)
* [read csv and iterator row by row](pandas/read_csv/read_csv.py)
* [read excel and iterator row by row](pandas/read_excel/read_excel.py)
* [replace/fill all nan](pandas/fillnan.py)
* [parse csv with special seperator](pandas/pandas_csv_special_seperator.ipynb)
* [read csv without header](pandas/pandas_read_csv_without_header.ipynb)
* [remove missing values row/column](pandas/pandas_remove_missing_values_row_column.ipynb)
* [null/missing values](pandas/null_value)
* [drop columns contains only zero](pandas/null_value/drop_columns_contains_only_zero.py)
* [data merge](pandas/movielens_data_analysis/data_merge_by_pandas.py)
* [data sort by count](pandas/movielens_data_analysis/sort_top_20_counts.py)
* [data sort by mean](pandas/movielens_data_analysis/get_top_20_high_rating.py)
* [modify string to int for each column data](pandas/replace_column_value.py)
* [count special column values distribution](pandas/value_count.py)
* [write excel to multiple sheets](pandas/excel_to_multi_sheet.ipynb)
* [merge two df: hstack](pandas/hstack.ipynb)
* [merge two df: vstack](pandas/vstack.ipynb)


## 2.5 seaborn

* [hotmap for confusion matrix](seaborn/confusion_matrix_heap_map.ipynb)
* [seaborn clustermap](seaborn/seaborn_clustermap.ipynb)
* [catplot](seaborn/seaborn_catplot.ipynb)


# 3. ML algorithms

Samples should be opened by ipython.

## 3.1 Supervised Learning 

### 3.1.1 Classification

**Linear Model**
* [Linear Discriminant Analysis](https://github.com/ybdesire/machinelearning/blob/master/linear_model/lda.ipynb)


**Decision Tree**
* [sklearn DecisionTreeClassifier](https://github.com/ybdesire/machinelearning/blob/master/decision_tree/1_DTs_predict.ipynb)
* [digit recognition by kaggle data(MNIST)](https://github.com/ybdesire/machinelearning/blob/master/decision_tree/DTs_Digit_Recognition/predict_and_generate_kaggle_result.ipynb)
* [shannon entropy calculation](https://github.com/ybdesire/machinelearning/blob/master/decision_tree/calc_shannon.ipynb)

**Random Forest**
* [Random Forest result](https://github.com/ybdesire/machinelearning/blob/master/random_forest/RF_digit_recognition.ipynb)
* [Random Forest predict probability](https://github.com/ybdesire/machinelearning/blob/master/random_forest/RF_digit_recognition_probability.ipynb)


**SVM**
* [SVM with diffirent kernels](https://github.com/ybdesire/machinelearning/blob/master/svm/svm_kernel.ipynb)


**Neural Network**
* [Basic Three Layers Network](https://github.com/ybdesire/machinelearning/blob/master/NN/toy_example.ipynb)


**Gradient Boosting for classification**
* [GradientBoostingClassifier](https://github.com/ybdesire/machinelearning/blob/master/ensemble/sklearn.ensemble%20learn.ipynb)


**CNN (Deep Learning)**
* [CNN for kaggle MNIST](https://github.com/ybdesire/machinelearning/blob/master/NN/CNN_mnist_kaggle.ipynb)
* [Create diffirent CNN structure](https://github.com/ybdesire/machinelearning/tree/master/CNN)
* [CNN Visualization](keras/keras_basic_cnn_visualization.ipynb)


**XGBoost**
* [Basic useage of XGB](https://github.com/ybdesire/machinelearning/blob/master/xgboost/easy_example/main.py)
* [XGB for IRIS dataset](https://github.com/ybdesire/machinelearning/blob/master/xgboost/easy_example/xgb_iris.ipynb)
* [XGB for kaggle digit recognition](https://github.com/ybdesire/machinelearning/blob/master/xgboost/xgboost_kaggle_digit_recognition.ipynb)


**DBN**
* [Install DBN lib at win](DBN/install_win.md)
* [DBN code example for python](DBN/dbn_iris.py)


**RNN**
* [RNN for MNIST](RNN/RNN_MNIST.ipynb)


**DCNN**
* [Deconvolution network](deconvolution_network/dcnn_model.ipynb)



### 3.1.2 Regression

**Linear Regression**

* [sklearn LinearRegression and linear function parameters](https://github.com/ybdesire/machinelearning/blob/master/regression/Linear_Regression.ipynb)
* [difference between  `np.linalg.lstsq` and `linear_model.LinearRegression`](https://github.com/ybdesire/machinelearning/blob/master/regression/Diff_np.linalg.lstsq_LinearRegression.ipynb)
* [regression by CNN](https://github.com/ybdesire/machinelearning/tree/master/CNN/cnn_regression_example/cnn_regression_example.ipynb)


## 3.2 Un-Supervised Learning

### 3.2.1 HMM
* [HMM basic](https://github.com/ybdesire/machinelearning/blob/master/HMM/basic_hmm.ipynb)
* HMM application
   * https://www.cnblogs.com/wangzming/p/7512607.html
   * https://zhuanlan.zhihu.com/p/20688517


### 3.2.1 Cluster

**k-Nearest Neighbor**
* [self-written kNN](https://github.com/ybdesire/machinelearning/blob/master/knn/knn.ipynb)
* [kNN by lib sklearn](https://github.com/ybdesire/machinelearning/blob/master/knn/KNeighborsClassifier.ipynb)
* [kNN cluster example](https://github.com/ybdesire/machinelearning/blob/master/cluster/KNN.ipynb)

**DBSCAN**
* [dbscan with precomputed metric](sklearn/sklearn_dbscan_distance_matrix_pre_computed.ipynb)

### 3.2.2 PCA
* [pca algorithm](https://github.com/ybdesire/machinelearning/blob/master/data_compression/PCA_demo.ipynb)


## 3.3 Model evaluation

* [KFold and StratifiedKFold](https://github.com/ybdesire/machinelearning/blob/master/model_evaluate_selection/KFold.ipynb)
* [accuracy score](https://github.com/ybdesire/machinelearning/blob/master/model_evaluate_selection/accuracy_score.ipynb)
* [confusion matrix](https://github.com/ybdesire/machinelearning/blob/master/model_evaluate_selection/confusion_matrix.ipynb)
* [P, R, F1 value](https://github.com/ybdesire/machinelearning/blob/master/model_evaluate_selection/P_R_F1.ipynb)
* [corss validation](https://github.com/ybdesire/machinelearning/blob/master/model_evaluate_selection/cross_validation_simplest.ipynb)
* [MSE](https://github.com/ybdesire/machinelearning/blob/master/model_evaluate_selection/MSE.ipynb)
* [Logloss](model_evaluate_selection/onehot_logloss.ipynb)
* [classification report](model_evaluate_selection/classification_report.ipynb)
* [feature selection by feature importance](model_evaluate_selection/feature_selection_by_importance.ipynb)
* [homogeneity & completeness & v_measure for unsupervised learning cluster evaluation metric](cluster/unsupervised_learning_metrics.ipynb)


## 3.4 Model selection

* [GridSearchCV](https://github.com/ybdesire/machinelearning/blob/master/model_evaluate_selection/grid_search.ipynb)
* [train & test split](model_evaluate_selection/train_test_split.ipynb)
* [Learning Curves](model_evaluate_selection/learning_curve.ipynb)
* [select k best feature](model_evaluate_selection/select_k_best_feature.ipynb)


## 3.5 Tensorflow

### 3.5.1 TF Basic
* [Install Tensorflow at windows](https://github.com/ybdesire/machinelearning/blob/master/tensorflow/install_tf_windows.md)
* ["hello world" for basic calculation](tensorflow/tensorflow_helloworld.ipynb)
* [basic linear model by tf-core low level api](tensorflow/tf_basic_basic_linear_model.ipynb)
* [basic linear model training by tf-core low level api](tensorflow/tf_basic_train.ipynb)
* [basic linear model by tf.estimator high level api](tensorflow/tf_basic_estimator.ipynb)
* [basic softmax regression at mnist dataset](tensorflow/tf_basic_mnist_softmax.ipynb)


## 3.5.2 tf.estimator
[DNNClassifier](tensorflow/tf_estimator_DNNClassifier/tf.estimator.DNNClassifier.ipynb)
[set batchsize and epoch](tensorflow/tf_estimator_DNNClassifier_batchsize_epoch/tf.estimator.DNNClassifier_epoch_batchsize.ipynb)


### 3.5.3 TF models
* [DNN-200N-100N-50N-10N](tensorflow/models/tf_dnn_200N-100N-50N-10N.ipynb)
* [basic CNN](tensorflow/tf_basic_cnn.ipynb)
* [CNN-120C5-MP2-200C3-MP2-100N-10N](tensorflow/models/tf_cnn_120C5-MP2-200C3-MP2-100N-10N.ipynb)
* [CNN-120C5-200C3-MP2-100N-10N](tensorflow/models/tf_cnn_120C5-200C3-MP2-100N-10N.ipynb)


## 3.6 Tensorboard

### 3.6.1 Tensorboard by Tensorflow
* [TF linearmodel Tensorboard](tensorboard/tensorboard_tf_basic_linearmodel.ipynb)
* [TF DNN Tensorboard](tensorboard/tensorboard_tf_dnn_200N-100N-50N-10N.ipynb)


### 3.6.2 Tensorboard by Keras
* [Tensorboard with keras](tensorboard/tensorboard_keras_basic.ipynb)
* [Tensorboard with keras CNN](tensorboard/tensorboard_keras_cnn.ipynb)
* [EMBEDDING: mnist data](tensorboard/embedding_mnist)
* [EMBEDDING: your own data](tensorboard/tensorboard_embedding_your_data.ipynb)



## 3.7 keras

### 3.7.1 basic

* [Install keras at Windows](https://github.com/ybdesire/machinelearning/blob/master/keras/install_at_win_conda_tf)
* [Modify keras backend between theano and tf](keras//modify_keras_backend.ipynb)
* [Keras basic neural network](keras/keras_basic_nn.ipynb)
* [int to one-hot & one-hot to int](keras/onehot_argmax.ipynb)
* [get log loss each epoch](keras/keras_log_loss_each_epoch.ipynb)
* [model dump, load](keras/keras_model_save_load.ipynb)
* [plot loss,acc history](keras/keras_plot_loss_acc_history.ipynb)
* [early stop](keras/keras_early_stop.ipynb)
* [adam optimization](keras/keras_adam_optimization.ipynb)
* [validation](keras/keras_validation_split.ipynb)
* [param of model.summary](keras/how_to_understand_params.ipynb)
* [keras fixed model random seed](keras/keras_fixed_model_random_seed.ipynb)


### 3.7.2 models

* [Keras 1D-CNN](keras/keras_basic_1dcnn_digit_sklearn.ipynb)
* [Keras 2D-CNN](keras/keras_basic_2dcnn_digit_sklearn.ipynb)
* [RNN by keras](keras/keras_rnn.ipynb)
* [RESNET-50 by keras](keras/keras_resnet50.ipynb)
* [LSTM](keras/keras_lstm.ipynb)
* [LSTM+Attention](attention/multi-head-attention-test.ipynb)
* [complex attention models](attention/multi-head-attention-complex-structure.ipynb)

### 3.7.3 complex

* [get middle layer output](keras/dnn_get_specific_layer_output.ipynb)


## 3.8 theano
* [Install theano at win](theano/install/install_theano_at_win.md)


## 3.9 Incremental learning

* [Incremental learning by SGDClassifier partial_fit](https://github.com/ybdesire/machinelearning/blob/master/incremental_learning/SGDClassifier_partial_fit.ipynb)


## 3.10 outlier detection

* [IsolationForest](outlier_detection/outlier_detection_IsolationForest.ipynb)

## 3.11 sklearn

* [model persistence(dump & load)](sklearn/sklearn_model_persistence.ipynb)
* [euclidean distance](sklearn/sklearn_distance.py)
* [boston house-prices dataset for regression](sklearn/boston_dataset.ipynb)
* [decision tree ensembel with adaboost](sklearn/decision_tree_ensembel_with_adaboost.ipynb)
* [kmeans cluster](sklearn/kmeans_sklearn.py)
* [mini batch kmeans cluster](sklearn/mini_batch_kmeans_sklearn.py)
* [random forest grid search](sklearn/randomforest_gridsearch.ipynb)
* [logloss](sklearn/sklearn_logloss.ipynb)
* [GMM cluster](sklearn/gmm_cluster.ipynb)
* [TSNE/T-SNE plot](sklearn/tsne_iris_sklearn.ipynb)
* [F1-P-R calculation for multi-class output by sklearn](sklearn/F1-P-R+calculation+for+multi-class+output+by+sklearn.ipynb)
* [Isomap plot](sklearn/dimensionality_reduction_Isomap.ipynb	)


## 3.12 jupyter
* [timeit/time performance](jupyter/performance_time_timeit.ipynb)
* [jupyter reload module](jupyter/auto_reload/jupyter_auto_reload.ipynb)


## 3.13 mxnet

### 3.13.1 NDArray

* [array create/reshape/index](mxnet/NDArray/NDArrayBasic.ipynb)
* [array concat](mxnet/NDArrayConcat.ipynb)
* [array operation](mxnet/NDArrayOperation.ipynb)
* [basic linear regression](mxnet/linear_regression_raw.ipynb)
* [linear regression by gluon](mxnet/linear_regression_gluon.ipynb)


### 3.13.2 Basic

* [get dataset fashion mnist](mxnet/dataset_fashion_mnist.ipynb)
* [softmax regression by gloun](mxnet/softmax_regression_gloun.ipynb)
* [plot activation relu/sigmoid with gradient](mxnet/plot_activation_func.ipynb)
* [mlp and model parameter access](mxnet/mlp.ipynb)
* [CNN LeNet](mxnet/cnn_lenet.ipynb)
* [dataset jaychou](mxnet/dataset_jaychou.ipynb)
* [generate jay text by rnn](mxnet/rnn_generate_jaychou.ipynb)
* [compute gradient](mxnet/compute_gradient.ipynb)


# 4. Feature Engineering

## 4.1 Working With Text Data

* [Extract 3 types of text feature: bag of words, TF, TF-IDF](https://github.com/ybdesire/machinelearning/blob/master/text_feature/work_with_text_data_basic.ipynb)

## 4.2 String Hash

* [Extract string hash value by FeatureHasher](https://github.com/ybdesire/machinelearning/blob/master/text_feature/FeatureHasher.ipynb)
* [HashingVectorizer](text_feature/HashingVectorizer.ipynb)

## 4.3 Normalization

* [MinMaxScaler](fea_eng/normalization/MinMaxScaler.ipynb)
* [StandardScaler](fea_eng/normalization/StandardScaler.ipynb)

## 4.4 Feature selection

* [feature selection by chi-square/chi2](feature_selection/fea_select_by_chi_squared.py)
* [feature selection by recursive feature elimination/rfe](feature_selection/fea_select_by_rfe.py)

## 4.5 imbalance data process
* [RandomOverSampler for Imbalance Data](imblearn/oversampling.ipynb)

## 4.6 missing values
* [count null/missing_value item for each column](pandas/null_value/column_null_count.py)
* [drop the column if it contains one missing_value](pandas/null_value/drop_null_column.py)
* [drop the column if the missing_value count > 0.5*full_count](pandas/null_value/drop_null_half_column.py)
* [check the data with null data > 20%](pandas/null_value/column_null_count_rate.py)
* [insert mean value for missing values](sklearn/preprocess/missing_value_mean_input.py)


# 5. Image process

## 5.1 OpenCV

### 5.1.1 OpenCV Python

**Installation**
* [Install opencv-python at Win-64 by conda py2 env](https://github.com/ybdesire/machinelearning/blob/master/image_process/opencv/install_opencv_windows_by_conda_env_py2.md)
* [Install opencv-python at Win-64 with conda(python3)](https://github.com/ybdesire/machinelearning/blob/master/image_process/opencv/install_opencv_windows_by_conda_py3.md)
* [Install opencv-python at Win by py2](https://github.com/ybdesire/machinelearning/blob/master/image_process/opencv/install_python_opencv_windows.md)

**Basic**
* [Image Read/Cut/Display](https://github.com/ybdesire/machinelearning/blob/master/image_process/opencv/img_read_display/1_img_read.py)
* [Image Read/Cut/Display by Jupyter](https://github.com/ybdesire/machinelearning/blob/master/image_process/opencv/img_read_display/2_img_basic.ipynb)
* [save image/imwrite](image_process/opencv/img_read_display/3_img_save.py)
* [cvtColor](image_process/opencv/img_read_display/2_img_basic.ipynb)
* [rgb to hsv](image_process/opencv/opencv_hsv/rgb_to_hsv_with_diff.py)
* [hsv image](image_process/opencv/opencv_hsv/hsv_each_channel.png)



**Preprocess**

* [Image smooth by blurring, Gaussian, median and bilateral filtering](https://github.com/ybdesire/machinelearning/blob/master/image_process/opencv/pre_process/smooth.ipynb)
* [modify pixel color](image_process/opencv/pre_process/modify_color_pixel/opencv_modify_color.py)
* [modify pixel color fast](image_process/opencv/pre_process/modify_color_pixel/opencv_modify_color_fast.py)
* [flattern rgb](image_process/opencv/flattern_rgb/flattern_rgb.py)
* [split rgb](image_process/opencv/flattern_rgb/split_rgb.py)
* [find connected component](image_process/opencv/connected_component/connectedComponents.py)
* [convert white color to green](image_process/opencv/white_color_to_green/)
* [blur detection](image_process/opencv/blur_detection)


**Projects**
* [defect detection](image_process/opencv/defect_detection/defect_detection.ipynb)


### 5.1.2 OpenCV CPP

opencv 2.4.9 & windows-7

* [Init opencv cpp project](https://github.com/ybdesire/machinelearning/blob/master/image_process/CppProject/image_pre_process)


### 5.1.3 Features & Matcher
* [SIFT](image_process/opencv/SIFT/get_sift_feature.ipynb)
* [Images matching with average distance output](image_process/opencv/SIFT/match_by_2_images_navie_ave.ipynb)
* [Images matching with most-similar-average distance output](image_process/opencv/SIFT/match_by_2_images_sort_ave.ipynb)
* [Different size images match by ORB feature and BFMatcher](image_process/opencv/SIFT/match_diff_size.ipynb)
* [Different direction images match by ORB feature and BFMatcher](image_process/opencv/SIFT/match_rotate.ipynb)


### 5.1.4 Geometric Transformations
* [resize](image_process/opencv/image_transformation/resize.ipynb)
* [Perspective Transformation](image_process/opencv/image_transformation/perspective_transformation.ipynb)
* [Affine Transformation](image_process/opencv/image_transformation/affine_transformation.ipynb)


## 5.2 Useful features
* [Image smooth, shift, rotate, zoom by scipy.ndimage](https://github.com/ybdesire/machinelearning/blob/master/image_process/image_process_scikit-image.ipynb)
* [image enhancement for ocr](image_process/opencv/image_enhance_for_ocr_tesseract/image_enhance_for_ocr.ipynb)
* Keep gray image pixel value range [0,255] when room/shift/rotate by setting order=1.   commit-220ac520a0d008e74165fe3aace42b93844aedde
* [template match](image_process/opencv/template_match_demo/template_match_opencv.ipynb)


## 5.3 OCR
* [tesseract basic feature](OCR/tesseract/basic_usage)
* [image enhancement for tesseract](image_process/opencv/image_enhance_for_ocr_tesseract/image_enhance_for_ocr.ipynb)


## 5.4 3D graph process
* [stl file parse](stl_file_3d/parse_stl_file.ipynb)


## 5.5 face_recognition
* [install face_recognition at ubuntu](face_recognition/install_at_ubuntu.md)
* [face location](face_recognition/face_location/face_location.ipynb)
* [find facial features](face_recognition/find_facial_features/find_facial_features.ipynb)
* [face recognition](face_recognition/face_recognition/face_recognition.ipynb)


# 6. Distributed ML

## 6.1 Spark

### 6.1.1 Spark Cluster Deployment

* [Deployment cluster environment](https://github.com/ybdesire/machinelearning/blob/master/spark/)

### 6.1.2 Jupyter Integrate with Spark

* [get sc at jupyter](https://github.com/ybdesire/machinelearning/blob/master/spark/spark_local_run_jupyter)

### 6.1.3 Spark One Node

* [Run at One Node](https://github.com/ybdesire/machinelearning/blob/master/spark/spark_local_run_basic)


### 6.1.4 Spark Cluster 

* [Run at Cluster](https://github.com/ybdesire/machinelearning/blob/master/spark/spark_cluster_run_basic)
* [Distributed programming demo for feature extraction](https://github.com/ybdesire/machinelearning/blob/master/spark/file_process_distributed)
* [Get running slave current directory and IP address](https://github.com/ybdesire/machinelearning/blob/master/spark/get_running_slave_address)
* [Submit multiple .py files which have dependency](https://github.com/ybdesire/machinelearning/blob/master/spark/spark_submit_multi_py_files)

### 6.1.5 Mlib

* [recommandation system](https://github.com/ybdesire/machinelearning/blob/master/spark/recommendation_system/basic_recommendation_system.ipynb)
* [kmeans at cluster](https://github.com/ybdesire/machinelearning/blob/master/spark/spark_mllib/kmeans_run_cluster)

### 6.1.6 spark at aws emr
* [steps to create/run spark code at emr](spark/spark_at_aws_emr/readme.md)



## 6.2 Hadoop

### 6.2.1 Environment Setup

* [Standalone Environment Setup](https://github.com/ybdesire/machinelearning/blob/master/hadoop/env_setup_standalone)
* [Single Node Cluster Environment Setup](https://github.com/ybdesire/machinelearning/blob/master/hadoop/env_setup_cluster_singlenode)

### 6.2.2 Run Hadoop self-example at Standalone mode

* [hadoop example](https://github.com/ybdesire/machinelearning/blob/master/hadoop/run_example_standalone)

### 6.2.3 HDFS

* [HDFS basic operation at single node cluster](https://github.com/ybdesire/machinelearning/tree/master/hadoop/env_setup_cluster_singlenode#9-hdfs-operation)

### 6.2.4 mrjob
* [word count for mrjob map reduce basic](hadoop/mrjob/word_count/main.py)



# 7. NLP

## 7.1 nltk
* [nltk basic usage such as tokenize/tagging](NLP/3_nltk/basic_nltk.ipynb)
* [nltk tag meaning en & zh](NLP/3_nltk/basic_nltk_tag.ipynb)
* [identify named entities basic](NLP/3_nltk/basic_identify_named_entities.ipynb)
* [nltk load and process built-in data/dataset/corpus](NLP/3_nltk/nltk_built-in_dataset.ipynb)
* [nltk load and process external data/dataset/corpus](NLP/3_nltk/nltk_external_text.ipynb)
* [normalizing text by stemmer](NLP/3_nltk/normalizing_text_stemmer.ipynb)
* [normalizing text by lemmatization](NLP/3_nltk/normalizing_text_lemmatization.ipynb)
* [nltk.Text.similar](NLP/3_nltk/nltk.Text.similar.ipynb)
* [more details about nltk.Text.similar](NLP/3_nltk/details_nltk.Text.similar.ipynb)
* [sentiment analysis](NLP/sentiment_analysis.ipynb)
* [offline isntall nltk_data](https://blog.csdn.net/qq_43140627/article/details/103895811)


## 7.2 word2vec
* [word2vec basic](https://github.com/ybdesire/machinelearning/blob/master/NLP/2_word2vec/word2vec_intro.ipynb)
* [gensim word2vec basic](NLP/2_word2vec/gensim_basic.ipynb)


## 7.3 Others

* [n-gram](https://github.com/ybdesire/machinelearning/blob/master/NLP/1_ngram/3grams.py)
* [preprocess for en text](https://github.com/ybdesire/machinelearning/blob/master/NLP/preprocess_en_text/Basic_Preprocess_En.ipynb)
* [Bags of words feature](https://github.com/ybdesire/machinelearning/blob/master/text_feature/extract_mine_strings_feature.ipynb)
* [TF, TF-IDF feature](https://github.com/ybdesire/machinelearning/blob/master/text_feature/extract_mine_strings_feature.ipynb)
* [basic text clean](NLP/preprocess_en_text/text_clean.py)


## 7.4 keyword & abstract extraction

* [from Chinese text](NLP/keyword_abstract_extraction_zh/)


## 7.5 gensim
* [get dict for text](NLP/4_gensim/get_dict_for_text.py)
* [dict filter](NLP/4_gensim/dict_filter.py)
* [doc2bow](NLP/4_gensim/doc2bow.py)
* [tfidf/tf-idf](NLP/4_gensim/tfidf.py)


## 7.6 AllenNLP

* [install allennlp](allennlp/readme.md)
* [NER by biLSTM with CRF layer and ELMo embeddingstrained on the CoNLL-2003 NER dataset 2018](allennlp/ner_basic.py)
* [fact predict by trained decomposable attention model 2017](allennlp/fact_predict.py)
* [question answer from passage by BiDAF 2017](allennlp/question_answer_passage.py)


## 7.7 Spacy

* [NER by spacy](spacy/ner_spacy_en.ipynb)
* [Tokenization with Part-of-speech (POS) Tagging](spacy/tokenization_pos_tag.ipynb)


## 7.8 gensim

* [Latent Dirichlet Allocation，LDA Model](gensim/lda_model.ipynb)
* [doc to BOW features](gensim/doc_2_bow_zh.ipynb)


## 7.9 keras-bert

* [Get text embeddings by pretrained BERT model](bert/keras_bert_pretrained_model.ipynb)

## 7.10 wordcloud
* [plot wordcloud basic]( wordcloud/wordcloud_basic.ipynb)

## 7.11 wordnet
* [wordnet basic and environment setup](wordnet/wordnet_basic_and_env_setup)

## 7.12 NER
* [BiLSTM-CRF-NER](bilstm_crf_ner)

## 7.13 LDA
* [LDA of sklearn](sklearn/lda_topic_model_by_sklearn.ipynb)


# 8. Audio

## 8.1 pyAudioAnalysis
* [basic intro](pyaudioanalysis/readme.md)
* [frequency and data extraction from wav file](pyaudioanalysis/basic_usage.py)
* [audio feature extraction](pyaudioanalysis/fea_extraction.py)
* [extract same length features for different audio](pyaudioanalysis/fea_extract_with_same_length.py)

## 8.2 signal data augmentation
* [add gaussian noise](numpy/gaussian_noise.ipynb)


# 9. GPU

* [GPU environment setup](gpu_related/env_driver_cuda_tf_installation.md)
* [GPU vs CPU basic](gpu_related/gpu_vs_cpu_basic.ipynb)
* [check GPU hardware status](gpu_related/check_gpu_status.md)
* [check if keras/tensorflow run on GPU](gpu_related/check_if_keras_tensorflow_run_on_gpu/Readme.md)
* [矩池云GPU](https://www.matpool.com/) 

# 10. Video
* [read mp4 by opencv py3]( video/opencv/read_video_py3.py )
* [get video fps by opencv py3]( video/opencv/get_mp4_fps.py )


# 11. recommandation system

## 11.1 surprise

* [load local dataset by pandas](surprise/load_local_dataset_by_pandas.py)
* [load local dataset by surprise](surprise/load_local_dataset_by_surprise.py)
* [load buidin dataset by surprise](surprise/load_buildin_dataset_by_surprise.py)
* [ml-100k dataset movie name to id, or id to movie](surprise/id_name_by_pandas.py)
* [basic/official example](surprise/basic_but_with_full_func_surprise.py)
* [basic algorithm and testing for local data](surprise/basic_alg_test_local_data_surprise.py)
* [predict rating for test dataset](surprise/predict_rating_for_test_dataset.py)
* [user based collaborative filtering and predict one item rating](surprise/user_based_cf.py)
* [item based collaborative filtering and predict one item rating](surprise/item_based_cf.py)
* [top-n recommandation for buildin data](surprise/top_n_recommend_buildin_data_surprise.py)
* [top-n recommandation for local data](surprise/top_n_recommend_local_data_surprise.py)
* [SVD collaborative filtering](surprise/svd_cf.py)


# 12. other machine learning related algorithm
* [correlation analysis](related_alg/correlation_analysis.ipynb)
* [association analysis](related_alg/association_analysis/association_analysis.ipynb)
* [jaccard similarity](model_evaluate_selection/jaccard_similarity.ipynb)
* [2 strings/list similarity (unequal length)](related_alg/levenshtein_distance.ipynb)
* [peak detect]( related_alg/peak_detect.ipynb)
* [panel data analysis: fixed/random effects model by linearmodels](related_alg/panel_data_analysis/panel_data_fixed_random_model_by_linearmodels.ipynb)
* [panel data analysis: mixed effects model by statsmodels](related_alg/panel_data_analysis/panel_data_mixed_effects_model_by_statsmodels.ipynb)



# 13. Small project/features

* [Human face image completion](https://github.com/ybdesire/machinelearning/blob/master/image_process/image_completion_face.ipynb)
* [Human detection by opencv hug & svm](image_process/opencv/image_human_detection/human_detection.ipynb)
* [gibberish text detection](NLP/gibberish_detection/gibberish_sentence_detection.ipynb)
* [poker game AI](poker_ai/Readme.md)
* [learn deeper for pklearn](poker_ai/learn_pklearn_deeper.ipynb)
* [Industrial Control System ICS Cyber Attack Dataset Classification](others/Industrial_Control_System_ICS_Cyber_Attack.ipynb)
* [red wine quelity classification](others/wine_quality_classification.ipynb)
* [chemistry dataset tox21 test by chainer-chemistry](others/test_chainer-chemistry.ipynb)
* [spam email/sms classification](others/spam_email_classification/spam_email_classification.ipynb)
* [jaychou text generate by lstm](others/jaychou_text_generate_lstm/jaychou_generate.ipynb)
* [SI/SIR model by scipy](related_alg/si_sir_simulation.ipynb)

# 14. related tools

## 14.1 conda

* [create env, delete env](conda/create_delete_env.md)
* [export env to yml, create env from yml](conda/export_create_env_yml.md)


# 15. front-end AI

## 15.1 JS access camera

* [js open camera and take photo](js_ai/js_get_camera_video/open_camera_and_take_photo.html)

## 15.2 face-api.js

* [run face-api.js examples]()


# 16. D3.js
* [tutorials and referencs](d3/1.basic)
* [js, select, svg basics](d3/1.basic)

