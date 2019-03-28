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
* [random](1_numpy/np.random.ipynb)
* [np.where to find index](1_numpy/npwhere.py)
* [batch replace array values](1_numpy/np_batch_replace.py)
* [np.split](1_numpy/npsplit.py)
* [np.arange](1_numpy/nparange.py)
* [multi-index for 1d array](1_numpy/index_array_1d.py)


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
* [sub-plot](3_matplotlib/sub_plot.ipynb)
* [set x/y axes limits](3_matplotlib/matplotlib_set_xy_axes_limits.ipynb)



## 2.3 scipy

* [leastsq](https://github.com/ybdesire/machinelearning/blob/master/20_scipy/leastsq.ipynb)
* [load matlab mat file](https://github.com/ybdesire/machinelearning/blob/master/20_scipy/load_matlab_mat_file.ipynb)
* [sparse matrix basic](20_scipy/sparse_matrix.ipynb)
* [gaussian filter](20_scipy/gaussian_filter.ipynb)
* [get data distribution probability](20_scipy/get_list_norm_distribution_probability.ipynb)
* [cdist euclidean]( 20_scipy/cdist_euclidean.py)


## 2.4 pandas

* [Series basic for pandas](26_pandas/basic_series.py)
* [DataFrame basic for pandas](26_pandas/basic_dataframe.py)
* [create DataFrame from list of data](26_pandas/create_dataframe_from_list.py)
* [read csv and iterator row by row](26_pandas/read_csv/read_csv.py)
* [read excel and iterator row by row](26_pandas/read_excel/read_excel.py)
* [replace/fill all nan](26_pandas/fillnan.py)
* [parse csv with special seperator](26_pandas/pandas_csv_special_seperator.ipynb)
* [read csv without header](26_pandas/pandas_read_csv_without_header.ipynb)
* [remove missing values row/column](26_pandas/pandas_remove_missing_values_row_column.ipynb)
* [null/missing values](26_pandas/null_value)
* [data merge](26_pandas/movielens_data_analysis/data_merge_by_pandas.py)
* [data sort by count](26_pandas/movielens_data_analysis/sort_top_20_counts.py)
* [data sort by mean](26_pandas/movielens_data_analysis/get_top_20_high_rating.py)
* [modify string to int for each column data](26_pandas/replace_column_value.py)



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
* [XGB for IRIS dataset](https://github.com/ybdesire/machinelearning/blob/master/10_xgboost/easy_example/xgb_iris.ipynb)
* [XGB for kaggle digit recognition](https://github.com/ybdesire/machinelearning/blob/master/10_xgboost/xgboost_kaggle_digit_recognition.ipynb)


**DBN**
* [Install DBN lib at win](27_DBN/install_win.md)
* [DBN code example for python](27_DBN/dbn_iris.py)


**RNN**
* [RNN for MNIST](28_RNN/RNN_MNIST.ipynb)


**DCNN**
* [Deconvolution network](39_deconvolution_network/dcnn_model.ipynb)



### 3.1.2 Regression

**Linear Regression**

* [sklearn LinearRegression and linear function parameters](https://github.com/ybdesire/machinelearning/blob/master/14_regression/Linear_Regression.ipynb)
* [difference between  `np.linalg.lstsq` and `linear_model.LinearRegression`](https://github.com/ybdesire/machinelearning/blob/master/14_regression/Diff_np.linalg.lstsq_LinearRegression.ipynb)
* [regression by CNN](https://github.com/ybdesire/machinelearning/tree/master/11_CNN/cnn_regression_example/cnn_regression_example.ipynb)


## 3.2 Un-Supervised Learning

### 3.2.1 HMM
* [HMM basic](https://github.com/ybdesire/machinelearning/blob/master/15_HMM/basic_hmm.ipynb)
* HMM application
   * https://www.cnblogs.com/wangzming/p/7512607.html
   * https://zhuanlan.zhihu.com/p/20688517


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

### 3.5.1 TF Basic
* [Install Tensorflow at windows](https://github.com/ybdesire/machinelearning/blob/master/23_tensorflow/install_tf_windows.md)
* ["hello world" for basic calculation](23_tensorflow/tensorflow_helloworld.ipynb)
* [basic linear model by tf-core low level api](23_tensorflow/tf_basic_basic_linear_model.ipynb)
* [basic linear model training by tf-core low level api](23_tensorflow/tf_basic_train.ipynb)
* [basic linear model by tf.estimator high level api](23_tensorflow/tf_basic_estimator.ipynb)
* [basic softmax regression at mnist dataset](23_tensorflow/tf_basic_mnist_softmax.ipynb)


## 3.5.2 tf.estimator
[DNNClassifier](23_tensorflow/tf_estimator_DNNClassifier/tf.estimator.DNNClassifier.ipynb)
[set batchsize and epoch](23_tensorflow/tf_estimator_DNNClassifier_batchsize_epoch/tf.estimator.DNNClassifier_epoch_batchsize.ipynb)


### 3.5.3 TF models
* [DNN-200N-100N-50N-10N](23_tensorflow/models/tf_dnn_200N-100N-50N-10N.ipynb)
* [basic CNN](23_tensorflow/tf_basic_cnn.ipynb)
* [CNN-120C5-MP2-200C3-MP2-100N-10N](23_tensorflow/models/tf_cnn_120C5-MP2-200C3-MP2-100N-10N.ipynb)
* [CNN-120C5-200C3-MP2-100N-10N](23_tensorflow/models/tf_cnn_120C5-200C3-MP2-100N-10N.ipynb)


## 3.6 Tensorboard

### 3.6.1 Tensorboard by Tensorflow
* [TF linearmodel Tensorboard](31_tensorboard/tensorboard_tf_basic_linearmodel.ipynb)
* [TF DNN Tensorboard](31_tensorboard/tensorboard_tf_dnn_200N-100N-50N-10N.ipynb)


### 3.6.2 Tensorboard by Keras
* [Tensorboard with keras](31_tensorboard/tensorboard_keras_basic.ipynb)
* [Tensorboard with keras CNN](31_tensorboard/tensorboard_keras_cnn.ipynb)
* [EMBEDDING: mnist data](31_tensorboard/embedding_mnist)
* [EMBEDDING: your own data](31_tensorboard/tensorboard_embedding_your_data.ipynb)



## 3.7 keras

### 3.7.1 basic

* [Install keras at Windows](https://github.com/ybdesire/machinelearning/blob/master/25_keras/install_at_win_conda_tf)
* [Modify keras backend between theano and tf](25_keras//modify_keras_backend.ipynb)
* [Keras basic neural network](25_keras/keras_basic_nn.ipynb)
* [int to one-hot & one-hot to int](25_keras/onehot_argmax.ipynb)
* [get log loss each epoch](25_keras/keras_log_loss_each_epoch.ipynb)
* [model dump, load](25_keras/keras_model_save_load.ipynb)
* [plot loss,acc history](25_keras/keras_plot_loss_acc_history.ipynb)
* [early stop](25_keras/keras_early_stop.ipynb)
* [adam optimization](25_keras/keras_adam_optimization.ipynb)
* [validation](25_keras/keras_validation_split.ipynb)
* [param of model.summary](25_keras/how_to_understand_params.ipynb)


### 3.7.2 models

* [Keras 1D-CNN](25_keras/keras_basic_1dcnn_digit_sklearn.ipynb)
* [Keras 2D-CNN](25_keras/keras_basic_2dcnn_digit_sklearn.ipynb)
* [RNN by keras](25_keras/keras_rnn.ipynb)
* [RESNET-50 by keras](25_keras/keras_resnet50.ipynb)
* [LSTM](25_keras/keras_lstm.ipynb)


## 3.8 theano
* [Install theano at win](29_theano/install/install_theano_at_win.md)


## 3.9 Incremental learning

* [Incremental learning by SGDClassifier partial_fit](https://github.com/ybdesire/machinelearning/blob/master/24_incremental_learning/SGDClassifier_partial_fit.ipynb)


## 3.10 outlier detection

* [IsolationForest](33_outlier_detection/outlier_detection_IsolationForest.ipynb)

## 3.11 sklearn

* [model persistence(dump & load)](37_sklearn/sklearn_model_persistence.ipynb)
* [euclidean distance](37_sklearn/sklearn_distance.py)
* [boston house-prices dataset for regression](37_sklearn/boston_dataset.ipynb)
* [decision tree ensembel with adaboost](37_sklearn/decision_tree_ensembel_with_adaboost.ipynb)
* [kmeans cluster](37_sklearn/kmeans_sklearn.py)
* [mini batch kmeans cluster](37_sklearn/mini_batch_kmeans_sklearn.py)
* [random forest grid search](37_sklearn/randomforest_gridsearch.ipynb)
* [logloss](37_sklearn/sklearn_logloss.ipynb)
* [GMM cluster](37_sklearn/gmm_cluster.ipynb)



## 3.12 jupyter
* [timeit/time performance](44_jupyter/performance_time_timeit.ipynb)
* [jupyter reload module](44_jupyter/auto_reload/jupyter_auto_reload.ipynb)


## 3.13 mxnet

### 3.13.1 NDArray

* [array create/reshape/index](46_mxnet/NDArray/NDArrayBasic.ipynb)
* [array concat](46_mxnet/NDArrayConcat.ipynb)
* [array operation](46_mxnet/NDArrayOperation.ipynb)
* [basic linear regression](46_mxnet/linear_regression_raw.ipynb)
* [linear regression by gluon](46_mxnet/linear_regression_gluon.ipynb)


# 4. Feature Engineering

## 4.1 Working With Text Data

* [Extract 3 types of text feature: bag of words, TF, TF-IDF](https://github.com/ybdesire/machinelearning/blob/master/18_text_feature/work_with_text_data_basic.ipynb)

## 4.2 String Hash

* [Extract string hash value by FeatureHasher](https://github.com/ybdesire/machinelearning/blob/master/18_text_feature/FeatureHasher.ipynb)
* [HashingVectorizer](18_text_feature/HashingVectorizer.ipynb)

## 4.3 Normalization

* [MinMaxScaler](36_fea_eng/normalization/MinMaxScaler.ipynb)
* [StandardScaler](36_fea_eng/normalization/StandardScaler.ipynb)

## 4.4 Feature selection

* [feature selection by chi-square/chi2](43_feature_selection/fea_select_by_chi_squared.py)
* [feature selection by recursive feature elimination/rfe](43_feature_selection/fea_select_by_rfe.py)

## 4.5 imbalance data process
* [RandomOverSampler for Imbalance Data](45_imblearn/oversampling.ipynb)

## 4.6 missing values
* [count null/missing_value item for each column](26_pandas/null_value/column_null_count.py)
* [drop the column if it contains one missing_value](26_pandas/null_value/drop_null_column.py)
* [drop the column if the missing_value count > 0.5*full_count](26_pandas/null_value/drop_null_half_column.py)
* [check the data with null data > 20%](26_pandas/null_value/column_null_count_rate.py)
* [insert mean value for missing values](37_sklearn/preprocess/missing_value_mean_input.py)


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
* [save image/imwrite](8_image_process/opencv/img_read_display/3_img_save.py)
* [cvtColor](8_image_process/opencv/img_read_display/2_img_basic.ipynb)
* [rgb to hsv](8_image_process/opencv/opencv_hsv/rgb_to_hsv_with_diff.py)
* [hsv image](8_image_process/opencv/opencv_hsv/hsv_each_channel.png)



**Preprocess**

* [Image smooth by blurring, Gaussian, median and bilateral filtering](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/opencv/pre_process/smooth.ipynb)
* [modify pixel color](8_image_process/opencv/pre_process/modify_color_pixel/opencv_modify_color.py)
* [modify pixel color fast](8_image_process/opencv/pre_process/modify_color_pixel/opencv_modify_color_fast.py)
* [flattern rgb](8_image_process/opencv/flattern_rgb/flattern_rgb.py)
* [split rgb](8_image_process/opencv/flattern_rgb/split_rgb.py)
* [find connected component](8_image_process/opencv/connected_component/connectedComponents.py)
* [convert white color to green](8_image_process/opencv/white_color_to_green/)
* [blur detection](8_image_process/opencv/blur_detection)


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


### 5.1.4 Geometric Transformations
* [resize](8_image_process/opencv/image_transformation/resize.ipynb)
* [Perspective Transformation](8_image_process/opencv/image_transformation/perspective_transformation.ipynb)
* [Affine Transformation](8_image_process/opencv/image_transformation/affine_transformation.ipynb)


## 5.2 Useful features
* [Image smooth, shift, rotate, zoom by scipy.ndimage](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/image_process_scikit-image.ipynb)
* [image enhancement for ocr](8_image_process/opencv/image_enhance_for_ocr_tesseract/image_enhance_for_ocr.ipynb)
* Keep gray image pixel value range [0,255] when room/shift/rotate by setting order=1.   commit-220ac520a0d008e74165fe3aace42b93844aedde
* [template match](8_image_process/opencv/template_match_demo/template_match_opencv.ipynb)


## 5.3 OCR
* [tesseract basic feature](32_OCR/tesseract/basic_usage)
* [image enhancement for tesseract](8_image_process/opencv/image_enhance_for_ocr_tesseract/image_enhance_for_ocr.ipynb)


## 5.4 3D graph process
* [stl file parse](34_stl_file_3d/parse_stl_file.ipynb)


## 5.5 face_recognition
* [install face_recognition at ubuntu](35_face_recognition/install_at_ubuntu.md)
* [face location](35_face_recognition/face_location/face_location.ipynb)
* [find facial features](35_face_recognition/find_facial_features/find_facial_features.ipynb)
* [face recognition](35_face_recognition/face_recognition/face_recognition.ipynb)


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
* [nltk basic usage such as tokenize/tagging](9_NLP/3_nltk/basic_nltk.ipynb)
* [nltk tag meaning en & zh](9_NLP/3_nltk/basic_nltk_tag.ipynb)
* [identify named entities basic](9_NLP/3_nltk/basic_identify_named_entities.ipynb)
* [nltk load and process built-in data/dataset/corpus](9_NLP/3_nltk/nltk_built-in_dataset.ipynb)
* [nltk load and process external data/dataset/corpus](9_NLP/3_nltk/nltk_external_text.ipynb)
* [normalizing text by stemmer](9_NLP/3_nltk/normalizing_text_stemmer.ipynb)
* [normalizing text by lemmatization](9_NLP/3_nltk/normalizing_text_lemmatization.ipynb)
* [nltk.Text.similar](9_NLP/3_nltk/nltk.Text.similar.ipynb)
* [more details about nltk.Text.similar](9_NLP/3_nltk/details_nltk.Text.similar.ipynb)



## 7.2 word2vec
* [word2vec basic](https://github.com/ybdesire/machinelearning/blob/master/9_NLP/2_word2vec/word2vec_intro.ipynb)
* [gensim word2vec basic](9_NLP/2_word2vec/gensim_basic.ipynb)


## 7.3 Others

* [n-gram](https://github.com/ybdesire/machinelearning/blob/master/9_NLP/1_ngram/3grams.py)
* [preprocess for en text](https://github.com/ybdesire/machinelearning/blob/master/9_NLP/preprocess_en_text/Basic_Preprocess_En.ipynb)
* [Bags of words feature](https://github.com/ybdesire/machinelearning/blob/master/18_text_feature/extract_mine_strings_feature.ipynb)
* [TF, TF-IDF feature](https://github.com/ybdesire/machinelearning/blob/master/18_text_feature/extract_mine_strings_feature.ipynb)

## 7.4 keyword & abstract extraction

* [from Chinese text](9_NLP/keyword_abstract_extraction_zh/)


## 7.5 gensim
* [get dict for text](9_NLP/4_gensim/get_dict_for_text.py)
* [dict filter](9_NLP/4_gensim/dict_filter.py)
* [doc2bow](9_NLP/4_gensim/doc2bow.py)
* [tfidf/tf-idf](9_NLP/4_gensim/tfidf.py)


# 8. Audio

## 8.1 pyAudioAnalysis
* [basic intro](41_pyaudioanalysis/readme.md)
* [frequency and data extraction from wav file](41_pyaudioanalysis/basic_usage.py)
* [audio feature extraction](41_pyaudioanalysis/fea_extraction.py)
* [extract same length features for different audio](41_pyaudioanalysis/fea_extract_with_same_length.py)


# 9. GPU

* [GPU environment setup](38_gpu_related/env_driver_cuda_tf_installation.md)
* [GPU vs CPU basic](38_gpu_related/gpu_vs_cpu_basic.ipynb)
* [check GPU hardware status](38_gpu_related/check_gpu_status.md)
* [check if keras/tensorflow run on GPU](38_gpu_related/check_if_keras_tensorflow_run_on_gpu/Readme.md)
 

# 10. Video
* [read mp4 by opencv py3]( 42_video/opencv/read_video_py3.py )
* [get video fps by opencv py3]( 42_video/opencv/get_mp4_fps.py )



# 11. other machine learning related algorithm
* [correlation analysis](ml_related_alg/correlation_analysis.ipynb)
* [association analysis](ml_related_alg/association_analysis/association_analysis.ipynb)
* [jaccard similarity](19_model_evaluate_selection/jaccard_similarity.ipynb)
* [2 strings/list similarity (unequal length)](ml_related_alg/levenshtein_distance.ipynb)
* [peak detect]( ml_related_alg/peak_detect.ipynb)
* [panel data analysis: fixed/random effects model by linearmodels](ml_related_alg/panel_data_analysis/panel_data_fixed_random_model_by_linearmodels.ipynb)
* [panel data analysis: mixed effects model by statsmodels](ml_related_alg/panel_data_analysis/panel_data_mixed_effects_model_by_statsmodels.ipynb)



# 12. Small project/features

* [Human face image completion](https://github.com/ybdesire/machinelearning/blob/master/8_image_process/image_completion_face.ipynb)
* [Human detection by opencv hug & svm](8_image_process/opencv/image_human_detection/human_detection.ipynb)
* [gibberish text detection](9_NLP/gibberish_detection/gibberish_sentence_detection.ipynb)
* [poker game AI](40_poker_ai/Readme.md)
* [learn deeper for pklearn](40_poker_ai/learn_pklearn_deeper.ipynb)
* [Industrial Control System ICS Cyber Attack Dataset Classification](others/Industrial_Control_System_ICS_Cyber_Attack.ipynb)
* [red wine quelity classification](others/wine_quality_classification.ipynb)
* [chemistry dataset tox21 test by chainer-chemistry](others/test_chainer-chemistry.ipynb)
* [spam email/sms classification](others/spam_email_classification/spam_email_classification.ipynb)
* [jaychou text generate by lstm](others/jaychou_text_generate_lstm/jaychou_generate.ipynb)


