import numpy as np
np.random.seed(1337)  # for reproducibility
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from dbn.tensorflow import SupervisedDBNClassification
# use "from dbn import SupervisedDBNClassification" for computations on CPU with numpy
from sklearn.datasets import load_iris


iris = load_iris()
data_x = iris.data
data_y = iris.target


x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[500, 1000, 500],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=20, # RBM training steps
                                         n_iter_backprop=50, # ANN training steps
                                         activation_function='relu',
                                         dropout_p=0.2)

classifier.fit(x_train, y_train)
