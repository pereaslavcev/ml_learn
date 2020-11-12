# remove conda warnings
import warnings
warnings.filterwarnings('ignore')
# import basic libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import ml libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv2D, MaxPool2D
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
# import user libraries

# load train data
train_df = pd.read_csv('mnist_train.csv').sample(frac=0.01)
X_train = MinMaxScaler().fit_transform(train_df.drop('label', axis=1))
y_train = LabelBinarizer().fit_transform(train_df['label'])

# reshape train features
X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1]**0.5), int(X_train.shape[1]**0.5), 1)

# load test data
test_df = pd.read_csv('mnist_test.csv').sample(frac=0.01)
X_test = MinMaxScaler().fit_transform(test_df.drop('label', axis=1))
y_test = LabelBinarizer().fit_transform(test_df['label'])

# reshape test features
X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1]**0.5), int(X_test.shape[1]**0.5), 1)

# define model
def define_model(dropout_rate=0.2):
	# create model
	model = Sequential()
	model.add(Conv2D(8, (3,3), activation='relu', input_shape=(X_test.shape[1], X_test.shape[2], X_test.shape[3])))
	model.add(BatchNormalization())
	model.add(MaxPool2D((2,2)))
	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(dropout_rate))
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(dropout_rate))
	model.add(Dense(10, activation='softmax'))
	# compile model
	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# create model
model = KerasClassifier(build_fn=define_model, verbose=1)

# define search parameters
dropout_rate = [0.2, 0.5]
batch_size = [10, 20]
epochs = [50, 100]
param_grid = dict(dropout_rate=dropout_rate, batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(X_train, y_train)
grid_predict = grid_result.predict(X_test)

# evaluate model
print(grid_result.best_score_, grid_result.best_params_)
y_true = np.argmax(y_test, axis=1)
y_pred = grid_predict
acc = accuracy_score(y_true, y_pred)
print(acc)