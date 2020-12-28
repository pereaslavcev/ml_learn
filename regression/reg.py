import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

X, y = make_regression(n_samples=100, n_features=1, 
                       n_informative=1, effective_rank=1, 
                       tail_strength=1, noise=3, 
                       random_state=1)

X_scal = StandardScaler().fit_transform(X)
y_scal = StandardScaler().fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scal, y_scal, 
                                                    test_size=0.3, random_state=1)

model = Sequential()
model.add(Dense(10, activation='linear', input_dim=X.shape[1]))
model.add(Dense(1, activation='linear'))
optimizer = Adam(lr=1e-3)
model.compile(optimizer=optimizer, loss='mse', metrics='mae')
model.fit(X_train, y_train, 
          batch_size=10, epochs=100, 
          validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

model.summary()

plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')
plt.plot(X_test, y_pred, color='red', label='pred')
plt.legend()