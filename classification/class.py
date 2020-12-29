import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

X, y = make_classification(n_samples=100, n_features=2, 
                           n_informative=2, n_redundant=0, 
                           n_repeated=0, n_classes=2, 
                           random_state=1)

X_scal = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scal, y, 
                                                    test_size=0.3, random_state=1)

model = Sequential()
model.add(Dense(10, activation='relu', input_dim=X.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
optimizer = Adam(lr=1e-3)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics='acc')
model.fit(X_train, y_train, 
          batch_size=10, epochs=100, 
          validation_data=(X_test, y_test))

y_pred = model.predict_classes(X_test).flatten()

model.summary()

sns.scatterplot(X_test[:,0], X_test[:,1], hue=y_pred)