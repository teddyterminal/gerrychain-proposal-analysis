import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import l2

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np

model = Sequential()
model.add(Dense(30, input_dim = 784, activation='relu', kernel_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(20, activation='relu', kernel_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(15, activation='relu', kernel_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='sigmoid'))
model.add(Dense(1, activation = 'sigmoid'))

sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'binary_crossentropy', 
	optimizer = sgd, 
	metrics = ['accuracy'])

print("Reading Data Files..... ")
neutral = pd.read_csv("../data/generated_datasets/PA_BN_50000_20190720")
gop = pd.read_csv("../data/generated_datasets/PA_GOPBN_50000_20190721")
del neutral["Unnamed: 0"]
del gop["Unnamed: 0"]

neutral["Y"] = 0
gop["Y"] = 1

'''
print("Combining Data......")
combined = pd.concat([neutral, gop])

print("Shuffling and sampling data.....")
combined = combined.sample(frac=1).reset_index(drop=True)

Y = combined["Y"].values
del combined["Y"]

X = combined.values

Xtrain = X[:81600]
Ytrain = Y[:81600]
Xtest = X[81600:]
Ytest = Y[81600:]

print(Xtrain, Ytrain, Xtest, Ytest)
'''
print("Training.....")
history = model.fit(x = Xtrain, y = Ytrain, batch_size = 128, epochs = 4, verbose = 2, validation_split = 0.125, shuffle = True)

e = model.evaluate(x = Xtest, y = Ytest, batch_size = 128, verbose = 1)

print(e)

# serialize model to JSON
model_json = model.to_json()
with open("../data/generated_datasets/model.json", "w") as json_file:
    json_file.write(model_json)

Yprob = model.predict(pcatest, verbose = 1)
Yclass = model.predict_classes(pcatest, verbose = 1)
print("Accuracy:", accuracy_score(Ytest, Yclass))
print("Precision:", precision_score(Ytest, Yclass))
print("Recall:", recall_score(Ytest, Yclass))
print("F1:", f1_score(Ytest, Yclass))
print("Cohen-Kappa:", cohen_kappa_score(Ytest, Yclass))
print("AUC:", roc_auc_score(Ytest, Yclass))
print(confusion_matrix(Ytest, Yclass))


