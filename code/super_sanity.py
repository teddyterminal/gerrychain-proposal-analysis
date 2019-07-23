import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

import pandas as pd
import numpy as np
import random

model = Sequential()
model.add(Dense(1, activation = 'sigmoid'))

sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'binary_crossentropy', 
	optimizer = sgd, 
	metrics = ['accuracy'])

'''print("Reading Data Files..... ")
neutral = pd.read_csv("../data/generated_datasets/PA_BN_50000_20190720")
print("Second File")
gop = pd.read_csv("../data/generated_datasets/PA_GOPBN_50000_20190721")
print("adding zeros.....")
del neutral["Unnamed: 0"]
del gop["Unnamed: 0"]

neutral["Y"] = 0
gop["Y"] = 1

print(gop)
print(neutral)

print("Combining Data......")
combined = pd.concat([neutral, gop])
print(combined)

print("Shuffling and sampling data.....")
combined = combined.sample(frac=1).reset_index(drop=True)
print(combined)

Y = combined["Y"].values
del combined["Y"]

X = combined.values
'''
Y = [random.randint(0, 1) for i in range(102000)]

X = []
for i in range(102000):
	q = [1 if random.random() < 0.185 else 0 for i in range(8921)]
	X.append(q)

Y = np.array(Y)
X = np.array(X)

Xtrain = X[:81600]
Ytrain = Y[:81600]
Xtest = X[81600:]
Ytest = Y[81600:]

print(Xtrain, Ytrain, Xtest, Ytest)

print("Training.....")
history = model.fit(x = Xtrain, y = Ytrain, batch_size = 128, epochs = 3, 
	verbose = 2, validation_split = 0.125, shuffle = True)

e = model.evaluate(x = Xtest, y = Ytest, batch_size = 128, verbose = 1)

print(e)

# serialize model to JSON
model_json = model.to_json()
with open("../data/generated_datasets/model.json", "w") as json_file:
    json_file.write(model_json)


