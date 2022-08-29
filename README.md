# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction. 

## Neural Network Model



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
``` python3
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("data1.csv")

df.head()

x=df[["Input"]].values

y=df[["Output"]].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

import tensorflow as tf

model=tf.keras.Sequential([tf.keras.layers.Dense(8,activation='relu'),
                           tf.keras.layers.Dense(16,activation='relu'),
                           tf.keras.layers.Dense(1)])
model.compile(loss="mae",optimizer="adam",metrics=["mse"])

history=model.fit(x_train,y_train,epochs=1000)

import numpy as np

x_test

preds=model.predict(x_test)
np.round(preds)

tf.round(model.predict([[20]]))

pd.DataFrame(history.history).plot()

r=tf.keras.metrics.RootMeanSquaredError()
r(y_test,preds)

```
## Dataset Information

![dataset](https://user-images.githubusercontent.com/75235006/187118047-2afb9318-852b-4bb8-85dc-3e9817aa7eee.png)

## OUTPUT

### Training Loss Vs Iteration Plot
![graph](https://user-images.githubusercontent.com/75235006/187118074-4ee7973e-0763-4086-82e0-90c978f0cffe.png)


### Test Data Root Mean Squared Error

![mean](https://user-images.githubusercontent.com/75235006/187118100-5184dd06-c018-4baa-8a02-456d1f9b7bcb.png)

### New Sample Data Prediction

![1](https://user-images.githubusercontent.com/75235006/187118115-0ffd4e77-1b8d-4129-9bab-21af164b2bb3.png)

## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully
