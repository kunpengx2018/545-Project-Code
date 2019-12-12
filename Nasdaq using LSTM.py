
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import random 


# In[47]:


nasdaq=pd.read_csv('nasdaq.csv')
x = nasdaq[['CVol', 'Return', 'Spread', 'Diff','5days mean','10days mean','5days std']]
y = nasdaq['Return']
test_len = round(len(x)*0.2)
x_test = x[0:test_len]
x_train = x[test_len:]
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# In[55]:


data1 = []
sequence_length = 5 #Set the number of days to traceback
for i in range(len(x_train) - sequence_length - 1):
    data1.append(x_train[i+1: i + sequence_length + 1])
reshaped_train = np.array(data1).astype('float64')

data2 = []
for i in range(len(x_test) - sequence_length - 1):
    data2.append(x_test[i+1: i + sequence_length + 1])
reshaped_test = np.array(data2).astype('float64')

y_test = y[0:test_len]
y_train = y[test_len:]
y_train = y_train[:- sequence_length - 1]
y_test = y_test[:- sequence_length - 1]
reshaped_train.shape


# In[49]:


def return_to_label(returns):
    labels=[]
    for y in returns:
        tag = 0
        if y>2:
            tag = 1
        elif y>0:
            tag = 2
        elif y>-2:
            tag = 3
        else:
            tag = 4
        labels.append(tag)
    return labels


# In[50]:


def build_model():
    model = Sequential()
    model.add(LSTM(input_shape=(reshaped_train.shape[1],reshaped_train.shape[2]), output_dim=42, return_sequences=True))
#     print(model.layers)
    model.add(LSTM(42, return_sequences=False))
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mae', optimizer='rmsprop')
    return model


def train_model(train_x, train_y, test_x, test_y,batch_size):
    model = build_model()
#     try:
    model.fit(train_x, train_y, batch_size, nb_epoch=50, validation_split=0.1,shuffle = False)
    predict = model.predict(test_x)
    predict = np.reshape(predict, (predict.size, ))
#     except KeyboardInterrupt:
#         print(predict)
#         print(test_y)
    return predict


# In[51]:


test_label = return_to_label(y_test)


# In[56]:


predict=train_model(reshaped_train, y_train, reshaped_test, y_test,40)


# In[57]:


test_label = return_to_label(y_test)
pred_label = return_to_label(predict)
np.sum([pred_label[i]==test_label[i] for i in range(len(pred_label))])/len(pred_label)


# In[58]:


import sklearn.metrics
sklearn.metrics.confusion_matrix(test_label, pred_label)

