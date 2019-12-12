
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import random 


# In[2]:


sp500=pd.read_csv('SP500_update.csv')
# sp500.head()


# In[3]:


random.seed(417)
x = sp500[['CVol', 'Return', 'Spread', 'Diff','5days mean','10days mean','5days std']]
y = sp500['Return']
test_len = round(len(x)*0.2)
x_test = x[0:test_len]
x_train = x[test_len:]
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# In[5]:


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


# In[6]:


def return_to_label(returns):
    labels=[]
    for y in returns:
        tag = 0
        if y > 2:
            tag = 1
        elif y>0:
            tag = 2
        elif y > -2:
            tag = 3
        else:
            tag = 4
        labels.append(tag)
    return labels


# In[13]:


def build_model():
    model = Sequential()
    model.add(LSTM(input_shape=(reshaped_train.shape[1],reshaped_train.shape[2]), output_dim=42, return_sequences=True))
    model.add(LSTM(42, return_sequences=False))
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mae', optimizer='rmsprop')
    return model


def train_model(train_x, train_y, test_x, test_y,batch_size):
    model = build_model()
    model.fit(train_x, train_y, batch_size, nb_epoch=50, validation_split=0.1,shuffle = False)
    predict = model.predict(test_x)
    predict = np.reshape(predict, (predict.size, ))
    return predict


# In[10]:


test_label = return_to_label(y_test)
mean_record=[]
for i in range(10):
    batch_size = 10*(i+1)
    accuracy=[]
    for j in range(10):
        predict=train_model(reshaped_train, y_train, reshaped_test, y_test,batch_size)
        pred_label = return_to_label(predict)
        accuracy.append(np.sum([pred_label[i]==test_label[i] for i in range(len(pred_label))])/len(pred_label))
    mean_record.append(np.mean(accuracy))


# In[21]:


accuracy=[]
for i in range(10):
    print("loop %s"%i)
    batch_size = 10*(i+1)
    predict=train_model(reshaped_train, y_train, reshaped_test, y_test,batch_size)
    pred_label = return_to_label(predict)
    accuracy.append(np.sum([pred_label[i]==test_label[i] for i in range(len(pred_label))])/len(pred_label))


# In[20]:


accuracy


# In[41]:


batch_range=np.arange(10,110,10)
plt.plot(batch_range,accuracy);
plt.xlabel('Batch_size');
plt.ylabel('Accuracy Rate')


# In[23]:



# predict=train_model(reshaped_train, y_train, reshaped_test, y_test,40)


# In[15]:


# def return_to_label(returns):
#     labels=[]
#     for y in returns:
#         tag = 0
#         if y>2:
#             tag = 1
#         elif y>0:
#             tag = 2
#         elif y>-2:
#             tag = 3
#         else:
#             tag = 4
#         labels.append(tag)
#     return labels


# In[42]:


test_label = return_to_label(y_test)
pred_label = return_to_label(predict)
np.sum([pred_label[i]==test_label[i] for i in range(len(pred_label))])/len(pred_label)


# In[17]:


# pred_label = pd.DataFrame(pred_label)
# test_label = pd.DataFrame(test_label)
# pred_3=pred_label[test_label==1]
# np.sum([pred_3[i]==1 for i in range(len(pred_3))])/len(pred_3)
# pred_3
pred_label=np.array(pred_label)
test_label=np.array(test_label)
np.sum(pred_label[test_label==4] == 4)/len(pred_label[test_label==4])


# In[43]:


import sklearn.metrics
sklearn.metrics.confusion_matrix(test_label, pred_label)


# In[31]:


import matplotlib.pyplot as plt
plt.plot(range(len(predict)),predict)
plt.plot(range(len(predict)),y_test);
plt.xlabel('Days');
plt.ylabel('Returns(%)');
plt.legend(['Predicted Returns','Actual Returns']);


# In[16]:


# predict=train_model(reshaped_train, y_train, reshaped_test, y_test,batch_size=40)
# pred_label = return_to_label(predict)
# np.sum([pred_label[i]==test_label[i] for i in range(len(pred_label))])/len(pred_label)


# In[4]:


# google=pd.read_csv('Google.csv')
# google.head()


# In[5]:


# random.seed(417)
# x = google[['CVol', 'Return', 'Spread', 'Diff']]
# y = google['Return']
# test_len = round(len(x)*0.2)
# x_test = x[0:test_len]
# x_train = x[test_len:]
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)


# In[22]:


# data1 = []
# sequence_length = 5 #Set the number of days to traceback
# for i in range(len(x_train) - sequence_length - 1):
#     data1.append(x_train[i: i + sequence_length + 1])
# reshaped_train = np.array(data1).astype('float64')

# data2 = []
# for i in range(len(x_test) - sequence_length - 1):
#     data2.append(x_test[i: i + sequence_length + 1])
# reshaped_test = np.array(data2).astype('float64')

# y_test = y[0:test_len]
# y_train = y[test_len:]
# y_train = y_train[:- sequence_length - 1]
# y_test = y_test[:- sequence_length - 1]
# reshaped_train.shape


# In[7]:


# def return_to_label(returns):
#     labels=[]
#     for y in returns:
#         tag = 0
#         if y>3:
#             tag = 1
#         elif y>0:
#             tag = 2
#         elif y>-3:
#             tag = 3
#         else:
#             tag = 4
#         labels.append(tag)
#     return labels


# In[19]:


# test_label = return_to_label(y_test)
# predict=train_model(reshaped_train, y_train, reshaped_test, y_test,40)
# pred_label = return_to_label(predict)
# np.sum([pred_label[i]==test_label[i] for i in range(len(pred_label))])/len(pred_label)


# In[23]:


# accuracy=[]
# for i in range(10):
#     print("loop %s"%i)
#     batch_size = 10*(i+1)
#     predict=train_model(reshaped_train, y_train, reshaped_test, y_test,batch_size)
#     pred_label = return_to_label(predict)
#     accuracy.append(np.sum([pred_label[i]==test_label[i] for i in range(len(pred_label))])/len(pred_label))
#     print(accuracy[i])


# In[24]:


# accuracy

