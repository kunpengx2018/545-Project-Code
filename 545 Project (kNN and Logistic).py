#!/usr/bin/env python
# coding: utf-8

# This document will sum up all the code we used in our 545 project.

# * Data import and modification

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[138]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


# In[91]:


# import data from .csv
raw_data = pd.read_csv('SP500.csv').sort_values(by = 'Date', ascending = True)
raw_data['Date'] = pd.to_datetime(raw_data['Date'])
raw_data['spread'] = (raw_data['High'] - raw_data['Low'])/raw_data['Low']
raw_data['opentoclose'] = (raw_data['Open'] - raw_data['Close'].shift(1))/raw_data['Close'].shift(1)
raw_data['return'] = raw_data['Close'].pct_change()
raw_data['ma5'] = raw_data['return'].rolling(window = 5).mean()
raw_data['ma10'] = raw_data['return'].rolling(window = 10).mean()
raw_data['std5'] = raw_data['return'].rolling(window = 5).std()

raw_data.tail(20)


# In[34]:


raw_data.columns


# In[35]:


variables = ['Volume', 'spread','opentoclose', 'ma5', 'ma10', 'std5']


# In[67]:


# define a function that can sign labels to different return levels
def get_level(data1, name, number):
    data = pd.DataFrame(data1[name])
#     print(data.index)
    if number == 2:
        for i in data.index:
            if data.loc[i, name]*100 >= 0:
                data.loc[i, 'level'] = 1
            else:
                data.loc[i, 'level'] = -1
        return data
    elif number == 4:
        for i in data.index:
            if data.loc[i, name]*100 < -2:
                data.loc[i, 'level'] = -2
            elif data.loc[i, name]*100 < 0:
                data.loc[i, 'level'] = -1
            elif data.loc[i, name]*100 < 2:
                data.loc[i, 'level'] = 1
            else:
                data.loc[i, 'level'] = 2
        return data
    else:
        raise ValueError('Input number should be 2 or 4!')


# In[37]:


# get_level(raw_data, 'return', 4)


# In[63]:


# define a function that aggregate previous dates data
def get_previous_data(data1, number, variables):
    data = data1.copy()
    new_variables = variables.copy()
    for variable in variables:
        for i in range(1, number):
            new_name = variable + '_sft_' + str(i)
            new_variables.append(new_name)
            data[new_name] = data[variable].shift(i)
            
    return data,new_variables


# now we do KNN classification:

# 1) Naive KNN

# In[169]:


spliting_date = pd.to_datetime('20190101')
knn_train_accuracy = []
knn_test_accuracy = []

for n in range(1,11):
    knn_data, new_variables = get_previous_data(raw_data, n, variables)
    knn_data['response'] = get_level(knn_data, 'return', 4)['level'].shift(-1)
    knn_data = knn_data.dropna(axis = 0, how = 'any').reset_index(drop = True)
    
    train = knn_data[knn_data['Date'] < spliting_date]
    train_x = train[new_variables]
    train_y = train['response']
    test = knn_data[knn_data['Date'] >= spliting_date]
    test_x = test[new_variables]
    test_y = test['response']
    
#     print(-1 in list(test_y))
    
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    
    knn_train_ks = []
    knn_test_ks = []
    
    for k in range(40, 71):
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(train_x, train_y)

        knn_train_pred = classifier.predict(train_x)
        knn_test_pred = classifier.predict(test_x)
        
        knn_train_ks.append(np.mean(knn_train_pred == train_y))
        knn_test_ks.append(np.mean(knn_test_pred == test_y))
    
    knn_train_accuracy.append(knn_train_ks)
    knn_test_accuracy.append(knn_test_ks)
    
#     print(n)
    
knn_train_accuracy = np.array(knn_train_accuracy).T
knn_test_accuracy = np.array(knn_test_accuracy).T


# In[170]:


plt.plot([k for k in range(40,71)], knn_train_accuracy)
plt.legend(['n = ' + str(n) for n in range(1,11)])
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('KNN training accuracy for different n and k')
plt.savefig('/Users/David/Desktop/KNN_train_accuracy.png')
plt.show()


# In[171]:


plt.plot([k for k in range(40,71)], knn_test_accuracy)
plt.legend(['n = ' + str(n) for n in range(1,11)])
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('KNN testing accuracy for different n and k')
plt.savefig('/Users/David/Desktop/KNN_test_accuracy.png')
plt.show()


# In[172]:


knn_data, new_variables = get_previous_data(raw_data, 5, variables)
knn_data['response'] = get_level(knn_data, 'return', 4)['level'].shift(-1)
knn_data = knn_data.dropna(axis = 0, how = 'any').reset_index(drop = True)

train = knn_data[knn_data['Date'] < spliting_date]
train_x = train[new_variables]
train_y = train['response']
test = knn_data[knn_data['Date'] >= spliting_date]
test_x = test[new_variables]
test_y = test['response']

#     print(-1 in list(test_y))

scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)


classifier = KNeighborsClassifier(n_neighbors=55)
classifier.fit(train_x, train_y)

# knn_train_pred = classifier.predict(train_x)
knn_test_pred = classifier.predict(test_x)
confusion_matrix(test_y, knn_test_pred)


# 2) CV KNN

# We run Cross Validation for optimal k, even though from the above figure we can see that this might not be of great help:
# 
# We choose $n = 5$, i.e., use 5 days of data as input.

# In[122]:


spliting_date = pd.to_datetime('20190101')
knn_cv_accuracy = []


knn_data, new_variables = get_previous_data(raw_data, 5, variables)
knn_data['response'] = get_level(knn_data, 'return', 4)['level'].shift(-1)
knn_data = knn_data.dropna(axis = 0, how = 'any').reset_index(drop = True)

train = knn_data[knn_data['Date'] < spliting_date]
train_x = train[new_variables]
train_y = train['response']
test = knn_data[knn_data['Date'] >= spliting_date]
test_x = test[new_variables]
test_y = test['response']

#     print(-1 in list(test_y))

scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)


for k in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(classifier,train_x, train_y, cv=10, scoring='accuracy')
    
    knn_cv_accuracy.append(scores.mean())
#     print(k)


# In[168]:


plt.plot([k for k in range(1,101)], knn_cv_accuracy)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('CV accuracy under n = 5')
plt.savefig('/Users/David/Desktop/KNN_CV_accuracy.png')
plt.show()


# In[ ]:





# We can see that vanilla KNN method can not effectively distinguish between different patterns, thus we try polynomial embedding in KNN to see if we can improve our result.

# In[173]:


spliting_date = pd.to_datetime('20190101')
knn_train_accuracy = []
knn_test_accuracy = []
poly = PolynomialFeatures(degree=2)

for n in range(1,6):
    knn_data, new_variables = get_previous_data(raw_data, n, variables)
    knn_data['response'] = get_level(knn_data, 'return', 4)['level'].shift(-1)
    knn_data = knn_data.dropna(axis = 0, how = 'any').reset_index(drop = True)
    
    # alse include poly of variables
    train = knn_data[knn_data['Date'] < spliting_date]
    train_x = poly.fit_transform(train[new_variables])
    train_y = train['response']
    test = knn_data[knn_data['Date'] >= spliting_date]
    test_x = poly.fit_transform(test[new_variables])
    test_y = test['response']
    
#     print(-1 in list(test_y))
    
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    
    knn_train_ks = []
    knn_test_ks = []
    
    for k in range(40, 71):
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(train_x, train_y)

        knn_train_pred = classifier.predict(train_x)
        knn_test_pred = classifier.predict(test_x)
        
        knn_train_ks.append(np.mean(knn_train_pred == train_y))
        knn_test_ks.append(np.mean(knn_test_pred == test_y))
    
    knn_train_accuracy.append(knn_train_ks)
    knn_test_accuracy.append(knn_test_ks)
    
    print(n)
    
knn_train_accuracy = np.array(knn_train_accuracy).T
knn_test_accuracy = np.array(knn_test_accuracy).T


# In[174]:


plt.plot([k for k in range(40,71)], knn_train_accuracy)
plt.legend(['n = ' + str(n) for n in range(1,11)])
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('KNN training accuracy for different n and k')
plt.savefig('/Users/David/Desktop/KNNp_train_accuracy.png')
plt.show()


# In[175]:


plt.plot([k for k in range(40,71)], knn_test_accuracy)
plt.legend(['n = ' + str(n) for n in range(1,11)])
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('KNN testing accuracy for different n and k')
plt.savefig('/Users/David/Desktop/KNNp_test_accuracy.png')
plt.show()


# In[135]:


classifier = KNeighborsClassifier(n_neighbors=35)
classifier.fit(train_x, train_y)

# knn_train_pred = classifier.predict(train_x)
knn_test_pred = classifier.predict(test_x)
confusion_matrix(test_y, knn_test_pred)


# We compare the above figures and can see that: even though ploynomial embedding (degree = 2) can slightly increase accuracy for testing data, the improvement is very limited. Thus we decide not to stick with KNN model in the case.

# * Logistic Regression

# We then try logistic regression:

# In[177]:


spliting_date = pd.to_datetime('20190101')
logistic_train_accuracy = []
logistic_test_accuracy = []

for n in range(1,11):
    log_data, new_variables = get_previous_data(raw_data, n, variables)
    log_data['response'] = get_level(log_data, 'return', 4)['level'].shift(-1)
    log_data = log_data.dropna(axis = 0, how = 'any').reset_index(drop = True)
    
    train = log_data[log_data['Date'] < spliting_date]
    train_x = train[new_variables]
    train_y = train['response']
    test = log_data[log_data['Date'] >= spliting_date]
    test_x = test[new_variables]
    test_y = test['response']
    
#     print(-1 in list(test_y))
    log_model = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial')
    log_model.fit(train_x, train_y)
    
    log_train_pred = log_model.predict(train_x)
    log_test_pred = log_model.predict(test_x)

    logistic_train_accuracy.append(np.mean(log_train_pred == train_y))
    logistic_test_accuracy.append(np.mean(log_test_pred == test_y))
    print(n)


# In[178]:


plt.plot([n for n in range(1,11)], logistic_train_accuracy)
plt.xlabel('n')
plt.ylabel('accuracy')
plt.title('Training accuracy under logistic regression')
plt.savefig('/Users/David/Desktop/log_train_accuracy.png')
plt.show()


# In[179]:


plt.plot([n for n in range(1,11)], logistic_test_accuracy)
plt.xlabel('n')
plt.ylabel('accuracy')
plt.title('Testing accuracy under logistic regression')
plt.savefig('/Users/David/Desktop/log_test_accuracy.png')
plt.show()


# In[180]:


np.max(logistic_test_accuracy)


# In[181]:


np.argmax(logistic_test_accuracy) + 1


# We can see that even the highest only has an accuracy of $58.4\%$

# In[ ]:





# In[153]:


log_data, new_variables = get_previous_data(raw_data, 5, variables)
log_data['response'] = get_level(log_data, 'return', 4)['level'].shift(-1)
log_data = log_data.dropna(axis = 0, how = 'any').reset_index(drop = True)

train = log_data[log_data['Date'] < spliting_date]
train_x = train[new_variables]
train_y = train['response']
test = log_data[log_data['Date'] >= spliting_date]
test_x = test[new_variables]
test_y = test['response']

#     print(-1 in list(test_y))
log_model = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial')
log_model.fit(train_x, train_y)

log_test_pred = log_model.predict(test_x)

confusion_matrix(test_y, log_test_pred)


# Also, it does not handle extreme case properly.

# We now try polynomial implementatin in our logistic regression.

# In[158]:


spliting_date = pd.to_datetime('20190101')
logistic_train_accuracy = []
logistic_test_accuracy = []

for n in range(1,6):
    log_data, new_variables = get_previous_data(raw_data, n, variables)
    log_data['response'] = get_level(log_data, 'return', 4)['level'].shift(-1)
    log_data = log_data.dropna(axis = 0, how = 'any').reset_index(drop = True)
    
    train = log_data[log_data['Date'] < spliting_date]
    train_x = train[new_variables]
    train_y = train['response']
    test = log_data[log_data['Date'] >= spliting_date]
    test_x = test[new_variables]
    test_y = test['response']
    
    logistic_train_p = []
    logistic_test_p = []
    for p in range(1,4):
        poly = PolynomialFeatures(degree=p)
        train_x_p = poly.fit_transform(train_x)
        test_x_p = poly.fit_transform(test_x)
        
        log_model = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial', penalty = 'none')
        log_model.fit(train_x_p, train_y)

        log_train_pred = log_model.predict(train_x_p)
        log_test_pred = log_model.predict(test_x_p)

        logistic_train_p.append(np.mean(log_train_pred == train_y))
        logistic_test_p.append(np.mean(log_test_pred == test_y))
    
    
    logistic_train_accuracy.append(logistic_train_p)
    logistic_test_accuracy.append(logistic_test_p)
    print(n)
    
logistic_train_accuracy = np.array(logistic_train_accuracy)
logistic_test_accuracy = np.array(logistic_test_accuracy)


# In[161]:


plt.plot([n for n in range(1,6)], logistic_train_accuracy)
plt.legend(['degree = ' + str(i) for i in range(1, 4)])
plt.xlabel('n')
plt.ylabel('accuracy')
plt.title('training accuracy under logistic regression with poly embedding')
plt.show()


# In[162]:


plt.plot([n for n in range(1,6)], logistic_test_accuracy)
plt.legend(['degree = ' + str(i) for i in range(1, 4)])
plt.xlabel('n')
plt.ylabel('accuracy')
plt.title('testing accuracy under logistic regression with poly embedding')
plt.show()


# Logistic regression has a convergence problem in our case, and even if we take the finitely iterated results, we have a poor classification accuracy.

# In[ ]:




