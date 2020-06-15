#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import keras


# In[2]:


from keras.datasets import mnist


# In[3]:


(x_train,y_train),(x_test,y_test) = mnist.load_data()


# In[6]:


x_train.shape


# In[8]:


x_test.shape


# In[11]:


plt.matshow(x_train[10])


# In[12]:


y_train[10]


# In[13]:


x_train[0]


# In[14]:


#conver value between 0 to 1
x_train = x_train/255
x_test = x_test/255


# In[15]:


x_train[0]


# In[46]:


# sequential model
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten   #Flatten convert 2D array in 1D array


# In[47]:


model = Sequential()


# In[48]:


model.add(Flatten(input_shape = [28,28]))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


# In[49]:


model.summary()


# In[50]:


model.compile(loss = "sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])


# In[51]:


model.fit(x_train, y_train, epochs = 10)


# In[52]:


model.evaluate(x_test,y_test)


# In[53]:


y_pred = model.predict(x_test)


# In[61]:


y_pred[100]


# In[60]:


plt.matshow(x_test[100])


# In[62]:


np.argmax(y_pred[100])


# In[ ]:




