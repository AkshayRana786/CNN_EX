#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import keras


# In[4]:


from keras.datasets import fashion_mnist


# In[5]:


(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()


# In[11]:


plt.matshow(x_train[1])


# In[16]:


y_train[1]


# In[17]:


x_train[0]


# In[18]:


#conver value between 0 to 1
x_train = x_train/255
x_test = x_test/255


# In[19]:


x_train[0]


# In[46]:


# sequential model
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten   #Flatten convert 2D array in 1D array


# In[47]:


model = Sequential()


# In[48]:


model.add(Flatten(input_shape = [28,28]))
model.add(Dense(200, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


# In[49]:


model.summary()


# In[50]:


model.compile(loss = "sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])


# In[51]:


model.fit(x_train, y_train, epochs = 5)


# In[42]:


plt.matshow(x_test[1])


# In[39]:


y_pred = model.predict(x_test)


# In[44]:


y_pred[1] #model.add(Dense(10, activation = 'softmax' using softmax it destribute 10 prediction


# In[43]:


np.argmax(y_pred[1])


# In[52]:


model.evaluate(x_test,y_test)


# In[ ]:




