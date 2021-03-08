#!/usr/bin/env python
# coding: utf-8

# In[1]:


a = lambda x, y : x + y


# In[2]:


a(1, 2)


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import sin, cos, exp


# In[10]:


ex = np.array([1,0])
ey = np.array([0,1])

plt.arrow(0,0,ex[0],ex[1],head_width=0.2,color='b')
plt.arrow(0,0,ey[0],ey[1],head_width=0.2,color='r')
plt.xlim(-2,2)
plt.ylim(-2,2)


# In[16]:


# alpha 라는 매개변수를 바꾸고 단위 벡터를 사용하여 일직선을 생성

u = np.array([1,1])
for alpha in np.arange(-1,1,0.2):     
    A = np.array([[1,alpha],
                  [0,alpha]])
    v = np.dot(A,u)
    
    plt.arrow(0,0,v[0],v[1],head_width=0.1,color='r')

plt.xlim(-2,2)
plt.ylim(-2,2)
plt.show


# In[13]:


np.dot(A,ex)


# In[14]:


np.dot(A,ey)


# In[19]:


# A*x = b 를 성립하는 x를 찾으시오.
# 가) 역행렬 linalg.inv
# 나)        linalg.solve
# 다) 소거법 사용

A = np.array([[1,2],
              [0,3]])
Ainv = np.linalg.inv(A)
b = np.array([5,4])


# In[20]:


Ainv


# In[21]:


np.dot(Ainv,b)


# In[23]:


xsol = np.linalg.solve(A,b)
print(xsol)


# 행인 A와 B가 1차원 선상에서 등가속운동을 하고있다. A와 B는 서로 100m 떨어져 있으며 속력 v0 = 0m/s로 출발한다. A는 가속도 a1, B는 가속도 a2로 걷는다고 가정하자. 두 사람이 반대방향으로 
# 걸을 때는 두 사람이 만날 때까지 30초가 걸리며, 두 사람이 같은 방향으로 걸을 때는 90초가 걸린다. a1과 a2를 구하는 script를 만들어라.

# In[24]:


t1 = 30 
t2 = 90
A = np.array([[t1**2/2, t1**2/2],
             [t2**2/2, -t2**2/2]])
b = np.array([100,100])

x = np.linalg.solve(A, b)
print(x)


# In[25]:


# A와 B행인이 등속 운동을 하고 있다. 서로의 초기 거리는 100미터이다.
# 서로 반대 방향으로 걸을 때 걸리는 시간은 5초
# 서로 같은 방향으로 걸을 때 걸리는 시간은 15초
# A, B 속도를 찾으시오.

t1 = 5 
t2 = 15
A = np.array([[t1, t1],
             [t2, -t2]])
b = np.array([100,100])

x = np.linalg.solve(A, b)
print(x)


# In[ ]:




