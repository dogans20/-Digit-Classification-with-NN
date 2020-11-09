# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:58:19 2019

@author: dgns
"""


import cv2
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import random
from os import listdir

df_train = pd.read_csv('C:/Users/Erkam Enes/Desktop/train.csv') ## ana veri çağırıldı
df_train.info()
Y_train_orig = df_train['label'].values     ## veriden etiketleri ayır
X_train_orig = df_train.drop('label', axis=1).values  ## veriden etiketleri ayır
# Veriyi test val train için böl
X_train= X_train_orig[0:14000,:].T  
X_val = X_train_orig[14000:28000,:].T
X_test = X_train_orig[28000:42000,:].T
Y = Y_train_orig.reshape(1, Y_train_orig.shape[0]) ## matris düzenlendi

Y_train = Y[:,0:14000]
Y_val = Y[:,14000:28000]
Y_test = Y[:,28000:42000]

Y_train_onehot = np.zeros((Y_train.max()+1, Y_train.shape[1]))
Y_train_onehot[Y_train, np.arange(Y_train.shape[1])] = 1
print("Shape of Y_train_onehot:", Y_train_onehot.shape)

#df_test = pd.read_csv('C:/Users/dgns/Desktop/test.csv')
#X_test = df_test.values.T
#X_test = X_test / 255.
#X_test.shape
##############################################################################
##### fonksiyonlar

def relu(z):      
    z_relu = np.maximum(z, 0)
    
    activation_cache = (z)
    return z_relu, activation_cache

def relu_derivative(z):
    z_derivative = np.zeros(z.shape)
    z_derivative[z > 0] = 1
    
    return z_derivative

def softmax(z):
    z_exp = np.exp(z - np.max(z))
    z_softmax = z_exp / np.sum(z_exp, axis=0) 
    
    activation_cache = (z)
    return z_softmax, activation_cache

###### parametreler ayarlanıyor
def initialize_parameters(layers_dims):
    L = len(layers_dims) - 1
    parameters = {}
    caches = {}
    for l in range(1, L+1):
        parameters["W"+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * (1 / layers_dims[l-1])
        parameters["b"+str(l)] = np.zeros((layers_dims[l], 1)) ## sıfır matrisi biaslar için oluştu
    
    return parameters

#####  ileri propagasyon fonksiyonları ağırlık çarpımı ve bias toplamı hesabı
def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    
    cache = (A_prev, W, b)
    return Z, cache

### aktivasyon ileri propagasyonu hesabı
def activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'softmax':
        A, activation_cache = softmax(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache

###  ileri propagasyon modeli
def L_forward_propagate(X, parameters):
    caches = []
    L = len(parameters) // 2
    
    A_prev = X
    for l in range(1, L):
        A, cache = activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], 'relu')
    
        A_prev = A
        caches.append(cache)
        
    AL, cache = activation_forward(A_prev, parameters["W"+str(L)], parameters["b"+str(L)], 'softmax')
    caches.append(cache)
    
    return AL, caches

###### hata hesaplama
def compute_cost(Y, AL):
    m = Y.shape[1]
    cost = (1/m) * np.sum((AL - Y)*(AL - Y))
#    cost = -(1/m) * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
    return cost


##### geri propagasyon fonksiyonları
def linear_backward(dZ, linear_cache):
    (A_prev, W, b) = linear_cache
    m = A_prev.shape[1]
    
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dW, db, dA_prev

def activation_backward(Y, cache, activation, A=0, dA=0):
    (linear_cache, activation_cache) = cache
    
    (Z) = activation_cache
    if activation == 'relu':
        dZ = dA * relu_derivative(Z)
    elif activation == 'softmax':
        dZ = A - Y
    
    dW, db, dA_prev = linear_backward(dZ, linear_cache)
    
    return dW, db, dA_prev


def L_backward_propagate(X, Y, AL, parameters, caches):
    m = Y.shape[1]
    L = len(parameters) // 2
    grads = {}
    
    cache = caches[L-1]
    dW, db, dA_prev = activation_backward(Y, cache, 'softmax', A = AL)
    grads["dW"+str(L)] = dW
    grads["db"+str(L)] = db
    
    dA = dA_prev
    for l in range(L-1, 0, -1):
        cache = caches[l-1]
        
        dW, db, dA_prev = activation_backward(Y, cache, 'relu', dA = dA)
        
        grads["dW"+str(l)] = dW
        grads["db"+str(l)] = db
        dA = dA_prev
    
    return grads
######## parametreler güncelleniyor
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    params = {}
    
    for l in range(1, L+1):
        params["W"+str(l)] = parameters["W"+str(l)] - learning_rate * grads["dW"+str(l)]
        params["b"+str(l)] = parameters["b"+str(l)] - learning_rate * grads["db"+str(l)]
    
    return params
######### tahmin fonksiyonu
def predict(X, parameters):
    A2, _ = L_forward_propagate(X, parameters)
    
    Y_predict = np.argmax(A2, axis=0)
    Y_predict = Y_predict.reshape(1, Y_predict.shape[0])
    
    return Y_predict
######## yapau sinirağı modeli
def model(X, Y, epochs=75, mini_batch_size=48, learning_rate=0.011):
    layers_dims = [X.shape[0], 64, 32, 16, Y.shape[0]] ### katman boyutları 
    costs = []
    beta = 0.9
    
    parameters = initialize_parameters(layers_dims) ### parametreler ayarlanıyor
    
    for i in range(0, epochs):
        print("Epoch", i+1, ":")
        
        m = X.shape[1]
        
        permutation = np.random.permutation(m)
        X_shuffle = X[:, permutation]
        Y_shuffle = Y[:, permutation]
        num_mini_batch = m // mini_batch_size ## veri bölünür
        
        avg_cost = 0
        for  j in range(0, num_mini_batch):
            k = np.arange(0,28000,dtype=np.float)
            k[:] = 0.00
            X_mini_batch = X_shuffle[:, (j*mini_batch_size) : (j+1)*mini_batch_size]
            Y_mini_batch = Y_shuffle[:, (j*mini_batch_size) : (j+1)*mini_batch_size]

            
            AL, caches = L_forward_propagate(X_mini_batch, parameters)
            cost = compute_cost(Y_mini_batch, AL)
            grads = L_backward_propagate(X_mini_batch, Y_mini_batch, AL, parameters, caches)                
            parameters = update_parameters(parameters, grads, learning_rate)
            if cost <= 0.005:
                break
            
            if j % 5 == 0:
                op = "Cost(%5d/%5d): %3.6f" % ((j+1) * mini_batch_size, m, cost)
                print(op, end='\r')
            
            avg_cost = beta * avg_cost + (1 - beta) * cost
        
        costs.append(avg_cost)
        
        if m % mini_batch_size != 0:
            X_mini_batch = X_shuffle[:, (num_mini_batch*mini_batch_size):]
            Y_mini_batch = Y_shuffle[:, (num_mini_batch*mini_batch_size):]

            
            AL, caches = L_forward_propagate(X_mini_batch, parameters)
            cost = compute_cost(Y_mini_batch, AL)
            grads = L_backward_propagate(X_mini_batch, Y_mini_batch, AL, parameters, caches)
            parameters = update_parameters(parameters, grads, learning_rate)
            
            op = "Cost(%5d/%5d): %3.6f" % (m, m, cost)
            print(op, end='\r')
            if cost <= 0.005:
                break
            avg_cost = beta * avg_cost + (1 - beta) * cost
            costs.append(avg_cost)
        
        print()
    return parameters, costs , caches

parameters, costs ,caches = model(X_train, Y_train_onehot, epochs=75, mini_batch_size=48)



Y_train_predict = predict(X_train, parameters)
Y_val_predict = predict(X_val,parameters)
Y_test_predict = predict(X_test, parameters)
Y_test_predict.shape
def accuracy(Y2, Y_pred):
    tp = np.sum((Y2 == Y_pred).astype(int))
    
    return tp / Y2.shape[1]

acc = accuracy(Y_train, Y_train_predict)
acc2 = accuracy(Y_val, Y_val_predict)
acc3 = accuracy(Y_test, Y_test_predict)

print("Accuracy on training data:", acc)
print("Accuracy on val data:", acc2)
print("Accuracy on test data:", acc3)




### testten resim çağırma
dn = random.randint(0,14000)
a = X_test[:,dn]
a = a.reshape(28,28)
plt.imshow(a)
print("Y_test_predict:", Y_test_predict[0,dn])

fig = plt.figure()
dn1 =random.randint(0,14000)
p11 = X_test[:,dn1]
p11 = p11.reshape(28,28)
ax11 =plt.subplot(221)
plt.imshow(p11)
ax11.title.set_text(Y_test_predict[0,dn1])

dn2 =random.randint(0,14000)
p21 = X_test[:,dn2]
p21 = p21.reshape(28,28)
ax21 =plt.subplot(222)
plt.imshow(p21)
ax21.title.set_text(Y_test_predict[0,dn2])

dn31 =random.randint(0,14000)
p31 = X_test[:,dn31]
p31 = p31.reshape(28,28)
ax31 =plt.subplot(223)
plt.imshow(p31)
ax31.title.set_text(Y_test_predict[0,dn31])

dn4 =random.randint(0,14000)
p41 = X_test[:,dn4]
p41 = p41.reshape(28,28)
ax41 =plt.subplot(224)
plt.imshow(p41)
ax41.title.set_text(Y_test_predict[0,dn4])


############# kendi test verimizi düzenleyip ysa ya veririz.
X_test2 = X_test[:,0]
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2126, 0.7152, 0.0722])
path = "C:/Users/Erkam Enes/Desktop/7"
images=listdir(path)
img55 =[]
for i in range (len(images)):
    img55 = cv2.imread("C:/Users/Erkam Enes/Desktop/7/"+images[i])
    dim3 = (28, 28)
    resized = cv2.resize(img55, dim3, interpolation = cv2.INTER_AREA)
    graynrm = rgb2gray(resized)
##    gray = rgb2gray(resized)    
    gray= abs(graynrm-255)
    graynrm= (gray- np.min(gray)) / (np.max(gray)-np.min(gray))
    for l in range (0,28):
        for b in range (0,28):
            if graynrm[l,b]< 0.2 : graynrm[l,b] = 0
            else : graynrm[l,b] = 1                    
    result=np.array(graynrm).flatten()
    X_test2=np.column_stack((X_test2, result))
    
Y_test_predict2 = predict(X_test2,parameters)


fig = plt.figure()
aa1= random.randint(0,len(images))
p1 = X_test2[:,aa1]
p1 = p1.reshape(28,28)
ax1 =plt.subplot(221)
plt.imshow(p1)
ax1.title.set_text(Y_test_predict2[0,aa1])

aa2= random.randint(0,len(images))
p2 = X_test2[:,aa2]
p2 = p2.reshape(28,28)
ax2 =plt.subplot(222)
plt.imshow(p2)
ax2.title.set_text(Y_test_predict2[0,aa2])

aa3=random.randint(0,len(images))
p3 = X_test2[:,aa3]
p3 = p3.reshape(28,28)
ax3 = plt.subplot(223)
ax3.title.set_text(Y_test_predict2[0,aa3])
plt.imshow(p3)

aa4=random.randint(0,len(images))
p4 = X_test2[:,aa4]
p4 = p4.reshape(28,28)
ax4 = plt.subplot(224)
ax4.title.set_text(Y_test_predict2[0,aa4])
plt.imshow(p4)
#plt.subplot_tool()
plt.show()

plt.plot(range(0, len(costs)*5, 5), costs)
