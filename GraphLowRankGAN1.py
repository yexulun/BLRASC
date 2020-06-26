from __future__ import division
from __future__ import print_function

import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from sklearn.manifold import spectral_embedding
from sklearn.decomposition import PCA
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from bayesianLowrankModel import *
from sklearn import metrics
from gan_7 import GAN
from tensorflow.examples.tutorials.mnist import input_data

import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_alg
from scipy.sparse import csgraph

import scipy.io as sio
from numpy import random as nr

from lr_init import *


K=50
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


# X=images
dataInput=sio.loadmat('D:\ye\lowRankSpectralClustering\datasets\ministFUll.mat')
batch=700;
batchSize=100;

X=dataInput['Y']



y=dataInput['z']

X=X[0:batchSize*batch,:]
y=y[0:batchSize*batch,:]


y=y
Nz,Dz=X.shape
s=(np.ones((Nz,1)))*0.2


#10,60
Knum=5
W = kneighbors_graph(X,Knum, mode='distance', include_self=True)
hidden_size=30

maps = spectral_embedding(W, n_components=hidden_size)
######################################################################
W=W
W=sparse.csr_matrix(W)

W1=W.toarray()

W=csr_matrix.toarray(W)
params,gammasC = lr_init(maps,K,K)
print(maps.dtype)
model = GAN(hidden_size, batchSize, 1e-1,maps)
numIter=0
loss_value=0
training_loss=0
training_loss1=0
while numIter<1:
    gammasC, params, P = bayesianLowrankModel(maps, params, gammasC, K, K, W)
    for i in range(batch):
        images=X[batchSize*i:batchSize*(i+1),:]/255
        maps1=maps[batchSize*i:batchSize*(i+1),:]

        R_loss, loss_value, loss_value1 = model.update_params1(images, images, maps1)
        loss_value, loss_value1, dtemp1, dtemp2 = model.update_params(images, images, maps1)
        model.update_params2(images, images, maps1)
        training_loss += loss_value
        training_loss1 += loss_value1
    training_loss = abs(training_loss) / batch
    training_loss1 = abs(training_loss1) / batch
    model.generate_and_save_images(batchSize, "", maps1)
    print(training_loss, training_loss1, P)
    training_loss = 0
    numIter = numIter + 1;
    # print(numIter)

posGaussian=gammasC



temp=np.max(posGaussian,axis=1)
temp.shape=(Nz,1)
storeIndex=np.zeros((2,Nz))
for i in range(Nz):
    index1=np.where(temp[i]==posGaussian[i,:])
    storeIndex[1,i]=index1[0][0]
tempA=np.array(posGaussian)
print(tempA.shape)
index1=storeIndex.tolist()

newData=maps
fig=plt.figure(3)
ax1=fig.add_subplot(111,projection='3d')
# colorStore='rgbyck'
marker=['.',',','o','v','^','<','>','1','2','3','4','8','s','p','*']
colorStore = ['r','y','g','b','r','y','g','b','r']
pca=PCA(n_components=30,whiten=True)
X=pca.fit_transform(X)
for i in range(newData.shape[0]):
    cho=0
    cho=int((index1[1][i]+1)%8)
    #plt.scatter(data[i,0],data[i,1],color=colorStore[cho])
    ax1.scatter(X[i,0],X[i,1],X[i,2],color=colorStore[cho],marker=marker[cho])
plt.show()

labelP=np.array(index1[1])+1

srcLabel=y
preLabel=index1
preLabel=np.sort(index1[1])
srcLabel=(np.sort(srcLabel).T)[0]


print('NMI:',metrics.adjusted_mutual_info_score(preLabel, srcLabel))
print('The estimated cluster number:',(np.unique(preLabel).shape)[0]);