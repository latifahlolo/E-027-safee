
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import keras
import tensorflow
import glob
import csv
import os

Images=[]
image_pixles = []
barcodes=[]


class MyImage():
    name=None
    nationality=None
    idnumber=None
    PID=None
    HajjAgency=None
    Image_url=None
    Barcode_url=None
    ImageID=None
    ImagePixles=[]
    BarcodePixles=[]



def LoadData():
    with open('C:/Users/t430/Desktop/Dataset/Data.csv', 'r') as csvfile: 
        spamreader = csv.reader(csvfile, delimiter=',')
        next(spamreader)
        for row in spamreader:
            Myimage=MyImage()
            Myimage.name=row[0]
            Myimage.nationality=row[1]
            Myimage.idnumber=row[2]
            Myimage.PID=row[3]
            Myimage.HajjAgency=row[4]
            Myimage.Image_url=row[5]+".jpg"
            Myimage.Barcode_url=row[6]+".png"
            imagepath=Myimage.Image_url
            barcodepath=Myimage.Barcode_url
            im1=Image.open(imagepath)
            im2=Image.open(barcodepath)
            im1 = im1.resize((128, 128))
            im2 = im2.resize((128, 128))
            Myimage.ImagePixles=(list(im1.getdata()))
            Myimage.BarcodePixles=(list(im2.getdata()))
            tmp=imagepath.split("/")[7]
            tmp2=tmp.split(".")
            Myimage.ImageID=tmp2[0]
            Images.append(Myimage)
            
           

def PrintHajjInfo(i):   
#    Image.open(Images[i].Image_url).show()
#    Image.open(Images[i].Barcode_url).show()
    print(Images[i].name)
    print(Images[i].idnumber)
    print(Images[i].PID)
    print(Images[i].ImageID)
    print(Images[i].nationality)
    print(Images[i].HajjAgency)





    
names=[]
Images=[]
LoadData()
PrintHajjInfo(3)            


# In[2]:


for i in Images:
    image_pixles.append(i.ImagePixles)

for i in Images:
    barcodes.append(i.BarcodePixles)
    
names=[]
Images=[]
LoadData()
PrintHajjInfo(3)            
print(len(image_pixles))
print(len(barcodes))


# In[3]:


DatasetPath  = []
for i in os.listdir("C:/Users/t430/Desktop/Dataset/images/Training/"):
    DatasetPath.append(os.path.join("C:/Users/t430/Desktop/Dataset/images/Training/", i))
    
imageLabels = []

for i in DatasetPath:
    labelRead = int(os.path.split(i)[1].split(".")[0].replace("subject", "")) - 1
    imageLabels.append(labelRead)
    
    
DatasetPath  = []
barcodeLabels = []

for i in os.listdir("C:/Users/t430/Desktop/Dataset/barcodes"):
    DatasetPath.append(os.path.join("C:/Users/t430/Desktop/Dataset/barcodes", i))
    
for i in DatasetPath:  
    labelRead = int(os.path.split(i)[1].split(".")[0].replace("subject", "")) - 1
    barcodeLabels.append(labelRead)


# In[3]:


from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

print(len(image_pixles))    

#image_pixles=image_pixles[:5]
image_pixles=image_pixles+image_pixles+image_pixles+image_pixles


    
#imageLabels=imageLabels[:5]
imageLabels=imageLabels+imageLabels+imageLabels+imageLabels
    
#barcodes=barcodes[:5]    
barcodes=barcodes+barcodes+barcodes+barcodes

#barcodeLabels=barcodeLabels[:5]    
barcodeLabels=barcodeLabels+barcodeLabels+barcodeLabels+barcodeLabels

print(len(barcodeLabels))
print(len(barcodes))

    
X_train, X_test, y_train, y_test = train_test_split(image_pixles, imageLabels, test_size=0.25)


X_train2, X_test2, y_train2, y_test2 = train_test_split(barcodes, barcodeLabels, test_size=0.25)


X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

X_train2=np.array(X_train)
X_test2=np.array(X_test)
y_train2=np.array(y_train)
y_test2=np.array(y_test)


print(X_train.shape)
n_samples, h, w = X_train.shape

X_train=X_train.reshape((60, 16384*3))

# introspect the images arrays to find the shapes (for plotting)
X_test=X_test.reshape(20, 16384*3)

print(X_train.shape)


print(X_train2.shape)
n_samples2, h2, w2 = X_train2.shape

X_train2=X_train2.reshape((60, 16384*3))

# introspect the images arrays to find the shapes (for plotting)
X_test2=X_test2.reshape(20, 16384*3)

print(X_train.shape)


# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 15

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))


n_components = 15

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train2.shape[0]))
t0 = time()
pca2 = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train2)
print("done in %0.3fs" % (time() - t0))




eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)



eigenfaces2= pca2.components_.reshape((n_components, h2, w2))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca2 = pca2.transform(X_train2)
X_test_pca2 = pca2.transform(X_test2)
print("done in %0.3fs" % (time() - t0))

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf2 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf2 = clf2.fit(X_train2, y_train2)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf2.best_estimator_)


# In[4]:


print("Fitting the classifier to the training set")


t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)



clf2 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf2 = clf2.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf2.best_estimator_)


# In[8]:


print("Predicting people's names on the test set")
t0 = time()



y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred))



print("Predicting people's names on the test set")
t0 = time()
y_pred2 = clf2.predict(X_test_pca2)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test2, y_pred2))


# In[10]:


from skimage import io
 
DatasetPath  = []
for i in os.listdir("C:/Users/t430/Desktop/testing/"):
    DatasetPath.append(os.path.join("C:/Users/t430/Desktop/testing/", i))
    
imageLabels = []
imageData=[]
X_scenario1=[]
tmp=[]
for i in DatasetPath:
    labelRead = int(os.path.split(i)[1].split(".")[0].replace("subject", "")) - 1
    imageLabels.append(labelRead)       
    m1=Image.open(i) 
    im1 = m1.resize((128, 128))
    im1=list(im1.getdata())
    tmp.append(im1)
    im1=np.array(im1)
    X_train_pca = pca2.transform(X_scenario1.reshape(1,-1))
  

print("Predicting people's names on the test set")
t0 = time()
y_pred2 = clf2.predict(X_test_pca2)
print("done in %0.3fs" % (time() - t0))

    

