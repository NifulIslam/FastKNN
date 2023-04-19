# FastKNN++ : A new KNN algrotihm based on binary search
FastKNN++ is an ensemble algorithm of FastKNN that sorts the dataset based on a randomly selected point on the training phase. While testing, it uses binary serach to find potential nearest neighbor candidates.
# How to use: 
- Before using please make sure you have the following libraries installed
    - Numpy
    - Pandas
    - Sklearn
- Clone the github repository using the code below
```  
! git clone git@github.com:NifulIslam/FastKNN.git 
```
- Import FastKNN++ to use 
``` python
from FastKNN.implementation.FastKNNplusplus import FastKNNpp
```
- Pass nescessary parametres while initializing the model
``` python
fastKNNpp=FastKNNpp(X_train,y_train,k=3,beta=10, no_of_models=5)
```
- Make prediction using the predict method
``` python
y_pred=fastKNNpp.predict(X_test)
```
A detailed example is given on the *FastKNN++_classification.ipynb* file
