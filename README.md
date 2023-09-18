# FastKNN++ : A new KNN algrotihm based on binary search
FastKNN++ is an ensemble algorithm of FastKNN that sorts the dataset based on a randomly selected point on the training phase. While testing, it uses binary serach to find potential nearest neighbor candidates.

# List of Datasets

| Serial No. | Dataset Name                                  | Link                                      |
|------------|-----------------------------------------------|-------------------------------------------|
| 1          | Lymphography                                 | [Dataset Link](https://archive.ics.uci.edu/dataset/63/lymphography) |
| 2          | Seeds                                         | [Dataset Link](https://archive.ics.uci.edu/dataset/236/seeds) |
| 3          | Pima                                          | [Dataset Link](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| 4          | Breast Cancer                                | [Dataset Link](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) |
| 5          | Diabetes                                     | [Dataset Link](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset) |
| 6          | Fertility                                    | [Dataset Link](https://archive.ics.uci.edu/dataset/244/fertility) |
| 7          | Glass                        | [Dataset Link](https://archive.ics.uci.edu/dataset/42/glass+identification) |
| 8          | Magic                     | [Dataset Link](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope) |
| 9          | Iris                                          | [Dataset Link](https://archive.ics.uci.edu/dataset/53/iris) |
| 10         | Spambase                                     | [Dataset Link](https://archive.ics.uci.edu/dataset/94/spambase) |
| 11         | Digits                                       | [Dataset Link](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) |
| 12         | Sepsis                                       | [Dataset Link](https://www.kaggle.com/datasets/davidechicco/sepsis-minimal-ehrs-from-norway) |
| 13         | Yeast                                        | [Dataset Link](https://archive.ics.uci.edu/dataset/110/yeast) |
| 14         | ECG                                          | [Dataset Link](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) |

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

[![DOI](https://zenodo.org/badge/624813756.svg)](https://zenodo.org/badge/latestdoi/624813756)
