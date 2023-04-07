import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from FastKNN import FastKNN
from FastKNNplusplus import FastKNNpp
import warnings


path=''# something
data=pd.read_csv(path)
data=data.sample(frac=1)
class_="" # y


warnings.filterwarnings("ignore")
k=3
n_fold=10
kf=KFold(n_splits=n_fold)
X=data.drop(class_,axis=1)
y=data[class_] 
pos=1
avg='macro'
actual_knn, fast_knn_pp, fast_knn, random_forest, adaboost=0,0,0,0,0
actual_knn_f1, fast_knn_pp_f1, fast_knn_f1, random_forest_f1, adaboost_f1=0,0,0,0,0
actual_knn_preci, fast_knn_pp_preci, fast_knn_preci, random_forest_preci, adaboost_preci=0,0,0,0,0
actual_knn_recall, fast_knn_pp_recall, fast_knn_recall, random_forest_recall, adaboost_recall=0,0,0,0,0
actual_knn_mcc, fast_knn_pp_mcc, fast_knn_mcc, random_forest_mcc, adaboost_mcc=0,0,0,0,0



for train,test in kf.split(data):
    X_train, y_train,X_test,y_test=X.iloc[train],y.iloc[train],X.iloc[test],y.iloc[test]
    model = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    model.fit(X_train,y_train)

    predicted= model.predict(X_test) 
    score = accuracy_score(y_test, predicted)  
    actual_knn+=score
    score = f1_score(y_test, predicted,pos_label=pos,average=avg)  
    actual_knn_f1+=score
    score = precision_score(y_test, predicted,pos_label=pos,average=avg)  
    actual_knn_preci+=score
    score = recall_score(y_test, predicted,pos_label=pos,average=avg)  
    actual_knn_recall+=score
    score = matthews_corrcoef(y_test, predicted) 
    actual_knn_mcc+=score
    

    fastKnnpp=FastKNNpp(X_train,y_train,k,10,13)
    predicted=fastKnnpp.predict(X_test)
    score = accuracy_score(y_test, predicted)  
    fast_knn_pp+=score
    score = f1_score(y_test, predicted,pos_label=pos,average=avg)  
    fast_knn_pp_f1+=score
    score = precision_score(y_test, predicted,pos_label=pos,average=avg)  
    fast_knn_pp_preci+=score
    score = recall_score(y_test, predicted,pos_label=pos,average=avg)  
    fast_knn_pp_recall+=score
    score = matthews_corrcoef(y_test, predicted) 
    fast_knn_pp_mcc+=score
    
    fastKnn=FastKNN(X_train,y_train,k)
    predicted=fastKnn.predict(X_test)
    score = accuracy_score(y_test, predicted)  
    fast_knn+=score
    score = f1_score(y_test, predicted,pos_label=pos,average=avg) 
    fast_knn_f1+=score
    score = precision_score(y_test, predicted,pos_label=pos,average=avg)  
    fast_knn_preci+=score
    score = recall_score(y_test, predicted,pos_label=pos,average=avg)  
    fast_knn_recall+=score
    score = matthews_corrcoef(y_test, predicted) 
    fast_knn_mcc+=score
    
    
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    predicted=clf.predict(X_test)
    score = accuracy_score(y_test, predicted)  
    random_forest+= score
    score = f1_score(y_test, predicted,pos_label=pos,average=avg)  
    random_forest_f1+=score
    score = precision_score(y_test, predicted,pos_label=pos,average=avg)  
    random_forest_preci+=score
    score = recall_score(y_test, predicted,pos_label=pos,average=avg)  
    random_forest_recall+=score
    score = matthews_corrcoef(y_test, predicted) 
    random_forest_mcc+=score
    
    
    clf=AdaBoostClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    predicted=clf.predict(X_test)
    score = accuracy_score(y_test, predicted)  
    adaboost+= score
    score = f1_score(y_test, predicted,pos_label=pos,average=avg)  
    adaboost_f1+=score
    score = precision_score(y_test, predicted,pos_label=pos,average=avg)  
    adaboost_preci+=score
    score = recall_score(y_test, predicted,pos_label=pos,average=avg)  
    adaboost_recall+=score
    score = matthews_corrcoef(y_test, predicted) 
    adaboost_mcc+=score
    
    
    

print("actual KNN accuracy: "+ str(actual_knn/n_fold) +" fast knn plus plus accuracy: " + str(fast_knn_pp/n_fold)+ " fast KNN accuracy: "+ str(fast_knn/n_fold)+ " random forest accuracy: "+ str(random_forest/n_fold)+" adaboost accuracy: "+ str(adaboost/n_fold) +"\n\n")
print("actual KNN F1: "+ str(actual_knn_f1/n_fold) +" fast knn plus plus F1: " + str(fast_knn_pp_f1/n_fold)+ " fast KNN F1: "+ str(fast_knn_f1/n_fold)+ " random forest F1: "+ str(random_forest_f1/n_fold)+" adaboost F1: "+ str(adaboost_f1/n_fold) +"\n\n")
print("actual KNN precision: "+ str(actual_knn_preci/n_fold) +" fast knn plus plus precision: " + str(fast_knn_pp_preci/n_fold)+ " fast KNN precision: "+ str(fast_knn_preci/n_fold)+ " random forest precision: "+ str(random_forest_preci/n_fold)+" adaboost precision: "+ str(adaboost_preci/n_fold) +"\n\n")
print("actual KNN recall: "+ str(actual_knn_recall/n_fold) +" fast knn plus plus recall: " + str(fast_knn_pp_recall/n_fold)+ " fast KNN recall: "+ str(fast_knn_recall/n_fold)+ " random forest recall: "+ str(random_forest_recall/n_fold)+" adaboost recall: "+ str(adaboost_recall/n_fold) +"\n\n")
print("actual KNN mcc: "+ str(actual_knn_mcc/n_fold) +" fast knn plus plus mcc: " + str(fast_knn_pp_mcc/n_fold)+ " fast KNN mcc: "+ str(fast_knn_mcc/n_fold)+ " random forest mcc: "+ str(random_forest_mcc/n_fold)+" adaboost mcc: "+ str(adaboost_mcc/n_fold) )
