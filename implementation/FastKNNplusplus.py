import numpy as np 
import pandas as pd 
import random
import functools
from FastKNN import FastKNN
class FastKNNpp:
    k=-1
    beta=-1
    no_of_models=1
    fast_knn_models=[]
    def __init__(self, x,y,k,beta=10,no_of_models=13,deep=True):
        self.k=k
        self.beta=beta
        self.no_of_models=no_of_models
        self.X=x.copy(deep=deep)
        self.Y=y.copy(deep=deep)
        for i in range(no_of_models):
            self.fast_knn_models.append(FastKNN(self.X,self.Y,k,beta))
    
    def predict(self,X_test):
        ans=[]
        allPredictions=[]
        
        for fastKnn in self.fast_knn_models:
            allPredictions.append(fastKnn.predict(X_test))
            
        for i in range(len(allPredictions[0])):
            majority={}
            for j in range(self.no_of_models):
                try:
                    majority[allPredictions[j][i]]+=1
                except:
                    majority[allPredictions[j][i]]=1
            mostVote=-1
            majorClass=""
            for l in majority.keys():
                if(majority[l]>=mostVote):
                    mostVote=majority[l]
                    majorClass=l
            
            ans.append(majorClass)

        return np.array(ans)
