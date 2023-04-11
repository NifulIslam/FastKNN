import numpy as np 
import pandas as pd 
import random
import functools

class FastKNN:
    def __init__(self, x,y,k,beta=5):
        self.X_train=x.copy(deep=False)
        self.y_train=y.copy(deep=False)
        self.distant_tuple=[] # Distant tuple
        self.train_list=[] # for index sorting
    
        self.k=k
        self.beta=beta
        
        # Distant tuple initialization
        
        for i in (x.columns):
            min_=self.X_train[i].min()
            max_=self.X_train[i].max()
            lo= random.uniform(- (2**32), min_-(1e3*(max_-min_)))
            high=random.uniform(max_-(1e3*(max_-min_)), 2**32)
            self.distant_tuple.append(random.choice([lo,high]))
        
        # each value of train_list will be a list of index and distance
        for i in range(len(self.X_train)):
            self.train_list.append([i, self.getDist(self.X_train.iloc[i].values) ] )
        


        # sorting
        self.train_list=sorted(self.train_list,key=functools.cmp_to_key(self.distComp))
        
       
    def predict(self,X_test):
        self.y=[]
        all_pred=[]
        
        # df to list
        for i in range(len(X_test)):
            self.y.append(list(X_test.iloc[i]))
    
        # distance appending
        for i in range(len(self.y)):
            self.y[i].append(self.getDist(self.y[i]))
        
        
        # prediction
        for i in self.y:
            index=self.binarySearch(i) 
            min_ind=max(0,index-self.k-self.beta)
            max_ind=min(len(self.train_list),index+self.k+self.beta)
    
            temp2 = self.train_list[min_ind:max_ind]
            temp=[]
            for k in temp2:
                d=list(self.X_train.iloc[k[0]].values)
                d.append(self.y_train.iloc[k[0]])
                temp.append(d)
            ans=[]
            
            # actual KNN
            for j in range(len(temp)):
                dist= self.getDistTwo(temp[j] ,i[:-1]) 
                ans.append([dist,temp[j][-1]])
            ans.sort()
            
            majority={}
            for j in range(self.k):
                try:
                    majority[ans[j][1]]+=1
                except:
                    majority[ans[j][1]]=1

            mostVot=-1
            majorClass=""
            for l in majority.keys():
                if(majority[l]>mostVot):
                    mostVot=majority[l]
                    majorClass=l
            
            all_pred.append(majorClass)
        
        return np.array(all_pred)
        
        
    
    def binarySearch(self,To_Find):
        lo = 0
        hi = len(self.train_list) - 1
        while hi - lo > 1:
            mid = (hi + lo) // 2
            if self.train_list[mid][-1] < To_Find[-1]:
                lo = mid + 1
            else:
                hi = mid
        return hi
        
    def distComp(self,row1, row2):
        return row1[-1]-row2[-1] 
    
    def getDist(self,row1):
        dist1=0
        for i in range(len(row1)):
            dist1+=(self.distant_tuple[i]-row1[i])**2
        return dist1
    
    def getDistTwo(self,v1, v2):
        dist=0
        for i in range(len(v1)-2):
            dist+= (v1[i]-v2[i])**2
        return dist
