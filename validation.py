import csv
import scipy as sp
import numpy as np
import time
import pandas as pd
import re
import numpy as np
import mmap


def printClusterAssign(df,labels):
    labelsList = np.unique(labels)
    count = 0
    empty = 0
    count1 = 0
    for i in labelsList:
        thisCluster = df[labels == i]
        x = thisCluster['0'].values

        if len(x)>1 and min(x)<10000:
            #print("New Cluster:" )
            #print(x)
            #print("\n\n" )
            count += 1
        if len(x) == 0:
            empty+=1
        if len(x) == 1:
            count1+=1
    print(len(labelsList))
    print("More than1: " + str(count) + ", Empty: " + str(empty) + ", Only 1: " + str(count1))

def createDictFromDataFrame(df):
    key = df[0].values
    values = df[1].values
    myDict= dict()
    for i in range(len(key)):
        myDict[key[i]] = values[i]
    return myDict


# Start of Program
#nRows = 10000 #maxRows = 1002276
#cutOff = 5 # Cutoff weights -Ignores attributes repeated less than cutoff times

validationFname = "mlchallenge_set_validation.tsv"


start_time = time.time()

#print(countFileLines(filename))
#end_time = time.time() - start_time

df_v = pd.read_csv(validationFname,header=None,delimiter="\t")
dict_v =  createDictFromDataFrame(df_v)
print(df_v)
index_v = df_v[0]
labels_v = df_v[1]

df_p = pd.read_csv('ClusterAssignment_Group_0.csv',header=None,delimiter=",")
print(df_p)
df_p = df_p[df_p[0].isin(index_v)]
print(df_p)
dict_p =  createDictFromDataFrame(df_p)
confusionMat = np.full((2,2),0)

# FASTER WAY TO DO THISSSSSSSSSSSSSSSSSS
runLoopN = len(index_v)
runLoopN = len(dict_p)
counter = 0
for idx in range(runLoopN):
    i = index_v[idx]
    thisAssign = dict_v[i]
    for jdx in range(idx+1,runLoopN):
        counter +=1
    #for jdx in range(idx,len(index_v)):
        j = index_v[jdx]
        if dict_v[j] == thisAssign and dict_p[i] == dict_p[j]:
            confusionMat[0,0] += 1
        elif not dict_v[j] == thisAssign and not dict_p[i] == dict_p[j]:
            confusionMat[1,1] += 1
        elif not dict_v[j] == thisAssign and dict_p[i] == dict_p[j]:
            confusionMat[0,1] += 1
        else:
            confusionMat[1,0] += 1
print(counter)
print(confusionMat)
randInd = (confusionMat[0,0] + confusionMat[1,1])/(runLoopN)/(runLoopN-1)*2
print("Rand Index: " + str(randInd))


#printClusterAssign(df_v,labels_v)
