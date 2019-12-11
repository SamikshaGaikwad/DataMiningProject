import csv
import scipy as sp
import numpy as np
import time
import pandas as pd
import re

# Python code to merge dict using a single
# expression
def mergeDict(dict1, dict2):
    return(dict2.update(dict1))

def findKeys(dict):
    keylist = []
    for thiskey in dict.keys():
        keylist.append(thiskey)
    return keylist

def readFile(filename,nRows):
    df = pd.read_csv(filename,delimiter="\t",usecols=(0,1,2,5,7),nrows=nRows)
    data = df.values

    #print(data)
    '''
    print("Printing Attributes: \n")
    print(df['attributes'].values)

    print("Printing Titles: \n")
    print(df['title'].values)

    print("Printing Subtitles: \n")
    print(df['subtitle'].values)

    print("Printing Categories: \n")
    #print(df['category'].values)
    print(df.category.unique())
    '''
    return df

def readValidation(filename):
    #data = np.genfromtxt("mlchallenge_set_validation.tsv",delimiter="\t",usecols=(0,1,2,5,7),max_rows=1000)
    data = np.genfromtxt("mlchallenge_set_validation.tsv",delimiter="\t")
    print(data[:,-1])
    print(len(np.unique(data[:,:1])))
    return data

def createMasterAttributeList(attribute):
    attList = []
    for thisattri in attribute[1:]:
        #print(thisattri)
        attList.extend(findKeys(processAttributeStr(thisattri)))
    return attList

def printMasterAttributeList(attList,fname):
    np.savetxt(fname, attList,fmt='%s', delimiter='\n')

def processAttributeStr(thisattri):
    attDict = dict()
    thisstr = thisattri.strip('(').strip(')')
    thisstr = re.split(":+",thisstr)
    #print(thisstr)
    for idx in range(len(thisstr)-1):
        attriTitle = re.split(',',thisstr[idx])[-1]
        attriVal = re.split(',',thisstr[idx+1])[0:-1]
        attDict[attriTitle] = attriVal

        #print(attDict)
    #print(attDict.keys())
    #print('')
    return attDict



# Start of Program
nRows = 1000
df = readFile("data.tsv",nRows)

attribute = df['attributes'].values

a = createMasterAttributeList(attribute)

printMasterAttributeList(np.sort(np.unique(a)),'attributeList.csv')
print(np.unique(a))




#print(data[1:6])
#print(len(df))

