import csv
import scipy as sp
import numpy as np
import time
import pandas as pd
import re
import mmap
#import sklearn.cluster
#import distance

def mergeDict(dict1, dict2):
    return(dict2.update(dict1))

def findKeys(dictn):
    keylist = []
    for thiskey in dictn.keys():
        keylist.append(thiskey)
    return keylist

def appendToKey(myDict,key,value):
    temp = myDict.pop(key,[])
    temp.append(value)
    myDict[key] = value
    return myDict

def dictPush(myDict,key,value):
    temp = {key:value}

def createMasterAttributeList(attributeRows):
    attList = []
    masterDict = dict()
    for thisattri in attributeRows:
        thisAttriDict = processAttributeStr(thisattri)
        thisAttrKeys = findKeys(thisAttriDict)
        attList.extend(thisAttrKeys)
        for x in list(thisAttrKeys):
            masterDict = appendToKey(masterDict,x,thisAttriDict[x])
    return attList,masterDict

def printMasterAttributeList(attList,fname):
    np.savetxt(fname, attList,fmt='%s', delimiter='\n')

def processAttributeStr(thisAttri):
    attDict = dict()

    # Taken care in pre-processing step
    #thisstr = thisAttri.strip('(').strip(')')
    thisstr = re.split(":+",thisAttri)
    strSz = len(thisstr)
    for idx in range(strSz-1):
        attriTitle = re.split(',',thisstr[idx].strip())[-1]
        if idx == strSz -2:
            attriVal = re.split(',',thisstr[idx+1].strip())[0:]
        else:
            attriVal = re.split(',',thisstr[idx+1].strip())[0:-1]

        #attDict[attriTitle] = attriVal
        appendToKey(attDict,attriTitle,attriVal)
        #print(attDict)
    return attDict

def calcAttributeWts(thisAttri):
    ##### IMPROVE THE SPEED OF THIS #################
    l = set()
    #uniqueList = [x for x in thisAttri if x not in l and not l.add(x)]
    #wt = [x for x in thisAttri ]
    b = {}
    for item in thisAttri:
        b[item] = b.get(item, 0) + 1
    #a = list(set(a))
    return list(b.keys()),list(b.values())

def filterAttriByWts(uniqAttriList,attriWts,cutOff):
    goodId = [x>cutOff for x in attriWts]
    uniqAttriList = np.array(uniqAttriList)
    attriWts = np.array(attriWts)
    return uniqAttriList[goodId],attriWts[goodId]

def filterAttriByIgnoreList(uniqAttriList,attriWts,attriWordsToIgnore):
    goodId = len(uniqAttriList) * [True]
    for thisWord in attriWordsToIgnore:
        goodId = np.logical_and(goodId,[x.find(thisWord)==-1 for x in uniqAttriList])
    uniqAttriList = np.array(uniqAttriList)
    attriWts = np.array(attriWts)
    return uniqAttriList[goodId],attriWts[goodId]

# FAILED: Tries to cluster list of attributes into clusters
def clusterAttributeNames(attributes):
    attributes = np.asarray(attributes)
    lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in attributes] for w2 in attributes])
    affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
    affprop.fit(lev_similarity)
    for cluster_id in np.unique(affprop.labels_):
        exemplar = attributes[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(attributes[np.nonzero(affprop.labels_==cluster_id)])
        cluster_str = ", ".join(cluster)
        print(" - *%s:* %s" % (exemplar, cluster_str))


# Main function to filter attributes
def createAttributeList(attribute):
    cutOff = 2 # Cutoff weights -Ignores attributes repeated less than cutoff times
    attriWordsToIgnore = ['condition','price','ship','return','location','service','tax','gift','seller','refund','payments','expir']

    fullAttributeList,masterDict = createMasterAttributeList(attribute)
    print("Original Number of Attributes: " + str(len(fullAttributeList)))
    #uniqAttriList = np.sort(np.unique(fullAttributeList))
    #uniqAttributeList = list(set(fullAttributeList))

    # Get Case Insensitive Unique list
    uniqAttriList,attriWts = calcAttributeWts(fullAttributeList) #set()
    #uniqAttriList = [x for x in fullAttributeList if x.lower() not in uniqAttriList and not uniqAttriList.add(x.lower())]

    uniqAttriList,attriWts = filterAttriByWts(uniqAttriList,attriWts,cutOff)

    uniqAttriList,attriWts = filterAttriByIgnoreList(uniqAttriList,attriWts,attriWordsToIgnore)

    # Cluster list of attributes to cut down on attributes further
    #clusterAttributeNames(filteredAttribute)

    # Delete the keys from master list based on new unique attribute list
    masterDictFiltered = dict()
    for keys in uniqAttriList:
        thisVal = masterDict[keys]

        # The ignore list words are also checked in the values of the attributes (eg: note: free shipping)
        ignWordsContained = False
        for thisValue in attriWordsToIgnore:
            if thisValue.find(thisWord)>0:
                ignWordsContained = True

        if len(thisVal) >0 and not ignWordsContained:
            masterDictFiltered.update({keys:masterDict[keys]})

    uniqAttriList = list(masterDictFiltered.keys())


    print("Filtered number of Attributes are: " + str(len(attriWts)))
    printMasterAttributeList(uniqAttriList,'attributeList.csv')
    printMasterAttributeList(attriWts,'attributeList1.csv')
    return masterDictFiltered
