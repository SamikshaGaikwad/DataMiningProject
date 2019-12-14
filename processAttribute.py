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

def findKeys(dict):
    keylist = []
    for thiskey in dict.keys():
        keylist.append(thiskey)
    return keylist

def createMasterAttributeList(attribute):
    attList = []
    for thisattri in attribute[1:]:
        #print(thisattri)
        attList.extend(findKeys(processAttributeStr(thisattri)))
    return attList

def printMasterAttributeList(attList,fname):
    np.savetxt(fname, attList,fmt='%s', delimiter='\n')

def processAttributeStr(thisAttri):
    attDict = dict()
    
    thisstr = thisAttri.strip('(').strip(')')
    thisstr = re.split(":+",thisstr)
    #print(thisstr)
    for idx in range(len(thisstr)-1):
        attriTitle = re.split(',',thisstr[idx].strip())[-1]
        attriVal = re.split(',',thisstr[idx+1].strip())[0:-1]
        attDict[attriTitle] = attriVal

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
    cutOff = 5 # Cutoff weights -Ignores attributes repeated less than cutoff times
    attriWordsToIgnore = ['condition','price','ship','return','location','service','tax','gift','seller','refund','payments','expir']

    fullAttributeList = createMasterAttributeList(attribute)
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

    print("Filtered number of Attributes are: " + str(len(attriWts)))
    printMasterAttributeList(uniqAttriList,'attributeList.csv')
    printMasterAttributeList(attriWts,'attributeList1.csv')
