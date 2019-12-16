import csv
import scipy as sp
import numpy as np
import time
import pandas as pd
import re
import mmap
from sklearn.cluster import AgglomerativeClustering

from processAttribute import appendToKey

# USES an input dictionary and creates a list of token values for each value in the dictionary
# MAIN FUNCTION
def getTokenDict(rawDict):
    str2numDict = dict()
    maxValueDict = dict()

    clusteredValueDict,MemberToLabelsDict,MemberToLabelsDict = replaceKeyWithClusterLabel(rawDict)

    # Find the list of values with same label
    #for i in memberList:
    #    clusteredValueDict[i] = findValuesFromKeyList(rawDict,LabelToMembersDict[MemberToLabelsDict[i]]

    ## WRITE A FUNCTION TO REVERSE THE DICT. WE need {keylist1:clusterLabelItBelongsTo}
    for
        str2numDict[keylist[i]] = len(rawDict[keylist[i]])


def getAgglomerateKeylist(keylist):
    clusterLabels = keylist
    keyClusterDict = dict()
    # REPLACE THE LINES BELOW WITH THE CLUSTERING ALGORITHM
    for i in range(len(keylist)):
        MemberToLabelsDict[keylist[i]] = i
        LabelToMembersDict[i] = keylist[i]
    #####

    return clusterLabels,LabelToMembersDict,MemberToLabelsDict

def replaceKeyWithClusterLabel(myDict):
    clusteredValueDict = dict()
    # Enumerate the attribute names to numbers for speed
    keylist = list(myDict.keys())

    # AGGOLOMARATE KEY LIST HERE ############
    clusterLabels,LabelToMembersDict,MemberToLabelsDict = getAgglomerateKeylist(keylist)

    # Create a replica of master dictionary containing keys as cluster labels
    for i in keylist:
        thisKey = MemberToLabelsDict[i]
        for j in

        clusteredValueDict = appendToKey(clusteredValueDict,thisKey,myDict[i])

    return clusteredValueDict,MemberToLabelsDict,MemberToLabelsDict

def replaceValue
