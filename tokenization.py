import csv
import scipy as sp
import numpy as np
import time
import pandas as pd
import re
import mmap
from sklearn.cluster import AgglomerativeClustering

from processAttribute import processAttributeStr

def tokenizeData(df,masterDict,paramName = 'attributes'):
    #paramName # contains the name of the tilte . Value = 'attributes' in this case
    keylist = list(masterDict.keys())


    # Create a dict to find out the idx for each key
    keyId = dict()
    for i in range(len(keylist)):
        keyId[keylist[i]] = i

    dataAttriList = df[paramName].values

    # Initialize the data and attribute matrix
    ###### 0 is reserved for non existing index  #####
    dataMatrix = np.full((len(dataAttri),len(masterDict)),0)

    for i in range(len(dataAttriList)):
        thisAttri = dataAttriList[i]
        thisAttri = processAttributeStr(thisAttri)
        for j in thisAttri.keys():
            dataMatrix[i,keyId[j]] = masterDict[i].index(thisAttri[j])
