import csv
import scipy as sp
import numpy as np
import time
import pandas as pd
import re
import numpy as np
import sklearn.cluster
import distance
import mmap


from processAttribute import *


# Python code to merge dict using a single
# expression

def countFileLines(filename):
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines

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


# Start of Program
nRows = 1002276 #maxRows = 1002276
#cutOff = 5 # Cutoff weights -Ignores attributes repeated less than cutoff times
#attriWordsToIgnore = ['condition','price','shipping','return','location']



filename = "data.tsv"
start_time = time.time()

#print(countFileLines(filename))
#end_time = time.time() - start_time

df = readFile(filename,nRows)

# Find all attributes and convert to lower case
attribute = df['attributes'].values
attribute = [x.lower().strip() for x in attribute]

filteredAttribute = createAttributeList(attribute)

#clusterAttributeNames(filteredAttribute)

#print(np.unique(a))
print("Time Taken: " + str(time.time() - start_time))



#print(data[1:6])
#print(len(df))
