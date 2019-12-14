import csv
import scipy as sp
import numpy as np
import time
import pandas as pd
import re
import numpy as np
import mmap
import sys
import nltk
stemmer = nltk.stem.porter.PorterStemmer()
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.datasets import load_files
#nltk.download('stopwords')
#import pickle
#from nltk.corpus import stopwords

from processAttribute import *

def saveDictToFile(myDict,fname):
    w = csv.writer(open(fname, "w"))
    for key, val in myDict.items():
        w.writerow([key, val])

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
    return df

def readValidation(filename):
    #data = np.genfromtxt("mlchallenge_set_validation.tsv",delimiter="\t",usecols=(0,1,2,5,7),max_rows=1000)
    data = np.genfromtxt("mlchallenge_set_validation.tsv",delimiter="\t")
    print(data[:,-1])
    print(len(np.unique(data[:,:1])))
    return data

def preProcess(df):
    for j in df.columns:
        if j in ['title', 'attributes']:
            thisDf = df[j].values
            for i in range(len(thisDf)):
                # Remove all special characters other than : and ,
                thisDf[i] = re.sub('[^,:,a-zA-Z0-9 \n\.]', ' ', thisDf[i])

                # remove all single characters
                thisDf[i] = re.sub(r'\s+[a-zA-Z]\s+', '', thisDf[i])

                # Convert to lower case and strip extra spaces
                thisDf[i] = thisDf[i].lower().strip()

                # Use nltk stemmer to create stemmed words
                thisDf[i]  = [stemmer.stem(word) for word in thisDf[i]]
                thisDf[i] =''.join(thisDf[i])
            # Assign the value back into dataframe
            df[j].value = thisDf
    return df

# Start of Program
nRows = 1002276 #maxRows = 1002276
#cutOff = 5 # Cutoff weights -Ignores attributes repeated less than cutoff times
#attriWordsToIgnore = ['condition','price','shipping','return','location']

filename = "data.tsv"
start_time = time.time()

#print(countFileLines(filename))
#end_time = time.time() - start_time

df = readFile(filename,nRows)
#print(df.columns)

# Pre-process the input file for simplification
df = preProcess(df)

# Get the attribute for processing
attribute = df['attributes'].values
masterDict = createAttributeList(attribute)
#print(masterDict)
saveDictToFile(masterDict,'MasterDictionary.csv')
#np.set_printoptions(threshold=sys.maxsize)
#print(np.array(df))
'''
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7)
X = tfidfconverter.fit_transform(attribute).toarray()
np.set_printoptions(threshold=sys.maxsize)
print(X)
'''




#print(np.unique(a))
print("Time Taken: " + str(time.time() - start_time))



#print(data[1:6])
#print(len(df))
