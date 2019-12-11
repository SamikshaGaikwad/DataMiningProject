import csv
import scipy as sp
import numpy as np
import time
import pandas as pd


def readFile(filename,nRows):
    df = pd.read_csv(filename,delimiter="\t",usecols=(0,1,2,5,7),nrows=nRows)
    data = df.values

    print(data)
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
    return data

def readValidation(filename):
    #data = np.genfromtxt("mlchallenge_set_validation.tsv",delimiter="\t",usecols=(0,1,2,5,7),max_rows=1000)
    data = np.genfromtxt("mlchallenge_set_validation.tsv",delimiter="\t")
    print(data[:,-1])
    print(len(np.unique(data[:,:1])))
    return data

def processAttributes(attribute):




# Start of Program
nRows = 5
data = readFile("data.csv",nRows)

attribute = df['attributes'].values






#print(data[1:6])
#print(len(df))
