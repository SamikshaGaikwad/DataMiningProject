import csv
import scipy as sp
import numpy as np
import time
import pandas as pd
"""
df = pd.read_csv("data.tsv",delimiter="\t",usecols=(0,1,2,5,7),nrows=100)
data = df.values

print("Printing Attributes: \n")
print(df['attributes'].values)

print("Printing Titles: \n")
print(df['title'].values)

print("Printing Subtitles: \n")
print(df['subtitle'].values)

print("Printing Categories: \n")
#print(df['category'].values)
print(df.category.unique())
"""
#data = np.genfromtxt("mlchallenge_set_validation.tsv",delimiter="\t",usecols=(0,1,2,5,7),max_rows=1000)
data = np.genfromtxt("mlchallenge_set_validation.tsv",delimiter="\t")
print(data[:,-1])
print(len(np.unique(data[:,:1])))

#print(data[1:6])
#print(len(df))
