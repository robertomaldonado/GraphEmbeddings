from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

file_name = "Gnutella"
# file_name = "CA-AstroPh"
ntwk = pd.read_csv('embeddings/deepwalk/' + str(file_name) + '.embeddings', header=0, sep=' ', lineterminator='\n', usecols=range(0,64))
print(ntwk.shape) 
n_rows = ntwk.shape[0]
# exit()
# print( ntwk.iloc[0,:].values ) 
 
mylist = []
current = 0
index = 0
lowest = 100000 #Define a large number
node = 3 #The node inspected

for x in range (0, n_rows):
    if x != node :
        vec1 = np.array( [ ntwk.iloc[node,:] ]  )
        vec2 = np.array( [ ntwk.iloc[x,:] ] )
        current = np.asscalar( euclidean_distances(vec1, vec2))
        mylist.append(current)
        if current < lowest:
            lowest = current
            index = x

print(mylist)
print("Lowest value is:" + str(lowest) + " at :" + str(index)) 
