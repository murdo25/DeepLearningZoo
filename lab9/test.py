
from PIL import Image
#

import tqdm
import tensorflow as tf
import numpy as np
import random 
 #
 # assumes list.txt is a list of filenames, formatted as
 #
 # ./lfw2//Aaron_Eckhart/Aaron_Eckhart_0001.jpg
 # ./lfw2//Aaron_Guiel/Aaron_Guiel_0001.jpg
 # ...
 #
  
files = open( './list.txt' ).readlines()
 
data = np.zeros(( len(files), 250, 250, 1 ))
labels = np.zeros(( len(files), 1 ))
 
# a little hash map mapping subjects to IDs
ids = {}
scnt = 0
 
# load in all of our images
ind = 0
for fn in files:
 
    subject = fn.split('/')[3]
    if not ids.has_key( subject ):
        ids[ subject ] = scnt
        scnt += 1
    label = ids[ subject ] 
    data[ ind, :, :, :] = np.reshape(np.array( Image.open( fn.rstrip() ) ),(250,250,1))    
    labels[ ind ] = label
    ind += 1
 
# data is (13233, 250, 250)
# labels is (13233, 1)

sorted_data = {}

for i in range(len(labels)):
    #sort the faces into dictionary slots
    if(sorted_data.has_key(int(labels[i][0]))):
        sorted_data[int(labels[i][0])].append(data[i])
    else:
        sorted_data[int(labels[i][0])] = [data[i]]




def get_indx(num = 100):
    batch = []
    for i in range(num):
        indx = random.randint(0,len(sorted_data)-1)
        while(len(sorted_data[indx]) == 1 ):
            indx = random.randint(0,len(sorted_data)-1)       
        batch.append(indx)
    return batch        
 

def getBad_indx(num = 200):
    batch = []
    for i in range(num):
        indx = random.randint(0,len(sorted_data)-1)
        while(len(sorted_data[indx]) > 1 ):
            indx = random.randint(0,len(sorted_data)-1)       
        batch.append(indx)
    return batch        
     
    
