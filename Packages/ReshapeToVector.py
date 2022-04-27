import numpy as np

def ReshapeToVector(a):
    
    a = np.float32(a)
    
    size_a = a.shape
    finalsize = 1
    
    for i in range(len(size_a)):
        finalsize = finalsize*size_a[i]
    
    reshaped_a = a.reshape(finalsize,1)
    
    return reshaped_a