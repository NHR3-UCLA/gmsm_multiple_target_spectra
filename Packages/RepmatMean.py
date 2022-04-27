import numpy as np
from ReshapeToVector import *
from MatrixSubtraction import *

def RepmatMean( X,Y,N_Scales,n_periods,n_big,n_small,my_device ):
    
    X = X.reshape(n_small,n_periods)

    Sum_X = np.sum(X,0)
        
    RepmatX = Sum_X.reshape(1,n_periods,1)
    RepmatX = np.repeat(RepmatX,n_big,axis = 0)
    RepmatX = np.repeat(RepmatX,N_Scales,axis = 2)

    RepmatX = ReshapeToVector(RepmatX)
    
    Mean = MatrixSubtraction(RepmatX,-Y,my_device)
    Mean = Mean / (n_small+1)
    
    return Mean