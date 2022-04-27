from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
from fun_IsEqual import *
from GPUExp import *
from ElementWiseMult import *
from ElementWiseDivide import *
from IsNotEqualFun import *
from GPUnatlog import *
from MatrixSubtraction import *
from MatrixElementRep import *
from MatrixElementRep_Larger import *
from MatrixElementRep_Scalar import *
from ReshapeToVector import *

def InitialRecordSelection(my_device,sampleBig,isScaled,soil_Vs30,gm,maxScale,nGM,nBig,Nperiods,notAllowed):
    soil_Vs30 = ReshapeToVector(soil_Vs30)
    gm = ReshapeToVector(gm)
    sampleBig = ReshapeToVector(sampleBig) 
    #Check if and soil Vs30's not given
    exp_sampleBig_allperiods = GPUExp(sampleBig,my_device)   
    if (isScaled == 1):
        exp_sampleBig_sqr = ElementWiseMult(exp_sampleBig_allperiods,exp_sampleBig_allperiods,my_device)
        exp_gm = GPUExp(gm,my_device)
        gm_sampleBig = ElementWiseMult(exp_gm,exp_sampleBig_allperiods,my_device)        
        gm_sampleBig = gm_sampleBig.reshape(nBig,Nperiods,nGM)
        exp_sampleBig_sqr = exp_sampleBig_sqr.reshape(nBig,Nperiods,nGM)
        exp_sampleBig_sqr = np.sum(exp_sampleBig_sqr,1)
        gm_sampleBig = np.sum(gm_sampleBig,1)
        gm_sampleBig = ReshapeToVector(gm_sampleBig)
        exp_sampleBig_sqr = ReshapeToVector(exp_sampleBig_sqr)
                
        scaleFac2 = ElementWiseDivide(gm_sampleBig,exp_sampleBig_sqr,my_device)         
        scaleFac = scaleFac2       
        scaleFac2 = scaleFac2.reshape(nBig,1,nGM)       
        scaleFac2  = np.repeat(scaleFac2,Nperiods,axis=1) 
        scaleFac2 = ReshapeToVector(scaleFac2)
        
        exp_sampleBig_allperiods_scaled = ElementWiseMult(exp_sampleBig_allperiods,scaleFac2,my_device)
        exp_sampleBig_allperiods_scaled = GPUnatlog(exp_sampleBig_allperiods_scaled,my_device)
        #Calculate error
        err = MatrixSubtraction(exp_sampleBig_allperiods_scaled,gm,my_device)       
        err = ElementWiseMult(err,err,my_device)
        err = err.reshape(nBig,Nperiods,nGM)
        err = np.sum(err,1)   
        err = ReshapeToVector(err)
        
        scaleFac = MatrixElementRep_Larger(scaleFac,exp_sampleBig_allperiods,1000000,-1,my_device)   
        
        err = MatrixElementRep(err,scaleFac,-1,1000000,my_device)
        err = MatrixElementRep_Larger(err,scaleFac,maxScale,1000000,my_device)
        err = MatrixElementRep(err,soil_Vs30,-1,1000000,my_device)   
        err = err.reshape(nBig,nGM)
        scaleFac = scaleFac.reshape(nBig,nGM)
    else:       
        err = MatrixSubtraction(sampleBig,gm,my_device)       
        err = np.abs(err)
        err = MatrixElementRep_Scalar(err,1000000,1000000,my_device)
        err = ElementWiseMult(err,err,my_device)
        err = err.reshape(nBig,Nperiods,nGM)
        err = np.sum(err,1)
        err = ReshapeToVector(err)
        err = MatrixElementRep(err,soil_Vs30,-1,1000000,my_device)
        scaleFac = np.ones((nBig,nGM))
        err = err.reshape(nBig,nGM)
                            
    return scaleFac,err