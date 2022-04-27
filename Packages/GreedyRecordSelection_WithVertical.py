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
from RepmatMean import *
from GPUsqrt import *
from TripleMatrixComp import *
import timeit

def GreedyRecordSelection_WithVertical(my_device,sampleBig_H,isScaled,soil_Vs30,maxScale,nGM,sampleSmall_H,
                          meanReq_H,Diag_covReq_H,weights,
                          isused,notAllowed,penalty,scales,
                          nBig,Nperiods,N_Scales,
                          sampleBig_V,meanReq_V,Diag_covReq_V,sampleSmall_V,
                          VerMismatchPenalty,
                          EQID,NumberOfUsedEachEvent,MaxNoEventsFromOneEvent):
   
    scaleFac = ReshapeToVector(scales)    
    soil_Vs30 = ReshapeToVector(soil_Vs30)
    ## Horizontal Records    
    sampleBig_H = ReshapeToVector(sampleBig_H) 
    Diag_covReq_H = ReshapeToVector(Diag_covReq_H)
    meanReq_H = ReshapeToVector(meanReq_H)
    exp_sampleBig_allperiods_H = GPUExp(sampleBig_H,my_device)   
    sampleSmall_H = ReshapeToVector(sampleSmall_H)
    ## Vertical Records
    sampleBig_V = ReshapeToVector(sampleBig_V) 
    Diag_covReq_V = ReshapeToVector(Diag_covReq_V)
    meanReq_V = ReshapeToVector(meanReq_V)
    exp_sampleBig_allperiods_V = GPUExp(sampleBig_V,my_device)
    sampleSmall_V = ReshapeToVector(sampleSmall_V)
        
        
    ################################   
    ###### Horizontal Records ###### 
    ################################ 
    if (isScaled == 1):            
        scaleFac_H = MatrixElementRep_Larger(scaleFac,sampleBig_H,1000000,1000000,my_device)
        scaleFac_H = GPUnatlog(scaleFac_H,my_device) 
        ScaledSampleBig = MatrixSubtraction(sampleBig_H,-1*scaleFac_H,my_device)      
    else:       
        ScaledSampleBig = sampleBig_H
        scaleFac_H = np.ones((nBig*Nperiods*N_Scales))
    MeanAll  = RepmatMean( sampleSmall_H,ScaledSampleBig,N_Scales,Nperiods,nBig,nGM-1,my_device )
    devMean = MatrixSubtraction(MeanAll,meanReq_H,my_device) # Compute deviations from target
    #Calculate Standard Deviation
    ScaledSampleBig_sqr = ElementWiseMult(ScaledSampleBig,ScaledSampleBig,my_device)
    sampleSmall_sqr = ElementWiseMult(sampleSmall_H,sampleSmall_H,my_device)
    E_X2 = RepmatMean( sampleSmall_sqr,ScaledSampleBig_sqr,N_Scales,Nperiods,nBig,nGM-1,my_device )
    E2X = ElementWiseMult(MeanAll,MeanAll,my_device)
    Var = MatrixSubtraction(E_X2,E2X,my_device) * nGM/(nGM-1) 
    Var = GPUsqrt(Var,my_device)
    devSig = Var - Diag_covReq_H
    devMean = ElementWiseMult(devMean,devMean,my_device)
    devSig = ElementWiseMult(devSig,devSig,my_device)
    devMean = weights[0] * (devMean) #Need to improve this later
    devSig = weights[1] * (devSig) #Need to improve this later  
    devMean = weights[0] * (devMean) #Need to improve this later
    devSig = weights[1] * (devSig) #Need to improve this later  
    devTotal1_H = MatrixSubtraction(devMean,-1*devSig,my_device)
    # Penalize bad Horizontal spectra (set penalty to zero if this is not required)
    Lim1 = MatrixSubtraction(meanReq_H,-1.9*Diag_covReq_H,my_device)
    Lim2 = MatrixSubtraction(meanReq_H,1.9*Diag_covReq_H,my_device)   
    devTotal1_H = TripleMatrixComp(devTotal1_H,ScaledSampleBig,Lim1,1000000,my_device)
    devTotal1_H = TripleMatrixComp(devTotal1_H,-1*ScaledSampleBig,-1*Lim2,1000000,my_device)
    
    ##############################   
    ###### Vertical Records ###### 
    ############################## 
    if (isScaled == 1):            
        scaleFac_V = MatrixElementRep_Larger(scaleFac,sampleBig_V,1000000,1000000,my_device)
        scaleFac_V = GPUnatlog(scaleFac_V,my_device) 
        ScaledSampleBig = MatrixSubtraction(sampleBig_V,-1*scaleFac_V,my_device)      
    else:       
        ScaledSampleBig = sampleBig_V
        scaleFac_V = np.ones((nBig*Nperiods*N_Scales))
    MeanAll  = RepmatMean( sampleSmall_V,ScaledSampleBig,N_Scales,Nperiods,nBig,nGM-1,my_device )#SUM/nGM
    devMean = MatrixSubtraction(MeanAll,meanReq_V,my_device) # Compute deviations from target
    #Calculate Standard Deviation
    ScaledSampleBig_sqr = ElementWiseMult(ScaledSampleBig,ScaledSampleBig,my_device)
    sampleSmall_sqr = ElementWiseMult(sampleSmall_V,sampleSmall_V,my_device)
    E_X2 = RepmatMean( sampleSmall_sqr,ScaledSampleBig_sqr,N_Scales,Nperiods,nBig,nGM-1,my_device )
    E2X = ElementWiseMult(MeanAll,MeanAll,my_device)
    Var = MatrixSubtraction(E_X2,E2X,my_device) * nGM/(nGM-1) 
    Var = GPUsqrt(Var,my_device)
    devSig = Var - Diag_covReq_V
    devMean = ElementWiseMult(devMean,devMean,my_device)
    devSig = ElementWiseMult(devSig,devSig,my_device)
    devMean = weights[0] * (devMean) #Need to improve this later
    devSig = weights[1] * (devSig) #Need to improve this later  
    devMean = weights[0] * (devMean) #Need to improve this later
    devSig = weights[1] * (devSig) #Need to improve this later  
    devTotal1_V = MatrixSubtraction(devMean,-1*devSig,my_device)
    devTotal1_V = MatrixSubtraction(devMean,-1*devSig,my_device)
    # Penalize bad Vertical spectra (set penalty to zero if this is not required)
    Lim1 = MatrixSubtraction(meanReq_V,-1.9*Diag_covReq_V,my_device)
    Lim2 = MatrixSubtraction(meanReq_V,1.9*Diag_covReq_V,my_device)   
    devTotal1_V = TripleMatrixComp(devTotal1_V,ScaledSampleBig,Lim1,1000000,my_device)
    devTotal1_V = TripleMatrixComp(devTotal1_V,-1*ScaledSampleBig,-1*Lim2,1000000,my_device)
       
    ##### Calculate the total deviance #####
    devTotal1 = MatrixSubtraction(devTotal1_H,-VerMismatchPenalty*devTotal1_V,my_device)
      
    devTotal1 = devTotal1.reshape(nBig,Nperiods,N_Scales) 
    devTotal = np.sum(devTotal1,1)    
    devTotal = ReshapeToVector(devTotal)
    ### HOROZONTAL SCALE FACTORS ###
    scaleFac1 = scaleFac_H.reshape((nBig,Nperiods,N_Scales))
    scaleFac1 = scaleFac1[:,1,:]
    scaleFac1 = ReshapeToVector(scaleFac1)      
    devTotal = MatrixElementRep_Larger(devTotal,scaleFac1,maxScale,1000000,my_device)
    ### VERTICAL SCALE FACTORS ###
    scaleFac1 = scaleFac_V.reshape((nBig,Nperiods,N_Scales))
    scaleFac1 = scaleFac1[:,1,:]
    scaleFac1 = ReshapeToVector(scaleFac1)      
    devTotal = MatrixElementRep_Larger(devTotal,scaleFac1,maxScale,1000000,my_device)
    
    devTotal = MatrixElementRep(devTotal,soil_Vs30,-1,1000000,my_device)
    devTotal = devTotal.reshape(nBig,N_Scales)
    Ind_sorted_devTotal = np.argsort(devTotal,axis=0)
    sorted_devTotal = np.sort(devTotal,axis=0)
    for i in range(sorted_devTotal.shape[0]):
        xx = np.min(sorted_devTotal[i,:])
        yy = np.argmin(sorted_devTotal[i,:])        
        minID_GM = Ind_sorted_devTotal[i,yy]
        minID_SC = yy
        minDev = sorted_devTotal[i,yy]
        if (isused[minID_GM]==0 and NumberOfUsedEachEvent[EQID[minID_GM]]<MaxNoEventsFromOneEvent ):
            break 
    scaleFac_H = GPUExp(scaleFac_H,my_device)
    scaleFac_H = scaleFac_H.reshape((nBig,Nperiods,N_Scales))
    scaleFac_V = GPUExp(scaleFac_V,my_device)
    scaleFac_V = scaleFac_V.reshape((nBig,Nperiods,N_Scales))
    
    return minDev,minID_GM,minID_SC,scaleFac_H,scaleFac_V