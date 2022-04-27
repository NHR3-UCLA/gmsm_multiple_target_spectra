import numpy as np
import math
import random
random.seed(0)

def Function_CalculateInterPeriodCorr(Periods,TypeOfCorr):

    #################################
    ###### Type Of Correlation ######
    #################################
    #TypeOfCorr = 0  horizontal across different periods
    #TypeOfCorr = 1  vertical across different periods
    #TypeOfCorr = 2  horizontal-vertical across different periods

    N_periods = len(Periods)
    CorrMatrix = np.zeros((N_periods,N_periods))

    if TypeOfCorr == 0:
        for i in range(N_periods):
            CorrMatrix[i,i] = 1
            for j in range(i+1,N_periods):
                T_max = max(Periods[j],Periods[i])
                T_min = min(Periods[j],Periods[i])
                T_min = min(10,max(0.01,T_min))
                T_max = min(10,max(0.01,T_max))
                C1 = (1-math.cos(math.pi/2 - np.log(T_max/max(T_min, 0.109)) * 0.366 ))
                if T_max < 0.2:
                    C2 = 1 - 0.105*(1 - 1/(1+np.exp(100*T_max-5)))*(T_max-T_min)/(T_max-0.0099)
                if T_max < 0.109:
                    C3 = C2
                else:
                    C3 = C1
                C4 = C1 + 0.5 * (np.sqrt(C3) - C3) * (1 + math.cos(math.pi*(T_min)/(0.109)))
                if T_max <= 0.109:
                    CorrMatrix[i,j] = C2
                elif T_min > 0.109:
                    CorrMatrix[i,j] = C1
                elif T_max < 0.2:
                    CorrMatrix[i,j] = min(C2, C4)
                else:
                    CorrMatrix[i,j] = C4
                CorrMatrix[j,i] = CorrMatrix[i,j]

    if TypeOfCorr == 1:
        for i in range(N_periods):
            CorrMatrix[i,i] = 1
            for j in range(i+1,N_periods):
                Tmax = max(Periods[j],Periods[i])
                Tmin = min(Periods[j],Periods[i])
                Tmin = min(5,max(0.05,Tmin))
                Tmax = min(5,max(0.05,Tmax))
                CorrMatrix[i,j] = 1 - 0.77*np.log(Tmax/Tmin) + 0.315*(np.log(Tmax/Tmin))**1.4
                CorrMatrix[j,i] = CorrMatrix[i,j]

    if TypeOfCorr == 2:
        for i in range(N_periods):
            for j in range(N_periods):
                Tmax = max(Periods[j],Periods[i])
                Tmin = min(Periods[j],Periods[i])
                Tmin = min(5,max(0.05,Tmin))
                Tmax = min(5,max(0.05,Tmax))
                CorrMatrix[i,j] = ( 0.64 + 0.021*np.log(np.sqrt(Tmax*Tmin)) ) * \
                                  ( 1 - math.cos( math.pi/2 - (np.log(Tmax/Tmin)) * (0.29+0.094*(Tmin<0.189)*np.log(Tmin/0.189))  ) )
    
 
    return CorrMatrix