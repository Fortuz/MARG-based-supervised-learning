"""
Data_mod_Plot(df, start=0, end=200, mode="all")
    df    : merged data frame containing IMU and MoCap infos
    start : starting element of visualization
    end   : end point of visualization
    mode  : visaualization mode (specifiing what do you need)
        all  - all graphes ploted
        qvat - only qvaternion graph
        dPos - only delta positions
        dT   - delta time only
        acc  - acceleration graph only
        gyro - gyro graphs only
        mag  - magnetometer graphs only

Infos: 
    Visualisation of a data set 
    Works on both Filtered and Not Filtered Data Frames       
"""

import matplotlib.pyplot as plt

def Data_mod_Plot(df, start=0, end=200, mode="all"):
    # Data Frame segment Plot
    dfs = df.loc[start:end]
    
    if mode=="all" or mode=="qvat":
        f1, (ax1) = plt.subplots(1, 1)
        f1.suptitle("Qvaternions", fontsize=16)
        dfs.plot(y='Qx',  color='blue',  ax=ax1, figsize=(6,2))
        dfs.plot(y='Qy',  color='red',  ax=ax1)
        dfs.plot(y='Qz',  color='green',  ax=ax1)
        dfs.plot(y='Qw',  color='black',  ax=ax1)

    if mode=="all" or mode=="dPos":
        f2, (ax2) = plt.subplots(1, 1)
        f2.suptitle("Delta Position", fontsize=16)
        dfs.plot(y='dx',  color='blue',   ax=ax2, figsize=(6,2))
        dfs.plot(y='dy',  color='orange', ax=ax2)
        dfs.plot(y='dz',  color='green',  ax=ax2)
    
    if mode=="all" or mode=="dT":
        f3, (ax3) = plt.subplots(1, 1)
        f3.suptitle("Delta Time", fontsize=16)
        dfs.plot(y='DeltaT',  color='blue',   ax=ax3, figsize=(12,2))

    if mode=="all" or mode=="acc":
        f4, (ax4) = plt.subplots(1, 1)
        f4.suptitle("Acceleration", fontsize=16)
        dfs.plot(y='acc0',  color='blue',  ax=ax4, figsize=(12,2))
        dfs.plot(y='acc1',  color='red',  ax=ax4)
        dfs.plot(y='acc2',  color='green',  ax=ax4)

    if mode=="all" or mode=="gyro":    
        f5, (ax5) = plt.subplots(1, 1)
        f5.suptitle("Gyroscope", fontsize=16)
        dfs.plot(y='gyro0',  color='blue',  ax=ax5, figsize=(12,2))
        dfs.plot(y='gyro1',  color='red',  ax=ax5)
        dfs.plot(y='gyro2',  color='green',  ax=ax5)
    
        f7, (ax7) = plt.subplots(1, 1)
        f7.suptitle("Gyroscope small", fontsize=16)
        dfs.plot(y='gyro0',  color='blue',  ax=ax7, figsize=(12,2))
        dfs.plot(y='gyro1',  color='red',  ax=ax7)
        #dfs.plot(y='gyro2',  color='green',  ax=ax7)

    if mode=="all" or mode=="mag":    
        f6, (ax6) = plt.subplots(1, 1)
        f6.suptitle("Magnetometer", fontsize=16)
        dfs.plot(y='mag0',  color='blue',  ax=ax6, figsize=(12,2))
        dfs.plot(y='mag1',  color='red',  ax=ax6)
        dfs.plot(y='mag2',  color='green',  ax=ax6)
