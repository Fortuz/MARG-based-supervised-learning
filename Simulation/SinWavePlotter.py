# Plot
import os
import numpy as np
import matplotlib.pyplot as plt

def PlotInput(folder, save):
    plt.plot(dtime, color='blue', label='dt with noise')
    #plt.plot(dtime_clc, color='green', label='dt without noise')
    plt.suptitle("Sampling time", fontsize=16)
    plt.xlabel("Iteration [-]")
    plt.ylabel("Time [ms]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\Timeing.png')
    plt.show()

    plt.plot(atime, color='blue', label='atime')
    plt.suptitle("Absolute time", fontsize=16)
    plt.xlabel("Iteration [-]")
    plt.ylabel("Time [ms]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\AbsoluteTime.png')
    plt.show()

    plt.plot(atime, sin_01, color='red',    label='sin_01')
    plt.plot(atime, sin_02, color='blue', label='sin_02')
    plt.suptitle("Input signals", fontsize=16)
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude [-]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\SineWaves.png')
    plt.show()
    return()

def PlotNoiseDifference(folder, save):
    plt.plot(dtime, color='blue', label='dtime_noisy')
    plt.plot(dtime_clc, color='green', label='dtime_clear')
    plt.suptitle("Sampling time", fontsize=16)
    plt.xlabel("Iteration [-]")
    plt.ylabel("Time [ms]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\TimeDiff.png')
    plt.show()
    
    plt.plot(atime, color='blue', marker='.', label='atime_noisy')
    plt.plot(atime_clc, color='green', marker='.', label='atime_clear')
    plt.suptitle("Absolute time", fontsize=16)
    plt.xlabel("Iteration [-]")
    plt.ylabel("Time [ms]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\AtimeDiff.png')
    plt.show()
    
    plt.plot(atime[0:40], sin_01[0:40],     color='red',    label='sin_01_noisy')
    plt.plot(atime[0:40], sin_01_clc[0:40], color='green', label='sin_01_clear')
    plt.suptitle("sin_01", fontsize=16)
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude [-]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\Sin_01_ShortDiff.png')
    plt.show()
    
    plt.plot(atime[0:20], sin_02[0:20],     color='blue',    label='sin_02_noisy')
    plt.plot(atime[0:20], sin_02_clc[0:20], color='green', label='sin_02_clear')
    plt.suptitle("sin_02", fontsize=16)
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude [-]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\Sin_02_ShortDiff.png')
    plt.show()
    
    plt.plot(atime, sin_01,     color='red',    label='sin_01')
    plt.suptitle("sin_01", fontsize=16)
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude [-]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\Sin_01.png')
    plt.show()
    
    plt.plot(atime, sin_02,     color='blue',    label='sin_02')
    plt.suptitle("sin_02", fontsize=16)
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude [-]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\Sin_02.png')
    plt.show()
    
    return()

def PlotOutput(folder, save):
    plt.plot(atime, dY, color='green', label='dY')
    #plt.plot(atime, dY_clc, color='green', label='dY_clc')
    plt.suptitle("Addition - dY", fontsize=16)
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude [-]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\Addition_dY.png')
    plt.show()

    plt.plot(atime, cdY, color='black', label='cdY')
    #plt.plot(atime, cdY_clc, color='green', label='cdY_clc')
    plt.suptitle("Commulative Addition - cdY", fontsize=16)
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude [-]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\ComAdd_cdY.png')
    plt.show()
    
    plt.plot(atime, iY, color='red', label='iY')
    #plt.plot(atime, iY_clc, color='green', label='iY_clc')
    plt.suptitle("Integration - iY", fontsize=16)
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude [-]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\Integration_iY.png')
    plt.show()
    
    plt.plot(atime, ciY, color='blue', label='ciY')
    #plt.plot(atime, ciY_clc, color='green', label='ciY_clc')
    plt.suptitle("Commulative Integration - ciY", fontsize=16)
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude [-]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\ComInt_ciY.png')
    plt.show()
    return()

def PlotTraining(folder, save):
    plt.plot(atime_train, Y_train, color='blue',   label='Y_Train')
    plt.plot(atime_valid, Y_valid, color='orange', label='Y_Valid')
    plt.plot(atime_test,  Y_test,  color='green',  label='Y_Test')
    plt.plot(atime_test,  Y_pred,  color='red',    label='Y_Prediction')
    plt.suptitle("Prediction", fontsize=16)
    plt.xlabel("Time [ms]")
    plt.ylabel("dPos [mm]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\Training.png')
    plt.show()
    return()
    
def History(folder, save):
    plt.plot(history.history['loss'],     label='train loss')
    plt.plot(history.history['val_loss'], label='valid loss')
    plt.suptitle("Loss", fontsize=16)
    plt.xlabel("Epochs [-]")
    plt.ylabel("Loss [-]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\History.png')
    plt.show()
    
def CommulativePlot(folder, save):
    plt.plot(atime_test,  np.cumsum(Y_test),  color='green',  label='Y_Test_Comm')
    plt.plot(atime_test,  np.cumsum(Y_pred),  color='red',    label='Y_Prediction_Comm')
    plt.suptitle("Prediction Commulative", fontsize=16)
    plt.xlabel("Time [ms]")
    plt.ylabel("Pos [mm]")
    plt.legend()
    if save == True:
        plt.savefig(folder + '\PredComm_Short.png')
    mse = (np.square( np.cumsum(Y_test) - np.cumsum(Y_pred))).mean(axis=None)
    print("Commulated Error")
    print(str(mse))
    plt.show()
    
def CommulativePrediction(folder, save):
    #plt.plot(atime_test,  np.cumsum(Y_test),  color='green',  label='Y_Test_Comm')
    plt.plot(atime, ciY, color='blue', label='ciY')
    plt.plot(atime_test,  (np.cumsum(Y_pred))+ciY[len(ciY)-len(Y_pred)],  color='red',    label='Y_Prediction_Comm')
    plt.suptitle("Prediction Commulative", fontsize=16)
    plt.xlabel("Time [ms]")
    plt.ylabel("Pos [mm]")
    plt.legend()
    if save == True:
            plt.savefig(folder + '\PredComm_Long.png')
    plt.show()
    
def GenerateTXT(folder, save):
    if save == True:
        f= open(folder + "\param.txt","w+")
        f.write(str(history.history['val_loss']))
        f.close() 


#====== Saving options ============================
# uncomment to save into folder
save = False
folder = str("Results")
"""
try:
    os.mkdir(folder)
except OSError:
    print ("Creation of the directory %s failed" % folder)
else:
    print ("Successfully created the directory %s " % folder)
"""

#====== Plotting ================
#(uncomment the needed parts)
#PlotInput(folder, save)
#PlotNoiseDifference(folder, save)
#PlotOutput(folder, save)
PlotTraining(folder, save)    
#History(folder, save)
#CommulativePlot(folder, save)
#CommulativePrediction(folder, save)
#GenerateTXT(folder, save)
#plot_model(model, to_file='model.png')