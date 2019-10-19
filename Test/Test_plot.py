import os
import numpy as np
import matplotlib.pyplot as plt

# ====== Result Saving Option ===============================
save = False
folder = str("Results\Model_08_cm")
"""
try:
    os.mkdir(folder)
except OSError:
    print ("Creation of the directory %s failed" % folder)
else:
    print ("Successfully created the directory %s " % folder)
"""
# ============================================================

plt.plot(Y_test*30, color = 'green', label='Test Z')
plt.plot(Y_pred*30, color = 'red', label='Prediction Z')
#plt.suptitle("Position Prediction", fontsize=16)
plt.xlabel("Time [ms]")
plt.ylabel("Position [cm]")
plt.legend()
if save == True:
        plt.savefig(folder + '\AbsolutePred.png')
plt.show()

plt.plot(Y_pred[100:300], color = 'red', label='Prediction dz')
plt.suptitle("Prediction", fontsize=16)
plt.xlabel("Time [ms]")
plt.ylabel("Pos [cm]")
plt.legend()
if save == True:
        plt.savefig(folder + '\Prediction2_dz.png')
plt.show()

plt.plot(np.cumsum(Y_test), color = 'green', label='Test z')
plt.suptitle("Test dataset", fontsize=16)
plt.xlabel("Time [ms]")
plt.ylabel("Pos [cm]")
plt.legend()
if save == True:
        plt.savefig(folder + '\Test2_z.png')
plt.show()

plt.plot(np.cumsum(Y_pred), color = 'red', label='Prediction z')
plt.suptitle("Prediction", fontsize=16)
plt.xlabel("Time [ms]")
plt.ylabel("Pos [cm]")
plt.legend()
if save == True:
        plt.savefig(folder + '\Prediction2_z.png')
plt.show()