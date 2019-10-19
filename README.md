===================================
MARG based supervised learning task
===================================

This project was made to examine if it is possible to use deep learning 
arhitecture to predict orientation and position data based on a MARG sensor.

The simulation part is a case study to examine the effect of the different 
noise types during the measurement.

The test case builds on a real life measuremnt where a Phidget Spatial MARG
sensor was mounted on a pendulum. The motion of the sensor was tracked with a 
motion capture (MoCap) system. This measuremnts serves as a reference signal
during the supervised learning task. The position data from the MARG sensor and 
the sensor signals were recorded syncronously.

MeasurementInfo file contains the neccecary information for the actual log files.

===============================================================================
*_IMU.txt file contains the measuremnt for the MARG sensor
*_MoCap.txt file contains the measurement for the MoCap system
