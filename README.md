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

Structure of the project:
==

RawData:
==
Contains 2 measurement set
Linear measurement
Pendulum measurement
Both test cases contans the MoCap and IMU data in a separate file

Simulation:
==
Sin wave based simulation to tst different learning architectures

Test:
==
Different algorithms that uses the measurement files from the RawData folder

JupyterNotebook:
==
Contains the different architectures from the Test folder in Jupyter Notebook implementations 
to make testing more convinient in GoogleColab environment 

===============================================================================
*_IMU.txt file contains the measuremnt for the MARG sensor

*_MoCap.txt file contains the measurement for the MoCap system
