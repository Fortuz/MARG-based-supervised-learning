# MARG based supervised learning

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

## Structure of the project:
### RawData:

Contains 2 measurement set

Linear measurement

Pendulum measurement

Both test cases contans the MoCap and IMU data in a separate file

### Simulation:
Sin wave based simulation to tst different learning architectures

### Test:
Different algorithms that uses the measurement files from the RawData folder

### JupyterNotebook:
Contains the different architectures from the Test folder in Jupyter Notebook implementations 

to make testing more convinient in GoogleColab environment 


*_IMU.txt file contains the measuremnt for the MARG sensor

*_MoCap.txt file contains the measurement for the MoCap system

### Citation
This repository was used in the following publications. If you want to use it or build on it please cite the following articles:

```
@article{nagy2019magnetic,
  title={Magnetic angular rate and gravity sensor based supervised learning for positioning tasks},
  author={Nagy, Bal{\'a}zs and Botzheim, J{\'a}nos and Korondi, P{\'e}ter},
  journal={Sensors},
  volume={19},
  number={24},
  pages={5364},
  year={2019},
  publisher={MDPI}
}
```

```
@article{nagy2020marg,
  title={MARG sensor based position estimation for periodic movements},
  author={Nagy, Bal{\'a}zs},
  journal={Recent Innovations in Mechatronics},
  volume={7},
  number={1},
  year={2020},
  publisher={University of Debrecen}
}
```
