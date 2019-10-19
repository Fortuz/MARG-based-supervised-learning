"""
df = Data_mod_Load(folder, date)
    folder - project folder of the measurement file 
    date   - date of the measurement file
    
    example: 
        folder = '..\\RawData\\Day_02\\Pendulum_03'
        date   = '2019_ 02_ 27_16_17_33_'

Info:
    This Function import the IMU and MoCap Data. 
    After the inport merges the data and provides a Data Frame
    
"""
import pandas as pd

def Data_mod_Load(folder, date):
    #Load a measurement files      

    imu_data = pd.read_csv(folder + '\\' + date + 'IMU.txt',
                               sep='\t',
                               decimal=',',
                               names=['time', 'acc0', 'acc1', 'acc2', 'gyro0', 'gyro1', 'gyro2', 'mag0', 'mag1', 'mag2'])

    mocap_data = pd.read_csv(folder + '\\' + date + 'MoCap.txt',
                                 sep='\t',
                                 decimal=',',
                                 names=['time', 'x', 'y', 'z', 'tracked', 'beta', 'Qx', 'Qy', 'Qz', 'Qw'])

    # Merge the two data file to synronise them. 
    
    df = pd.merge(imu_data, mocap_data, on=['time'], how='outer')  
    #In both dataset there are some data row that can't be matched, this data can be trown away during merging changing the
    #how parameter to 'inner'
    
    return df