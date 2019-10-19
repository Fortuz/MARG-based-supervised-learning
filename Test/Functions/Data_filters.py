"""
df = Data_mod_Filter(df, scale=0, drop='True')
    df    - base Data Frame to the filtering
    scale - with(1) or without(0) scaling
    drop  - drop out('True') the time and track features or not('Nope')
    
Info:
    Functions for data filtering and data manipulation.
    dx, dy, dz is calculated
    DeltaT is calculated
    beta is deleted
    Qvaternion filtering, offset calibration.
    Time and track deleted
    
    After filtering provides a DataFrame.
    
"""
import pandas as pd

def Data_mod_Filter(df, scale=0, drop='True'):
    if scale==1:
        pos_scale = 30
        time_scale = 100
    else:
        pos_scale = 1
        time_scale = 1
    
    # DeltaT 
    df['timeprew'] = df['time']
    df['timeprew'] = df['timeprew'].shift(1)
    df['DeltaT'] = (df['time'] - df['timeprew'])

    # dx
    df['xprew'] = df['x']
    df['xprew'] = df['xprew'].shift(1)
    df['dx'] = (df['x'] - df['xprew'])*pos_scale

    # dY
    df['yprew'] = df['y']
    df['yprew'] = df['yprew'].shift(1)
    df['dy'] = (df['y'] - df['yprew'])*pos_scale

    # dz
    df['zprew'] = df['z']
    df['zprew'] = df['zprew'].shift(1)
    df['dz'] = (df['z'] - df['zprew'])*pos_scale

    # Drop unused columns
    df.drop(['timeprew', 'xprew', 'yprew', 'zprew', 'beta'],inplace=True, axis=1)
    # Drop out NAN row caused by the shifting
    df = df.dropna()
    
    df_array = df.values
    
    # Qvaternion magic (tracked column is in the data base still)    
    for row in range(1, df_array.shape[0]):
        errx = abs(df_array[row, 14] - df_array[row-1, 14])
        erry = abs(df_array[row, 15] - df_array[row-1, 15])
        errz = abs(df_array[row, 16] - df_array[row-1, 16])
        errw = abs(df_array[row, 17] - df_array[row-1, 17])
        err = errx + erry + errz + errw
        
        errxx = abs(- df_array[row, 14] - df_array[row-1, 14])
        erryy = abs(- df_array[row, 15] - df_array[row-1, 15])
        errzz = abs(- df_array[row, 16] - df_array[row-1, 16])
        errww = abs(- df_array[row, 17] - df_array[row-1, 17])
        eerr = errxx + erryy + errzz + errww
        
        if eerr < err:
            df_array[row, 14] = - df_array[row, 14]
            df_array[row, 15] = - df_array[row, 15]
            df_array[row, 16] = - df_array[row, 16]
            df_array[row, 17] = - df_array[row, 17]
                         
    # Magic Constants   
    for row in range(df_array.shape[0]):
        if (df_array[row, 15] < 0):
            df_array[row, 14] = - df_array[row, 14] - 1.15
            df_array[row, 15] = - df_array[row, 15] + 0.042
            df_array[row, 16] = - df_array[row, 16] + 0.80
            df_array[row, 17] =   df_array[row, 17] + 0.25
    
 
    header = df.columns.values

    df = pd.DataFrame(df_array, columns = header)
    df = df[df.tracked != 0]
    if drop == 'True':
        df.drop(['time', 'tracked'],inplace=True, axis=1)
    
    cols = df.columns.tolist()
    str = cols[16]
    del cols[16]
    cols.insert(0, str)
    df = df[cols]
    
    df = df[df.DeltaT < 1]
    df['DeltaT'] = df['DeltaT']*time_scale
    return df