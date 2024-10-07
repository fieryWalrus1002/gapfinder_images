# a python script to open a df, iterate through the rows, and check the orientation of the roi coordinates
# There are x1, y1, x2, y2 coordinates in the the row. 
# The x1, y1 should be the top left corner and x2, y2 should be the bottom right corner.
# If the orientation is not correct, the script will switch the coordinates.
# The script will then save the df to a new csv file.

import pandas as pd

import pandas
import numpy as np
import cv2
import matplotlib.pyplot as plt

# read the csv file
df = pandas.read_csv('081624_roi_metadata_mod.csv')
df.head()


for i, row in df.iterrows():
    
    image = cv2.imread(row['filename'])
    x1 = row['x1']
    y1 = row['y1']
    x2 = row['x2']
    y2 = row['y2']
    

    # check distance from origin to x1, y1
    dist1 = np.sqrt(x1**2 + y1**2)
    
    # check distance from origin to x2, y2
    dist2 = np.sqrt(x2**2 + y2**2)
    
    if dist1 > dist2:
        # switch the coordinates
        x1, y1, x2, y2 = x2, y2, x1, y1
        
    # update the df
    df.at[i, 'x1'] = x1
    df.at[i, 'y1'] = y1
    df.at[i, 'x2'] = x2
    df.at[i, 'y2'] = y2
    
# save the df back out to a new csv file
df.to_csv('081624_roi_metadata_oriented.csv', index=False)