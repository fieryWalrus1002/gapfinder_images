import pandas
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from scipy.signal import find_peaks
from scipy.signal import peak_widths

def get_data_files(type: str, search_path: str) -> list:
    return glob.glob(f"{search_path}/**/*{type}.csv", recursive=True)


    # mean_widths = calculate_mean_lumen_width(lumen_files)
def calculate_mean_lumen_width(lumen_files: list) -> list:
    mean_widths = []
    for file in lumen_files:
        data = pandas.read_csv(file)
        widths = data['width']
        mean_widths.append(np.mean(widths))
    return mean_widths

if __name__ == "__main__":
    trial = 1
    take = 3 # in case you're doing a second combination of data
    # get a list of all subdirectories in the output directory
    base_path = f"./output/trial_{trial}"
    processed_image_path = f"{base_path}/poi"
    csv_export_path = f"{base_path}/poi"
    
    os.makedirs(csv_export_path, exist_ok=True)

    lumen_files = get_data_files("lumen", processed_image_path)
    membrane_files = get_data_files("membrane", processed_image_path)

    # import all of those files into one big df for each type
    lumen_df = pandas.concat([pandas.read_csv(file) for file in lumen_files])
    membrane_df = pandas.concat([pandas.read_csv(file) for file in membrane_files])

    # create a column for lumen_df, called lumen_width_nm, which is the lumen_width column * nm_per_px
    lumen_df['lumen_width_nm'] = lumen_df['lumen_width'] * lumen_df['nm_per_px']
    membrane_df['membrane_width_nm'] = membrane_df['membrane_width'] * membrane_df['nm_per_px']

    # Now write them out to a csv
    lumen_df.to_csv(f"{csv_export_path}/lumen_{take}.csv", index=False)
    membrane_df.to_csv(f"{csv_export_path}/membrane_{take}.csv", index=False)
    print(f"lumen_df shape: {lumen_df.shape}")
    print(f"membrane_df shape: {membrane_df.shape}")
    
    
    print(f"saved data: {csv_export_path}/lumen_{take}.csv and {csv_export_path}/membrane_{take}.csv")
    
    # strip,grana_height,num_lumen,repeat_distance,px_per_nm,nm_per_px,scale,scale_pixels,lumen_width,type,index,process
    # 101,72.84482758620689,5,12.34,0.88,1.1363636363636365,500,440,4.906048906048905,lumen,0,otsu1_thresholding_blurred_equalized
    # 101,72.84482758620689,5,14.13,0.88,1.1363636363636365,500,440,4.861471861471863,lumen,1,otsu1_thresholding_blurred_equalized