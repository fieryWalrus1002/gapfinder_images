# # Extract the scale data from the raw images
# 
# - Calculate the nm/pixel ratio for the images
#   - go through the raw images. 
#   - Isolate the bottom of the image, where the black border is.
#   - This border contains a scale bar in white, and white text that says "200 nm" or "500 nm"
#   - Use OCR to extract the scale, and then get a contour of the scale bar
#   - The scale bar will be a varying amount of pixels
#   - We can then use this to calculate the nm/pixel ratio, and then use this to calculate the nm/pixel ratio for the entire image
#   - Export this number to a csv file, and then use this to calculate the nm/pixel ratio for the entire image
#     - Should have the raw image name, the nm/pixel ratio, the scale value, the scale bar length in pixels, and the scale bar length in nm to verify it is correct
# - Calibrate the extracted values from the histograms (9_signal_processing.ipynb) using the nm/pixel ratio
#   - This will allow us to get the nm values for the extracted peak widths
#   - This will allow us to compare the extracted peaks to the delta peak center values
# 

# # Set up the functions to extract the scale data

import os
import glob
from PIL import Image
import csv
import cv2
import pytesseract
import re
import os
import glob
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd

def print_metadata(metadata):
    # print this out in a pretty way. 
    print(f"Filename: {metadata['filename']}")
    print(f"Genotype: {metadata['genotype']}")
    print(f"Condition: {metadata['condition']}")
    print(f"Date: {metadata['date']}")
    print(f"Block: {metadata['block']}")
    print(f"Layer number: {metadata['layer_number']}")
    print(f"K value: {metadata['k_value']}")
    print(f"Scale: {metadata['scale']}")
    print("\n")

def process_image(filename, input_folder='./input_images'):
    # Normalize paths to avoid issues with different OS path separators
    input_folder = os.path.normpath(input_folder)
    filename = os.path.normpath(filename)
    
    print(f"Filename: {filename}")
    
    # Extract metadata from folder structure
    folder_parts = filename.strip().split(os.sep)  # Use os.sep to split based on the OS-specific separator
    
    # Check if input_folder is a prefix of the filename path
    if os.path.commonpath([input_folder, filename]) == input_folder:
        folder_parts = folder_parts[len(input_folder.split(os.sep)):]

    # print(f"Folder parts: {folder_parts}")
    genotype_condition = folder_parts[0].split('-')
    # print(f"Genotype and condition: {genotype_condition}")
    genotype = genotype_condition[0].strip(' ')
    # print(f"Genotype: {genotype}")

    condition = genotype_condition[1]

    # print(f"Condition: {condition}")
    date = folder_parts[1]
    # print(f"Date: {date}")
    block = folder_parts[2].split(' ')[-1]
    # print(f"Block: {block}")


    basename = os.path.basename(filename).split('.tif')[0]
    filename_parts = basename.split(' ')
    layer_number = filename_parts[0]
    k_value = filename_parts[-1].split('.')[0]
    scale = scan_for_text(filename)

    # Create dict to store metadata
    metadata = {
        'filename': basename,
        'genotype': genotype.strip(),
        'condition': condition.strip(),
        'date': date.strip(),
        'block': block,
        'layer_number': layer_number,
        'k_value': k_value,
        'scale': scale
        }
    
    print_metadata(metadata)

    return metadata

def scan_for_text(filename):#
        # load the image using opencv
        bgr_image = cv2.imread(filename)

        # Convert the image to grayscale
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # take only the lowest band  of pixels of the image
        lowest_30_pixels = gray[-65:, :]
        
        # Preprocess the image (e.g., apply thresholding, denoising)
        _, thresh = cv2.threshold(lowest_30_pixels, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processed_image = cv2.medianBlur(thresh, 3)

        text = pytesseract.image_to_string(processed_image)
        
        # Extract numeric values using regular expressions
        numeric_values = [int(num) for num in re.findall(r'\d+', text)]

        # Find the greatest numeric value
        return max(numeric_values)

def isolate_scalebar_image(filename, scale_bar_height: int = 65) -> np.array:
    """ Takes a numpy array of an image and isolates the scale bar image from the image.
    
        Args:
        filename (str): The filename of the image to process. has to be a png.
        scale_bar_height (int): The height of the scale bar in pixels
        
        Returns the numpy array of the isolated scale bar image.
    """

    # load the image using opencv
    bgr_image = cv2.imread(filename)

    # Convert the image to grayscale
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    
    # take only the lowest band  of pixels of the image
    scale_bar_image = gray[-scale_bar_height:, :]
    
    # Preprocess the image (e.g., apply thresholding, denoising)
    _, thresh = cv2.threshold(scale_bar_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    processed_image = cv2.medianBlur(thresh, 3)
    
    return processed_image

def extract_scale_number_from_scale_image(scale_image: np.array) -> int:
    """ Extracts the scale number from the scale image.
    
        Args:
        scale_image (np.array): The image of the scale bar
        
        Returns the scale number as an integer.
    """
    text = pytesseract.image_to_string(scale_image)
    
    # Extract numeric values using regular expressions
    numeric_values = [int(num) for num in re.findall(r'\d+', text)]

    # Find the greatest numeric value
    return max(numeric_values)

# input_folder = './raw_images'
# output_folder = './scale_bars'
# csv_filename = './metadata/image_scale_conversion.csv'
def extract_scale_conversion_metadata(input_folder: str, output_folder: str, csv_filename: str, verify: bool = False):
    os.makedirs(output_folder, exist_ok=True)

    scale_data = []

    # Iterate through the png files in the input folder
    for i, filename in enumerate(glob.glob(os.path.join(input_folder, '**', '*.png'), recursive=True)):

        # Isolate the scale bar image
        scale_bar_image = isolate_scalebar_image(filename)
        
        # Extract the scale number from the scale bar image
        scale = extract_scale_number_from_scale_image(scale_bar_image)
        
        # invert the image
        scale_bar_image = cv2.bitwise_not(scale_bar_image)
        
        hist = np.sum(scale_bar_image, axis=0)
        
        # calculate the peaks with scipy
        peaks, _ = find_peaks(hist, height=1000)
        
        # # plot the histogram
        # plt.plot(hist)
        # # add the peaks to the plot
        # plt.plot(peaks, hist[peaks], "x")
        
        # get the distance between the first two peaks
        scale_pixels = peaks[1] - peaks[0]
        
        # Save the scale bar image
        output_filename = os.path.join(output_folder, os.path.basename(filename))
        cv2.imwrite(output_filename, scale_bar_image)
        
        scale_dict = {
            'filename': filename,
            'scalebar_filename': output_filename,
            'scale': scale,
            'x0':peaks[0],
            'x1':peaks[1],
            'scale_pixels': scale_pixels,
            'nm_per_pixel': scale / scale_pixels,
            'pixel_per_nm': scale_pixels / scale
        }
        
        print(scale_dict)
        
        # append to our list of scale data
        scale_data.append(scale_dict)

    # combine the list of dictionaries into a csv file
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'scalebar_filename', 'scale', 'x0', 'x1', 'scale_pixels', 'nm_per_pixel', "pixel_per_nm"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for scale_dict in scale_data:
            writer.writerow(scale_dict)
            
    if verify:
        verify_scale_conversion_metadata(csv_filename, output_folder)
            

def verify_scale_conversion_metadata(csv_filename: str, output_folder: str):

    df = pd.read_csv(csv_filename)
    # # We can verify the output by plotting the scale bar on the image

    verified_folder = f"{output_folder}/verified"
    os.makedirs(verified_folder, exist_ok=True)

    # iterate through the dataframe, and plot the scale bar image with the scale bar overlaid
    for i, row in df.iterrows():
        # load the image
        image = cv2.imread(row['scalebar_filename'])

        #convert iamge to color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # create a plot
        fig, ax = plt.subplots()

        # plot a vertical line in red at (0, x0)
        ax.axvline(x=row['x0'], color='red')

        # plot a vertical line in red at (0, x1)
        ax.axvline(x=row['x1'], color='red')

        # plot the image
        ax.imshow(image, cmap='gray')
        
        plt.title(f"Scale: {row['scale']} nm")
        plt.savefig(f"{verified_folder}/{os.path.basename(row['scalebar_filename'])}")
        plt.close()


