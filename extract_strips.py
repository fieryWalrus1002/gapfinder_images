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
import pandas as pd


def extract_centroid(bgr_image, x, y, box_size):
    """ Take a slice of the image, centered on the x and y centroid. """
    
    # Define the min/max coordinates for the bounding box
    x_min = max(0, x - box_size // 2)
    y_min = max(0, y - box_size // 2)
    x_max = min(bgr_image.shape[1], x + box_size // 2)
    y_max = min(bgr_image.shape[0], y + box_size // 2)
      
    # Take the slice of the image
    centroid_slice = bgr_image[y_min:y_max, x_min:x_max].copy()
    
    return centroid_slice


def main():
    os.makedirs('strip_images', exist_ok=True)
    
    box_size = 200

    # open the centroids.csv in the raw_images folder
    df = pd.read_csv("./raw_images/centroids.csv")
    
    for index, row in df.iterrows():
        filename = row['Filename']
        new_filename = row['strip_filename'] 
        x = row['X'] 
        y = row['Y']
        
        print(filename, new_filename, x, y)
        
        bgr_image = cv2.imread(filename)

        centroid = extract_centroid(bgr_image, x, y, box_size)
        
        if centroid.shape != (box_size, box_size, 3):
            print("ERRRRRROR", centroid.shape)
        
        output_filename = f"strip_images/{new_filename}.png"
        print(output_filename)
        
        # Write the centroid image to strip_images directory
        cv2.imwrite(output_filename, centroid)


if __name__ == "__main__":
    main()
