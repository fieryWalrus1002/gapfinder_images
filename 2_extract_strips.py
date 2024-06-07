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

def save_strip_metadata(image_name, dest_folder, index, rect_size, x, y):
    """Saves the strip metadata information to a csv file."""
    csv_path = os.path.join(dest_folder, 'strip_metadata.csv')
    
    # if the file doesn't exist yet, write the header
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['strip', 'filename', 'box_size', 'x', 'y'])
    
    with open(csv_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([index, image_name, rect_size, x, y])
            

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def main():
    clear_directory('strip_images')
    
    box_size = 200

    # open the centroids.csv in the raw_images folder
    df = pd.read_csv("./raw_images/centroids.csv")
    
    for index, row in df.iterrows():
        filename = f"./raw_images/{row[0]}"
        x = row[1] 
        y = row[2]
        new_filename = f"strip_{index}"
        
        bgr_image = cv2.imread(filename)

        centroid = extract_centroid(bgr_image, x, y, box_size)
                
        output_filename = f"strip_images/{new_filename}.png"
        
        save_strip_metadata(filename, 'strip_images', index, box_size, x, y)
        
        # Write the centroid image to strip_images directory
        cv2.imwrite(output_filename, centroid)

        # if centroid.shape == (box_size, box_size, 3):
        #     output_filename = f"strip_images/{new_filename}.png"
            
        #     save_strip_metadata(filename, 'strip_images', index, box_size)
            
        #     # Write the centroid image to strip_images directory
        #     cv2.imwrite(output_filename, centroid)
        # else: 
        #     print("ERROR: image centroid too close to the edge of the image.", centroid.shape)

if __name__ == "__main__":
    main()
