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
import shutil


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

def extract_strips_from_images(centroids_csv, raw_images_folder, strip_images_folder, size=200):
    """
        The extract_strips_from_images takes the centroids and extracts a region of interest
        around that point. The region of interest is a square of size `size` centered at
        the centroid. The extracted images are saved in the `strips` folder in the input
        image folder. If the region of interest would go outside the image, the image is
        skipped.
    """
    clear_directory(strip_images_folder)

    box_size = size

    df = pd.read_csv(centroids_csv)

    if df.empty:
        print("No centroids found in the CSV file.")
        return

    for index, row in df.iterrows():
        filename = f"{raw_images_folder}/{row.iloc[0]}"
        x = row.iloc[1]
        y = row.iloc[2]
        new_filename = f"strip_{index}"

        bgr_image = cv2.imread(filename)

        centroid = extract_centroid(bgr_image, x, y, box_size)

        # check to see if the centroid is within the image
        if centroid.shape[0] != box_size or centroid.shape[1] != box_size:
            print(f"Skipping centroid [{x}, {y}] in image {filename} because the centroid is too close to the edge.")
            continue

        output_filename = f"{strip_images_folder}/{new_filename}.png"

        save_strip_metadata(filename, strip_images_folder, index, box_size, x, y)

        # Write the centroid image to strip_images directory
        cv2.imwrite(output_filename, centroid)

if __name__ == "__main__":
    main()
