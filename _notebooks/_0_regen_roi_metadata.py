# overview:
# We underestimated how large of a region we needed to extract from the original image to rotate and get the ROI.
# We need to use what we have to extract a larger region from the original image, rotate the region, and save it, so we can then run our roi extraction on the rotated image.
# At this point, we have a list of centroids, we have a list of rotation angles. These are located in 
# AFter rotating, we had some artifacts because the image that we rotated was a sub-image of the original image.
# If we want an ROI, rotated from the original image, we need to extract a larger region from the original image than we need.

# Step 0:
# revised! We were wrong, we can't load the roi metadata directly. It is incomplete. We need to go back a step and combine two files. 
# We need to load the rotated_images\rotation_angles.txt file and strip_images\strip_metadata.csv instead and combine them.
# strip_metadata.csv has the following columns:
# strip,filename,box_size,x,y
# 0,./raw_images/2022-1-10_Wild Type_500uE 1 hour_1_13_29k.png,200,596,1664
# and then we have the rotation_angles.txt file with the following format:
# strip_0.png: 0.0
# strip_1.png: 0.0
# So you can see how we need a function to load these files, then a function to combine them and save them as roi_metadata.csv. Then we can load roi_metadata.csv and continue with the rest of the script.

import os
import pandas as pd

def load_rotation_angles(rotation_file):
    """ Load rotation angles from the file """
    rotation_angles = {}
    with open(rotation_file, 'r') as file:
        for line in file:
            strip_filename, angle = line.strip().split(': ')
            rotation_angles[strip_filename] = float(angle)
    return rotation_angles

def combine_metadata(strip_metadata_file, rotation_angles_file, output_file):
    # Load strip metadata
    strip_metadata = pd.read_csv(strip_metadata_file)

    # Load rotation angles
    rotation_angles = load_rotation_angles(rotation_angles_file)

    # Add rotation angle and strip filename to the metadata
    strip_metadata['strip_filename'] = strip_metadata['strip'].apply(lambda x: f'strip_{x}.png')
    strip_metadata['rotation_angle'] = strip_metadata['strip_filename'].map(rotation_angles)

    # Initialize x1, y1, x2, y2 columns with placeholder values (to be calculated later)
    strip_metadata['x1'] = 0
    strip_metadata['y1'] = 0
    strip_metadata['x2'] = 0
    strip_metadata['y2'] = 0

    # Save the combined metadata
    strip_metadata.to_csv(output_file, index=False)

    print(f"Combined metadata saved to '{output_file}'")

# Define file paths
strip_metadata_file = './strip_images/strip_metadata.csv'
rotation_angles_file = './rotated_images/rotation_angles.txt'
output_file = './metadata/roi_metadata_revised.csv'

# Combine metadata
combine_metadata(strip_metadata_file, rotation_angles_file, output_file)
