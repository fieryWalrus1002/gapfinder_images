# overview:
# We underestimated how large of a region we needed to extract from the original image to rotate and get the ROI.
# We need to use what we have to extract a larger region from the original image, rotate the region, and save it, so we can then run our roi extraction on the rotated image.
# At this point, we have a list of centroids, we have a list of rotation angles. These are located in 
# AFter rotating, we had some artifacts because the image that we rotated was a sub-image of the original image.
# If we want an ROI, rotated from the original image, we need to extract a larger region from the original image than we need.

# Step 1, load the ROI metadata.
# Here are the headers for the ./roi_metadata.csv file:
# strip,filename,box_size,x,y,strip_filename,rotation_angle,x1,y1,x2,y2
# 'strip': the strip number, a unique identifier for the strip
# 'filename': the original image filename, that we will work from
# 'box_size':  the size of the box that we will extract. This is the width and heighth of the box, centered around the x,y coordinates
# 'x', 'y': indicates the centroid of the box
# 'strip_filename': indicates the strip filename
# 'rotation_angle': indicates the rotation angle of the strip
# 'x1','y1','x2','y2': The upper left and lower right coordinates of the box in the rotated strip image. Cannot be used as is, because they are relative to the original rotated image size.


# Step 2, iterate through each row in the ROI metadata and perform our operations as listed below:
# - Load the original image
# - Extract a larger padded box (original box + a padding parameter)
# - Rotate the padded subimage
# - Save the rotated image in the ./rotated_images folder
# - Save the new ROI metadata in the ./ folder as a csv file, ./roi_metadata_padded.csv. This will have the same columns as the original metadata, but with the x1, y1, x2, y2 columns updated to reflect the new coordinates in the rotated image.
import cv2
import os
import pandas as pd
import numpy as np

# Function to load an image in color
def load_image_bgr(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

# Function to rotate an image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# Load the ROI metadata
roi_metadata = pd.read_csv('./roi_metadata.csv')

# Set padding parameter
padding = 50  # Adjust this value as needed

# Create output directories if they don't exist
rotated_images_folder = './rotated_images'
if not os.path.exists(rotated_images_folder):
    os.makedirs(rotated_images_folder)

# DataFrame to store new ROI metadata
new_roi_metadata = pd.DataFrame(columns=roi_metadata.columns)

# Iterate through each row in the ROI metadata
for index, row in roi_metadata.iterrows():
    # Load the original image
    original_image_path = row['filename']
    original_image = load_image_bgr(original_image_path)

    # Extract a larger padded box
    box_size = row['box_size'] + padding * 2
    x, y = row['x'], row['y']
    x1 = max(0, x - box_size // 2)
    y1 = max(0, y - box_size // 2)
    x2 = min(original_image.shape[1], x + box_size // 2)
    y2 = min(original_image.shape[0], y + box_size // 2)
    padded_subimage = original_image[y1:y2, x1:x2]

    # Rotate the padded subimage
    rotation_angle = row['rotation_angle']
    rotated_image = rotate_image(padded_subimage, rotation_angle)

    # Save the rotated image
    strip_filename = row['strip_filename']
    rotated_image_path = os.path.join(rotated_images_folder, strip_filename)
    cv2.imwrite(rotated_image_path, rotated_image)

    # Calculate the new ROI coordinates in the rotated image
    h, w = padded_subimage.shape[:2]
    center_x, center_y = w // 2, h // 2
    M = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
    new_x1, new_y1 = np.dot(M, [row['x1'] - x1, row['y1'] - y1, 1])[:2]
    new_x2, new_y2 = np.dot(M, [row['x2'] - x1, row['y2'] - y1, 1])[:2]

    # Update the row with new ROI coordinates
    row['x1'], row['y1'] = int(new_x1), int(new_y1)
    row['x2'], row['y2'] = int(new_x2), int(new_y2)
    new_roi_metadata = pd.concat([new_roi_metadata, row.to_frame().T])

# Save the new ROI metadata
new_roi_metadata.to_csv('./roi_metadata_padded.csv', index=False)

print("Processing completed. New ROI metadata saved to './roi_metadata_padded.csv'")
