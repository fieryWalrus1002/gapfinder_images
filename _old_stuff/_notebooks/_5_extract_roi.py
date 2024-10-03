import cv2
import os
import pandas as pd

# overview:
# At this point, we have a list of centroids, we have a list of rotation angles, and we have a list of ROIs.
# AFter rotating, we had some artifacts because the image that we rotated was a sub-image of the original image.
# If we want an ROI, rotated from the original image, we need to extract a larger region from the original image than we need.

# Step 1, load the ROI metadata
#strip,filename,box_size,x,y,strip_filename,rotation_angle,x1,y1,x2,y2
# strip indicates the strip number
# filename indicates the original image filename, that we will work from
# box_size indicates the size of the box that we will extract. This is the width and heighth of the box, centered around the x,y coordinates
# x,y indicates the centroid of the box
# strip_filename indicates the strip filename
# rotation_angle indicates the rotation angle of the strip
# x1,y1,x2,y2 indicates the coordinates of the box in the rotated strip image

# So we need to load the original image, extract a larger box size, rotate the box, and then extract the ROI from the rotated box.

# Step 2, iterate through each row in the ROI metadata

# Load the ROI metadata
roi_metadata_file = './metadata/roi_metadata.csv'
roi_metadata = pd.read_csv(roi_metadata_file)

# Folder paths
rotated_images_folder = './rotated_images'
roi_images_folder = './roi_images'

# Create the ROI images folder if it doesn't exist
if not os.path.exists(roi_images_folder):
    os.makedirs(roi_images_folder)

# Iterate through each row in the ROI metadata
for index, row in roi_metadata.iterrows():
    strip_filename = row['strip_filename']
    x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])

    if x1 == x2 or y1 == y2:
        print(f"Invalid coordinates for strip {strip_filename}")
        continue

    # Load the rotated image
    image_path = os.path.join(rotated_images_folder, strip_filename)
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error loading image {image_path}")
        continue

    # Extract the ROI
    roi = image[y1:y2, x1:x2]

    # Save the ROI
    roi_image_path = os.path.join(roi_images_folder, strip_filename)
    try:
        cv2.imwrite(roi_image_path, roi)
    except Exception as e:
        print(f"Error saving ROI to {roi_image_path}: {e}")
        continue

    print(f"Saved ROI to {roi_image_path}")

print("ROI extraction completed.")
