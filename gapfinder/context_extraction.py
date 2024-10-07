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


def collate_centroids_and_rotation(rough_rois_csv: str, rotation_file: str) -> pd.DataFrame:
    """
    centroids has fields:
    filename,x,y,scale
    """

    roi_metadata = pd.read_csv(rough_rois_csv)
    roi_metadata = roi_metadata.sort_values(by='strip')

    # check for length of the roi_metadata
    if len(roi_metadata) == 0:
        raise ValueError("ROI metadata is empty.")

    if (not os.path.exists(rotation_file)):
        raise ValueError(f"Rotation file '{rotation_file}' does not exist.")

    rotation_angles = {}
    with open(rotation_file, 'r') as file:
        for line in file:
            line = line.strip().split(': ')
            image_name = os.path.basename(line[0])
            
            rotation_angles[image_name] = float(line[1])
            
    # create a pandas dataframe from the dictionary
    rotation_df = pd.DataFrame(rotation_angles.items(), columns=['strip_filename', 'rotation_angle'])

    # create a 'strip' column in the rotation_df, which is the strip number isolated from the strip_filename
    rotation_df['strip'] = rotation_df['strip_filename'].apply(lambda x: int(x.split('_')[1].split('.')[0]))

    # merge them on strip
    merged_df = pd.merge(roi_metadata, rotation_df, on='strip')

    # check for length of the merged_df
    if len(merged_df) == 0:
        raise ValueError("Merged dataframe is empty.")

    return merged_df

    
# Function to load an image in color
def load_image_bgr(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

# Function to shift an image so that a given point is at the center
def shift_image(image: np.ndarray, point: tuple) -> np.ndarray:
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    shift_x = center[0] - point[0]
    shift_y = center[1] - point[1]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, M, (w, h))
    return shifted_image

def rotate_image(image: np.ndarray, origin: tuple = None, theta: float = 0.0) -> np.ndarray:
    (h, w) = image.shape[:2]
    
    if origin is None:
        origin = (w // 2, h // 2)
    
    # Ensure origin is a tuple of two values
    if not isinstance(origin, tuple) or len(origin) != 2:
        raise ValueError("Origin must be a tuple of two values (x, y).")
    
    M = cv2.getRotationMatrix2D(origin, theta, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated

# Function to shift and rotate an image
def shift_and_rotate_image(image: np.ndarray, point: tuple, theta: float = 0.0) -> np.ndarray:
    shifted_image = shift_image(image, point)
    rotated_image = rotate_image(shifted_image, theta=theta)
    return rotated_image

def get_new_roi_centroid(origin: tuple, theta: float = 0.0):
    """
    Returns the (y, x) coordinates of the new origin after rotating the original coordinates by the specified angle.
    """
    # what are the new coordinates of the origin after rotation?
    x, y = origin
    x_new = x * np.cos(np.radians(theta)) - y * np.sin(np.radians(theta))
    y_new = x * np.sin(np.radians(theta)) + y * np.cos(np.radians(theta))
    return y_new, x_new


def extract_rotated_context_image(input_folder: str, output_folder: str, collated_metadata: pd.DataFrame, output_csv: str, box_size: int, padding: int = 0):
    #collated metadata has the following columns:
    # strip   filename  box_size     x   y strip_filename  rotation_angle  
    os.makedirs(output_folder, exist_ok=True)

    # add some columns for the new ROI metadata
    collated_metadata['x1'] = 0
    collated_metadata['y1'] = 0
    collated_metadata['x2'] = 0
    collated_metadata['y2'] = 0

    # Iterate through each row in the ROI metadata
    for index, row in collated_metadata.iterrows():
        strip_filename = row['strip_filename']

        # Load the original image
        original_image_path = row['filename']
        original_image = load_image_bgr(original_image_path)

        rotated_image = shift_and_rotate_image(original_image, (row['x'], row['y']), row['rotation_angle'])
        
        if (rotated_image is None):
            print(f"Error rotating image '{original_image_path}'. Skipping this image.")
            continue
        
        # The rotated image will have a new centerpoint of rotated_image.shape[1] // 2, rotated_image.shape[0] // 2
        new_origin = (rotated_image.shape[1] // 2, rotated_image.shape[0] // 2)
        
        # Extract a larger padded box
        new_box_size = box_size + padding * 2
        
        # get the upper left point, and the lower right point
        x1 = max(0, new_origin[0] - new_box_size // 2)
        y1 = max(0, new_origin[1] - new_box_size // 2)
        
        x2 = min(rotated_image.shape[1], new_origin[0] + new_box_size // 2)
        y2 = min(rotated_image.shape[0], new_origin[1] + new_box_size // 2)
        
        sliced_image = rotated_image[y1:y2, x1:x2]
                
        cv2.imwrite(os.path.join(output_folder, strip_filename), sliced_image)
        
        # add the context metdata to the collated_metadata
        collated_metadata.at[index, 'x1'] = x1
        collated_metadata.at[index, 'y1'] = y1
        collated_metadata.at[index, 'x2'] = x2
        collated_metadata.at[index, 'y2'] = y2

        

    # Save the new ROI metadata
    collated_metadata.to_csv(output_csv, index=False)

    print(f"Processing completed. New ROI metadata saved to '{output_csv}'")
