import cv2
import glob
import os
import csv
import pandas as pd

# Initialize global variables
x_init, y_init = -1, -1
drawing = False
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
image = None
original_image = None

# Load the existing metadata
def load_metadata(metadata_file) -> pd.DataFrame:
    """ Import the metadata from the metadata file, and return as a pandas dataframe """          
    metadata = pd.read_csv(metadata_file)

    # Add a new column 'strip_filename' to the metadata dataframe, which is the filename of the strip image
    metadata['strip_filename'] = metadata['strip'].apply(lambda x: f"strip_{x}.png")
    return metadata

def load_rotation_angles(rotation_file) -> pd.DataFrame:
    rotation_angles = {}
    with open(rotation_file, 'r') as file:
        for line in file:
            line = line.strip().split(': ')
            image_name = os.path.basename(line[0]) # will be something like strip_0.png
            rotation_angles[image_name] = float(line[1])
            
    # convert this to a pandas dataframe
    rotation_df = pd.DataFrame(rotation_angles.items(), columns=['strip_filename', 'rotation_angle'])
    return rotation_df

def roi_metadata_exists(output_file):
    return os.path.exists(output_file)

def load_existing_roi_metadata(output_file) -> pd.DataFrame:
    """ This existing roi metadata file will have the following format:
    strip_filename,raw_filename,box_size,x,y,rotation,x1,y1,x2,y2. Load as a pandas dataframe
    """
    rois = pd.read_csv(output_file)
    return rois

# Mouse callback function to draw rectangle and capture ROI coordinates
def draw_rectangle(event, x, y, flags, param):
    global x_init, y_init, drawing, top_left_pt, bottom_right_pt, image, original_image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_init, y_init = x, y
        image = original_image.copy()  # Reset the image to the original state

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            top_left_pt = (x_init, y_init)
            bottom_right_pt = (x, y)
            temp_image = image.copy()
            cv2.rectangle(temp_image, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
            cv2.imshow('Image', temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)
        cv2.rectangle(image, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
        cv2.imshow('Image', image)

def save_roi_coordinates(output_file, metadata: pd.DataFrame, rois: list):
    """ Save the ROI coordinates list of dicts to the csv file. If the file already exists, append to it. Each item in the list is a dict with the following keys:
    strip_filename, raw_filename, box_size, x, y, rotation, x1, y1, x2, y2
    """
    
    metadata_rows = metadata[metadata['strip_filename'].isin([roi['strip_filename'] for roi in rois])]
    
    # if the file does not exist, create a new file and write the roi data to it
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            
            # the new output file will be all the columns for the metadata, plus the roi columns of x1, y1, x2, y2
            writer.writerow(metadata_rows.columns.tolist() + ['x1', 'y1', 'x2', 'y2'])
            for roi in rois:
                # get the metadata row for this roi
                metadata_row = metadata_rows[metadata_rows['strip_filename'] == roi['strip_filename']]
                metadata_row = metadata_row.values.tolist()[0]
                writer.writerow(metadata_row + [roi['x1'], roi['y1'], roi['x2'], roi['y2']])
    else:
        # if the file already exists, open it in append mode and write the roi data to it
        with open(output_file, 'a', newline='') as file:
            writer = csv.writer(file)
            for roi in rois:
                # get the metadata row for this roi
                metadata_row = metadata_rows[metadata_rows['strip_filename'] == roi['strip_filename']]
                metadata_row = metadata_row.values.tolist()[0]
                writer.writerow(metadata_row + [roi['x1'], roi['y1'], roi['x2'], roi['y2']])

def displayCommands():
    print("Press 's', 'Tab', or '9' to save the ROI coordinates and move to the next image.")
    print("Press 'n' to skip to the next image.")
    print("Press 'r' to reset the current image, erasing the roi rectangle.")
    print("Press 'q' to quit the program.")
    print("Press 'h' to display this message.")

def combine_metadata_and_angles(metadata: pd.DataFrame, rotation_angles: pd.DataFrame) -> pd.DataFrame:
    """ Combine the metadata and rotation angles into a single dataframe. Retain only the metadata for the images that have rotation angles. """
    combined_df = pd.merge(metadata, rotation_angles, on='strip_filename', how='inner')
    
    print(f"Combined metadata and angles: {combined_df.columns}")
    return combined_df

def select_rois(image_folder='./rotated_images', rotation_file='./rotated_images/rotation_angles.txt', metadata_file='./strip_images/strip_metadata.csv', output_file='./metadata/roi_metadata.csv', roi_context_dir='./roi_images_context'):
    global image, original_image, x_init, y_init, drawing, top_left_pt, bottom_right_pt

    if not os.path.exists(roi_context_dir):
        os.makedirs(roi_context_dir)

    metadata = load_metadata(metadata_file)
    rotation_angles = load_rotation_angles(rotation_file)

    print(f"Metadata columns: {metadata.columns}")


    # Merge metadata with rotation angles
    metadata = combine_metadata_and_angles(metadata, rotation_angles)

    # Check if there is an existing ROI metadata file
    if roi_metadata_exists(output_file):
        # if it already exists, load the existing roi file
        existing_rois = load_existing_roi_metadata(output_file)
        existing_filenames = set(existing_rois['strip_filename'])
        print(f"Existing ROI metadata loaded from {output_file}, including: ")
        for index, row in existing_rois.iterrows():
            print(f"{row['strip_filename']} - x1: {row['x1']}, y1: {row['y1']}, x2: {row['x2']}, y2: {row['y2']}")
    else:
        existing_filenames = set()

    new_rois = [] # List to store the new ROI coordinates until we save them to the output file

    # Iterate through images
    for image_path in glob.glob(os.path.join(image_folder, '*.png')):
        image_name = os.path.basename(image_path)

        # If the image_name is already in the existing_rois DataFrame, skip this image
        if image_name in existing_filenames:
            continue

        image = cv2.imread(image_path)
        original_image = image.copy()  # Keep a copy of the original image

        cv2.imshow('Image', image)
        cv2.setMouseCallback('Image', draw_rectangle)

        print(f"Select ROI for {image_name}. Press 's' to save ROI, 'q' to quit.")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s') or key == 9 or key == ord('t'):  # Combine 's', 'n', Tab, and 't' keys
                roi_data = {
                    'strip_filename': image_name,
                    'x1': x_init,
                    'y1': y_init,
                    'x2': bottom_right_pt[0],
                    'y2': bottom_right_pt[1]
                }
                new_rois.append(roi_data)
                # save the displayed roi to the output file
                cv2.imwrite(f"{roi_context_dir}/{image_name}", image)
                break
            elif key == ord('n'):
                # skip to next
                break
            elif key == ord('r'):
                # reset the image
                image = original_image.copy()
                cv2.imshow('Image', image) 
            elif key == ord('h'):
                displayCommands()
                break
            elif key == ord('q'):
                # now we save the new_rois to the output_file
                if new_rois:
                    save_roi_coordinates(output_file, metadata, new_rois)

                cv2.destroyAllWindows()
                return  # Exit the function

    if new_rois:
        save_roi_coordinates(output_file, metadata, new_rois)
        
    cv2.destroyAllWindows()

    print(f"ROI coordinates saved to {output_file}.")

# Example usage:
# if __name__ == "__main__":
#     select_rois()