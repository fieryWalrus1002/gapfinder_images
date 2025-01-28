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
fixed_width = 0
delta_theta = 0.0
theta_increment = 0.5

# For fine adjustment during the boundary selection
def rotate_image(image, angle):
    """ Rotates an image by a given angle and returns the rotated image """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated


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
    global x_init, y_init, drawing, top_left_pt, bottom_right_pt, image, original_image, fixed_width, delta_theta

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_init, y_init = x, y
        image = original_image.copy()  # Reset the image to the original state
        
        if delta_theta != 0:
            image = rotate_image(image, delta_theta)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            top_left_pt = (x_init, y_init)
            
            if fixed_width > 0:
                bottom_right_pt = (x_init + fixed_width, y)
            else:
                bottom_right_pt = (x, y)
            
            temp_image = original_image.copy()
            
            if delta_theta != 0:
                temp_image = rotate_image(temp_image, delta_theta)
            
            cv2.rectangle(temp_image, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
            cv2.imshow('Image', temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if fixed_width > 0:
            bottom_right_pt = (x_init + fixed_width, y)
        else:
            bottom_right_pt = (x, y)
        
        # Draw the final rectangle on the original image
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
            writer.writerow(metadata_rows.columns.tolist() + ['x1', 'y1', 'x2', 'y2', 'adj_theta'])
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



# select_rois(input_folder, output_folder, metadata_file)
def select_rois(input_folder: str, output_folder: str, output_file: str, metadata_file: str, optional_context_folder: str = None, roi_width: int = 0):
    global image, original_image, x_init, y_init, drawing, top_left_pt, bottom_right_pt, fixed_width, delta_theta, theta_increment
    
    fixed_width = roi_width
    
    os.makedirs(output_folder, exist_ok=True)
    
    if optional_context_folder is not None:
        os.makedirs(optional_context_folder, exist_ok=True)

    metadata = pd.read_csv(metadata_file)

    # Check if there is an existing ROI metadata file
    output_file = os.path.join(output_folder, 'roi_metadata.csv')
    
    if os.path.exists(output_file):
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
    for image_path in glob.glob(os.path.join(input_folder, '*.png')):
        # reset the delta_theta
        delta_theta = 0
        
        image_name = os.path.basename(image_path)

        # If the image_name is already in the existing_rois DataFrame, skip this image
        if image_name in existing_filenames:
            continue

        image = cv2.imread(image_path)
        original_image = image.copy()  # Keep a copy of the original image

        cv2.imshow('Image', image)
        cv2.setMouseCallback('Image', draw_rectangle)

        print(f"Select ROI for {image_name}. Press 's' to save ROI, 'q' to quit.")

        # add the name of the image to the window title
        window_title = f"Image - {image_name}"
        cv2.setWindowTitle('Image', window_title)

        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('a'):  # rotate image counter-clockwise by theta_increment degrees
                delta_theta += theta_increment
                image = rotate_image(original_image, delta_theta)
                cv2.imshow('Image', image)
            elif key == ord('d'):  #  rotate image clockwise by theta_increment degrees
                delta_theta -= theta_increment
                image = rotate_image(original_image, delta_theta)
                cv2.imshow('Image', image)
            elif key == ord('s') or key == 9 or key == ord('t'):  # Combine 's', 'n', Tab, and 't' keys
                roi_data = {
                    'strip_filename': image_name,
                    'x1': x_init,
                    'y1': y_init,
                    'x2': bottom_right_pt[0],
                    'y2': bottom_right_pt[1],
                    'delta_theta': delta_theta
                }
                new_rois.append(roi_data)
                if optional_context_folder is not None:
                    cv2.imwrite(f"{optional_context_folder}/{image_name}", image)
                break
            elif key == ord('n'):
                print(f"Skipping {image_name}.")
                # skip to next
                break
            elif key == ord('r'):
                # reset the image
                image = original_image.copy()
                delta_theta = 0
                cv2.imshow('Image', image) 
            elif key == ord('h'):
                displayCommands()
            elif key == ord('q'):
                handle_quit(new_rois=new_rois, output_file=output_file, input_folder=input_folder, output_folder=output_folder, metadata=metadata)
                return


    # if we reached the end, save the roi data to the output file
    # and close the window
    handle_quit(new_rois=new_rois, output_file=output_file, input_folder=input_folder, output_folder=output_folder, metadata=metadata)

def handle_quit(new_rois: list, output_file: str, input_folder: str, output_folder: str, metadata: pd.DataFrame):
    
    if new_rois:
        save_roi_coordinates(output_file, metadata, new_rois)
        
        for roi in new_rois:
            print(f"ROI for {roi['strip_filename']} saved. x1: {roi['x1']}, y1: {roi['y1']}, x2: {roi['x2']}, y2: {roi['y2']}, delta_theta: {roi['delta_theta']}")
            
            image_path = f"{input_folder}/{roi['strip_filename']}"
            
            print(f"loading image from {image_path}")
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Could not load image from {image_path}.")
                continue
            
            print(f"image shape: {image.shape}")
            
            if roi['delta_theta'] != 0:
                image = rotate_image(image, roi['delta_theta'])
            
            sliced_image = image[roi['y1']:roi['y2'], roi['x1']:roi['x2']]

            # check to see if the sliced image is empty
            if sliced_image.size == 0:
                print(f"ROI for {roi['strip_filename']} is empty. Skipping.")
                continue

            cv2.imwrite(f"{output_folder}/{roi['strip_filename']}", sliced_image)
        print(f"ROI coordinates saved to {output_file}.")
    else:
        print("No new ROIs selected, exiting program.")
        
    cv2.destroyAllWindows()
