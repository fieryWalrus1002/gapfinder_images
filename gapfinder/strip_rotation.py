import cv2
import numpy as np
import os
import glob
import shutil

# Global variables for mouse callback
destination_folder = "./rotated_images"
drawing = False
x_start, y_start, x_end, y_end = -1, -1, -1, -1
image = None
original_image = None

def displayCommands():
    print("Available commands:")
    print("  s: Save the current rotation and move to the next image.")
    print("  r: Reset the current image and remove the line.")
    print("  q: Quit the program and save the rotation angles processed so far.")
    print("  h: Display this help message again.")

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def load_image_bgr(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def draw_line(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, drawing, image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_image = image.copy()
            cv2.line(temp_image, (x_start, y_start), (x, y), (0, 255, 0), 2)
            cv2.imshow('Image', temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_end, y_end = x, y
        cv2.line(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow('Image', image)

def calculate_rotation_angle_from_line(x_start, y_start, x_end, y_end):
    angle = np.rad2deg(np.arctan2(y_end - y_start, x_end - x_start))
    return angle

def rotate_image(image_to_rotate, angle):
    (h, w) = image_to_rotate.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_to_rotate, M, (w, h))
    return rotated

def load_rotation_angles(rotation_file):
    rotation_angles = {}
    with open(rotation_file, 'r') as file:
        for line in file:
            filename, angle = line.strip().split(':')
            rotation_angles[filename.strip()] = float(angle)
    return rotation_angles

def save_rotation_angles(rotation_angles, rotation_file):
    with open(rotation_file, 'w') as file:
        for filename, angle in rotation_angles.items():
            file.write(f"{filename}: {angle}\n")

def rotate_strips(input_folder="./strip_images", output_folder="./rotated_images", rotation_file="rotation_angles.txt"):
    global destination_folder, image, original_image, x_start, y_start, x_end, y_end, drawing

    destination_folder = output_folder

    # If the destination folder does not exist, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    image_paths = glob.glob(f"{input_folder}/*.png")
    
    rotation_file = os.path.join(destination_folder, rotation_file)
    existing_rotation_angles = {}
    if os.path.exists(rotation_file):
        existing_rotation_angles = load_rotation_angles(rotation_file)

    print(f"Draw a line on the image to indicate the desired rotation. Press 's' to save, 'r' to reset, or 'q' to quit.")

    # Filter out all of the image_path in image_paths that have already been rotated
    image_paths = [image_path for image_path in image_paths if os.path.basename(image_path) not in existing_rotation_angles]
    
    for image_path in image_paths:
        image = load_image_bgr(image_path)
        original_image = image.copy()

        cv2.imshow('Image', image)
        cv2.setMouseCallback('Image', draw_line)
        
        displayCommands()

        while image_path not in existing_rotation_angles:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s') or key == ord('n') or key == 9 or key == ord('t'):  # Combine 's', 'n', Tab, and 't' keys
                rotation_angle = calculate_rotation_angle_from_line(x_start, y_start, x_end, y_end)
                print(f"Rotation angle: {rotation_angle}")
                rotated_image = rotate_image(original_image, rotation_angle)
                output_path = os.path.join(destination_folder, os.path.basename(image_path))
                print(f"Saving rotated image to: {output_path}")
                cv2.imwrite(output_path, rotated_image)
                existing_rotation_angles[os.path.basename(image_path)] = rotation_angle
                break
            elif key == ord('r') or key == ord('e'):
                image = original_image.copy()
                cv2.imshow('Image', image)
            elif key == ord('h'):
                displayCommands()
            elif key == ord('q'):
                cv2.destroyAllWindows()
                # Save the rotation angles to disk before quitting
                print("Processing aborted, saving rotation angles to disk.")
                save_rotation_angles(existing_rotation_angles, rotation_file)
                return  # Exit the function

        cv2.destroyAllWindows()
        
    # Save the rotation angles to disk after completing all images
    print("Processing completed, saving rotation angles to disk.")
    save_rotation_angles(existing_rotation_angles, rotation_file)

# Example usage:
# if __name__ == "__main__":
#     rotate_strips()