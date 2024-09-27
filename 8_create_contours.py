
import pandas
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

def get_processed_image(image_path, trial, process_name):
    filename = os.path.basename(image_path)
    return cv2.imread(f"./output/trial_{trial}/masks/{process_name}/{os.path.basename(image_path)}", cv2.IMREAD_GRAYSCALE)

def get_subdirectories(directory):
    return [f.path for f in os.scandir(directory) if f.is_dir()]

def get_process_names(directory):
    return [os.path.basename(f) for f in get_subdirectories(directory)]

def load_images_for_given_process(image_path, trial_number, process_name):
    """
        Load images from the given filenames. Returns raw image and the processed image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    p_image = cv2.imread(f"./output/trial_{trial_number}/masks/{process_name}/{os.path.basename(image_path)}", cv2.IMREAD_GRAYSCALE)
    
    return image, p_image

def plot_images_for_process(process_name, trial_number):
    images = glob.glob(f"./output/trial_{trial_number}/masks/{process_name}/*.png")
    for image_name in images:
        image, p_image = load_images_for_given_process(image_name, trial_number, process_name)
        print(f"Image: {image_name}")
        # display them both together
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(p_image, cmap='gray')
        plt.suptitle(f"{os.path.basename(image_name)} - {process_name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def get_filtered_contours(image, min_area=None, max_area=None):
    if min_area is None:
        min_area = 0
    
    if (max_area is None) or (max_area <= min_area):
        max_area = np.inf
        
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return [c for c in contours if min_area < cv2.contourArea(c) < max_area]

def create_output_directories(output_directory):
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(f"{output_directory}/lumen", exist_ok=True)
    os.makedirs(f"{output_directory}/membrane", exist_ok=True)
    os.makedirs(f"{output_directory}/contour_comparison", exist_ok=True)

def execute_process(process_name, trial_number, images):
    images_dict = {}
    output_directory = f"./output/trial_{trial_number}/processed_images/{process_name}"

    create_output_directories(output_directory)
    num_hash = (80 - len(process_name) - 2) // 2
    hashes = "#" * num_hash
    print(f"{hashes}#{"#" * len(process_name)}#{hashes}")
    print(f"{hashes} {process_name} {hashes}")
    print(f"{hashes}#{"#" * len(process_name)}#{hashes}")
    for i, image_name in enumerate(images):
        print(f"Processing image {i+1} of {len(images)}: {image_name}", flush=True)
        
        image, p_image = load_images_for_given_process(image_name, trial_number, process_name)

        # get the lumen image
        p_image_inv = cv2.bitwise_not(p_image) # the p image has black for membrane. Contours is looking for the white area. So to 
        lumen_contours = get_filtered_contours(p_image_inv, min_area=80) # get the lumen contours, filter out small areas
        
        # create a secondary filter to remove the small contours, based on the area of the contours we found
        adj_min_area = np.mean([cv2.contourArea(c) for c in lumen_contours])/3
        print(f"Adjusted min area: {adj_min_area}")
        
        lumen_contours = get_filtered_contours(p_image_inv, min_area=adj_min_area)
        
        lumen_image = np.ones_like(p_image) # create a white image
        
        cv2.drawContours(lumen_image, lumen_contours, -1, (255, 255, 255), -1) # fill the lumen contours with black
        
        # get the membrane image, which is already black in the p_image. So invert the lumen image
        membrane_image = cv2.bitwise_not(lumen_image)

        # save the lumen_image and membrane_image to disk as binary images
        lumen_image_filename = f"{output_directory}/lumen/{os.path.basename(image_name)}"
        membrane_image_filename = f"{output_directory}/membrane/{os.path.basename(image_name)}"
        
        print(f"Saving lumen image to {lumen_image_filename}")
        print(f"Saving membrane image to {membrane_image_filename}")
        
        cv2.imwrite(lumen_image_filename, lumen_image)
        cv2.imwrite(membrane_image_filename, membrane_image)

        image_dict = {
            "image_name": image_name,
            "image": image,
            "p_image": p_image,
            "lumen": lumen_image,
            "membrane": membrane_image
        }
        
        # add it to images_dict
        print("Processing image", i+1, "of", len(images), ": ", image_name)
        images_dict[os.path.basename(image_name)] = image_dict

    # print(f"Finished processing {len(images_dict)} images for {process_name}", flush=True)

    # for i, image_dict in enumerate(images_dict):
        
        # image_name, image, p_image, lumen_image, membrane_image = images_dict[os.path.basename(images[i])].values()

        # print("Image", i+1, "of", len(images_dict))

        image_name, image, p_image, lumen_image, membrane_image = image_dict.values()

        fig, ax = plt.subplots(1, 4, figsize=(15, 5))

        ax[0].imshow(image, cmap='gray')
        ax[0].axis('off')
        ax[0].set_title('Original Image')

        ax[1].imshow(p_image, cmap='gray')
        ax[1].axis('off')
        ax[1].set_title('Processed Image')

        ax[2].imshow(lumen_image, cmap='gray')
        ax[2].axis('off')
        ax[2].set_title('Lumen Contours')

        ax[3].imshow(membrane_image, cmap='gray')
        ax[3].axis('off')
        ax[3].set_title('Membrane Image')

        plt.suptitle(f"{os.path.basename(image_name)} - {process_name}")
        plt.tight_layout()
        
        #f"{output_directory}/contour_comparison/{os.path.basename(image_name)}_{process_name}_filtered.png
        filtered_image_name = f"{output_directory}/contour_comparison/{os.path.basename(image_name)}_{process_name}_filtered.png"
        print(f"Saving filtered image to {filtered_image_name}")
        
        # save it to disk
        plt.savefig(filtered_image_name)
        plt.close()
        
if __name__ == "__main__":

    trial_number = 1
    all_process_names = get_process_names(f"output/trial_{trial_number}/masks")
    
    images = glob.glob(f"output/trial_{trial_number}/rois/*.png")
    
    process_names = ["106_otsuOffset", "107_otsuOffset", "126_otsuOffset", "127_otsuOffset", "146_otsuOffset", "147_otsuOffset"]
    # output\trial_1\masks\106_otsuOffset\strip_101.png

    for process_name in process_names:
        print(process_name)
        execute_process(process_name, trial_number, images)