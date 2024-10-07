
import pandas
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

def get_processed_image(image_path, masks_path, process_name):
    filename = os.path.basename(image_path)
    return cv2.imread(f"{masks_path}/{process_name}/{os.path.basename(image_path)}", cv2.IMREAD_GRAYSCALE)

def get_subdirectories(directory):
    return [f.path for f in os.scandir(directory) if f.is_dir()]

def get_process_names(directory):
    return [os.path.basename(f) for f in get_subdirectories(directory)]

def load_images_for_given_process(image_path, masks_path, process_name):
    """
        Load images from the given filenames. Returns raw image and the processed image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(f"{masks_path}/{process_name}/{os.path.basename(image_path)}", cv2.IMREAD_GRAYSCALE)
    
    return image, mask

def plot_images_for_process(process_name, masks_path):
    images = glob.glob(f"{masks_path}/{process_name}/*.png")
    for image_name in images:
        image, mask = load_images_for_given_process(image_name, masks_path, process_name)
        print(f"Image: {image_name}")
        # display them both together
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
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

def execute_process(process_name, masks_path, output_path, images):
    images_dict = {}
    output_directory = f"{output_path}/processed_images/{process_name}"

    create_output_directories(output_directory)
    num_hash = (80 - len(process_name) - 2) // 2
    hashes = "#" * num_hash
    print(f"{hashes}#{"#" * len(process_name)}#{hashes}")
    print(f"{hashes} {process_name} {hashes}")
    print(f"{hashes}#{"#" * len(process_name)}#{hashes}")

    for i, image_name in enumerate(images):
        print(f"Processing image {i+1} of {len(images)}: {image_name}", flush=True)
        
        image, mask = load_images_for_given_process(image_name, masks_path, process_name)

        if (image is None) or (mask is None):
            print(f"Skipping image: {image_name}")
            continue

        # mask has the membrane in black, and the lumen in white    
        # invert the image, so that the lumen is black, and the membrane is white. This 
        # THis makes it easy to get the contours of all of those small areas of 'membrane' that the thresholding has resulted in
        mask = cv2.bitwise_not(mask) 
        
        # get the filtered contours of the membrane from the mask
        membrane_contours = get_filtered_contours(mask, min_area=80)
        
        # create a secondary filter to remove the small contours, based on the area of the contours we found
        adj_min_area = np.mean([cv2.contourArea(c) for c in membrane_contours])/3
        print(f"Adjusted min area: {adj_min_area}")
        
        membrane_contours = get_filtered_contours(mask, min_area=adj_min_area)
        
        # create a black image
        membrane_image = np.zeros_like(mask)
        
        # draw the white, filtered membrane contours on the black image
        cv2.drawContours(membrane_image, membrane_contours, -1, (255, 255, 255), -1) 
        
        # get the membrane image, where the membranes will be white, on a black background
        lumen_image = cv2.bitwise_not(membrane_image)

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
            "mask": mask,
            "lumen": lumen_image,
            "membrane": membrane_image
        }
        
        # add it to images_dict
        print("Processing image", i+1, "of", len(images), ": ", image_name)
        images_dict[os.path.basename(image_name)] = image_dict

    # print(f"Finished processing {len(images_dict)} images for {process_name}", flush=True)

    # for i, image_dict in enumerate(images_dict):
        
        # image_name, image, mask, lumen_image, membrane_image = images_dict[os.path.basename(images[i])].values()

        # print("Image", i+1, "of", len(images_dict))

        image_name, image, mask, lumen_image, membrane_image = image_dict.values()

        fig, ax = plt.subplots(1, 4, figsize=(15, 5))

        ax[0].imshow(image, cmap='gray')
        ax[0].axis('off')
        ax[0].set_title('Original Image')

        ax[1].imshow(mask, cmap='gray')
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
        filtered_image_name = f"{output_directory}/contour_comparison/{os.path.basename(image_name).strip(".png")}_{process_name}_filtered.png"
        print(f"Saving filtered image to {filtered_image_name}")
        
        # save it to disk
        plt.savefig(filtered_image_name)
        plt.close()

def create_contours(masks_path: str, roi_image_path: str, output_path: str):

    process_names = get_process_names(masks_path)
    
    images = glob.glob(f"{roi_image_path}/*.png")
    
    for process_name in process_names:
        print(process_name)
        execute_process(process_name, masks_path, output_path, images)