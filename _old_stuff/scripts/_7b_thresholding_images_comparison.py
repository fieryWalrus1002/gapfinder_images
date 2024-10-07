
# # Project Overview
# 
# 
# <img src="images/pld3280-fig-0001-m.jpg" width="800" height="600">


# # Testing the various thresholding methods
# 
# ## Otsu thresholding, post processing, and display
# ## vs Adaptive thresholding, post processing, and display 
# ## vs Adaptive Gaussian thresholding, post processing, and display
# 


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas
import glob
from itertools import product

def get_isolated_membrane(image, method, blur=True, equalize=True, metadata=None, th_metadata=None):
    
    if metadata is None:
        raise ValueError("Metadata must be provided")
    
    if th_metadata is None:
        raise ValueError("Thresholding metadata must be provided")
    
    processed = process_image(image, method, blur=blur, equalize=equalize, metadata=metadata, th_metadata=th_metadata[method])
    return processed["overlayed"]

def load_gray_image(image_path):
    print(f"Loading image from {image_path}")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def load_images_from_metadata(metadata_path, image_dir):
    """
    roi_metadata_path = "./metadata/rois_metadata_bignine.csv"
    image_dir = "rois
    """
    all_images = []
    
    # load the df and print the first few rows
    metadata = pandas.read_csv(metadata_path)
    
    # if the df is empty, return an empty, throw
    if len(metadata) == 0:
        raise ValueError(f"No metadata found in {metadata_path}")
    else:
        print(f"Loaded metadata from {metadata_path}, with {len(metadata)} rows")
    
    for image_number in len(metadata):
        image = load_gray_image(images[image_number])
        all_images.extend(image)
    
    return all_images

def equalize_image(image, clip_low=100, clip_high=255):
    equalized = cv2.equalizeHist(image)
    equalized = np.clip(equalized, clip_low, clip_high)
    return equalized

def gauss_blur(image, ksize=(5, 5), sigmaX=0):
    return cv2.GaussianBlur(image, ksize, sigmaX)

def morphological_filter(image, th_low=0, th_high=255):
    closing1 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, closing_kernel, iterations=closing_iterations)
    opening2 = cv2.morphologyEx(closing1, cv2.MORPH_OPEN, opening_kernel, iterations=opening_iterations)
    return opening2


def add_color_overlay_to_lumen(image, overlay, alpha=0.3, color=(0, 255, 0)):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Create the overlay_mask, where the overlay is color and everything else is black
    overlay_mask = np.zeros_like(image)
    overlay_mask[overlay == 255] = color   
    
    # add the overlay to the image
    cv2.addWeighted(overlay_mask, alpha, image, 1 - alpha, 0, image)
    
    print(f"image shape: {image.shape}")
    
    # return the image with the overlay
    return image


def add_membrane_overlay(image, overlay, color=(0, 0, 0)):
    """
    Show the original grayscale image wherever the overlay is white, and color (black) wherever the overlay is black.

    Args:
    - image: Grayscale input image (2D array).
    - overlay: Binary mask (2D array, same size as image).
    - color: Tuple representing the color to display where the overlay is black.

    Returns:
    - Image with the mask applied.
    """
    # Ensure the input image is grayscale and convert to BGR
    if len(image.shape) == 2:  # Check if image is grayscale
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image  # Already in BGR format

    # Create a mask for the areas to be blacked out (where overlay is 0)
    mask_inv = cv2.bitwise_not(overlay)

    # Convert the binary mask to a 3-channel BGR format
    mask_inv_bgr = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

    # Create an image with the color (black by default) in areas where the overlay is black
    color_image = np.full_like(image_bgr, color, dtype=np.uint8)

    # Use the mask to combine the original image and the color image
    result = cv2.bitwise_and(image_bgr, mask_inv_bgr) + cv2.bitwise_and(color_image, cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR))

    return result

        
def get_raw_images(metadata_path, image_dir):
    
    metadata = pandas.read_csv(metadata_path)
    strip_filenames = metadata['strip_filename'].tolist()
    images = [image_dir + strip_filename for strip_filename in strip_filenames]
    
    return images


def remove_end_slash(path):
    # lambda to remove trailing slash from mask_output_dir if needed
    remove_end_slash = lambda path: path if not path.endswith("/") else path[:-1]

    return remove_end_slash(path)

def get_filename(base_path, suptitle, strip: str):
    print("get_filename() got strip = ", strip)
    output_dir = remove_end_slash(base_path)
    os.makedirs(output_dir, exist_ok=True)

    output_filename =  f"{output_dir}/{suptitle.replace(' ', '_').replace(',', '')}_strip_{strip}.png"

def save_membrane_mask(mask_output_directory, image_dict, suptitle, strip):
    """
    save_membrane_mask(image_dict, suptitle, image_dict['metadata']['strip'].values[0])
    
    Save the membrane mask (image_dict['morphed']) to a file, with a name that includes the suptitle and strip number.
    
    Ensures that the output directory exists.
    
    """
    
    mask_output_path = f"{mask_output_directory}/{suptitle.replace(' ', '_').replace(',', '')}"
    os.makedirs(mask_output_path, exist_ok=True)    
    cv2.imwrite(f"{mask_output_path}/strip_{strip}.png", image_dict['morphed'])


# function to take blur and non blur and create a plot of them
def plot_images(base_path:str, image_dict:dict, titles=None, display_image=True):
    if titles is None:
        titles = ["Original","Thresholded", "Morphed", "Overlayed"]
    suptitle = f"{image_dict['method']} thresholding, {'blurred' if image_dict['blur'] else 'not blurred'}, {'equalized' if image_dict['equalize'] else 'not equalized'}"
    images = [image_dict["image"], image_dict["thresholded"], image_dict["morphed"], image_dict["overlayed"]]
    fig, axes = plt.subplots(1, 4, figsize=(12, 5.5), facecolor='white')
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
        ax.set_title(titles[i])
    plt.tight_layout()
    plt.suptitle(suptitle, fontsize=16)

    # save just the membrane image, so we can use it later
    save_membrane_mask(f"{base_path}/masks", image_dict,suptitle, image_dict['metadata']['strip'].values[0])

    # save the comparison of raw and processed images step by step to a file in method_comparison
    comparison_output_path = f"{base_path}/method_comparison"
    print(f"comparison_output_path: {comparison_output_path}")
    os.makedirs(comparison_output_path, exist_ok=True)
    
    filename = f"{comparison_output_path}/{suptitle.replace(' ', '_').replace(',', '')}_strip_{image_dict['metadata']['strip'].values[0]}.png"
    print(f"Saving image to {filename}")    
    plt.savefig(filename)
    
    if display_image:
        plt.show()
    else:
        plt.close()
    

################### Thresholding methods to use #######################################
def adaptive_threshold(image, th_low=0, th_high=255):
    # Adaptive Mean Thresholding: Uses the mean of the neighborhood area to calculate the threshold.
    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)
    return thresholded

def adaptive_gaussian_threshold(image, th_low=0, th_high=255):
    # Adaptive Gaussian Thresholding: Uses the weighted sum of the neighborhood area to calculate the threshold.
    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)
    return thresholded

def otsu_threshold(image, th_low=0, th_high=255):
    # Otsu's Thresholding: Automatically calculates the optimal threshold value.
    _, thresholded = cv2.threshold(image, th_low, th_high, cv2.THRESH_OTSU)
    return thresholded

def simple_threshold(image, th_low=127, th_high=255):
    # Simple Thresholding: Applies a fixed threshold value to the image.
    _, thresholded = cv2.threshold(image, th_low, th_high, cv2.THRESH_BINARY)
    return thresholded

def trunc_threshold(image, th_low=127, th_high=255):
    # Truncation Thresholding: Pixels above the threshold are set to the threshold value.
    _, thresholded = cv2.threshold(image, th_low, th_high, cv2.THRESH_TRUNC)
    return thresholded

def tozero_threshold(image, th_low=127, th_high=255):
    # To Zero Thresholding: Pixels below the threshold are set to zero.
    # Can use this as a way to remove the background from the image, leaving only the foreground.
    _, thresholded = cv2.threshold(image, th_low, th_high, cv2.THRESH_TOZERO)
    return thresholded

########################################################################################

def process_image(image, method_key, equalize=True, blur=True, metadata=None, th_metadata=None):
    """ 
        This function takes an image and processes it using the specified method. Here 
        we need to define the helper methods we use for each of the steps.
    """
    
    print(f"Processing image with method {method_key}, th_metadata: {th_metadata}")
    
    method_value = th_metadata["method"]
    th_low = th_metadata["th_low"]
    th_high = th_metadata["th_high"]
    
    # morphological filter parameters
    if equalize:
        image = equalize_image(image, clip_low=0, clip_high=255)
    if blur:
        image = gauss_blur(image, ksize=(5, 5), sigmaX=0)
        
    # threshold the image using the specified method
    if method_value == "otsu":
        thresholded = otsu_threshold(image, th_low=th_low, th_high=th_high)
    elif method_value == "adaptive":
        thresholded = adaptive_threshold(image, th_low=th_low, th_high=th_high)
    elif method_value == "adaptive_gaussian":
        thresholded = adaptive_gaussian_threshold(image, th_low=th_low, th_high=th_high)
    elif method_value == "simple":
        thresholded = simple_threshold(image, th_low=th_low, th_high=th_high)
    elif method_value == "trunc":
        thresholded = trunc_threshold(image, th_low=th_low, th_high=th_high)
    elif method_value == "tozero":
        thresholded = tozero_threshold(image, th_low=th_low, th_high=th_high)
    else:
        raise ValueError(f"Method {method_value} not recognized")
    
    morphed = morphological_filter(thresholded)
    overlayed = add_membrane_overlay(image, morphed)
    
    return {"method": method,
            "blur": blur,
            "equalize": equalize,
            "metadata": metadata,
            "image": image, "thresholded": thresholded, "morphed": morphed, "overlayed": overlayed}


if __name__ == "__main__":

    # Helmut chose 9 strips
    chosen_strip_numbers = [101, 106, 134, 135, 161, 176, 187, 229, 232]

    # parameters for the image processing
    closing_kernel = np.ones((1, 7), np.uint8)
    opening_kernel = np.ones((1, 5), np.uint8)
    closing_iterations = 1
    opening_iterations = 1
    th_low = 0
    th_high = 130
    trial_number = 1
    base_path = f"./output/trial_{trial_number}/"
    mask_path = f"{base_path}/masks/"
    th_comparison_path = f"{base_path}/threshold_comparison/"
    roi_metadata_path = f"{base_path}/081624_rois_metadata_bignine.csv"
    image_dir = f"{base_path}/rois/"
    roi_metadata = pandas.read_csv(roi_metadata_path)

    th_metadata = {
        "otsu": {"method": "otsu", "th_low": 0, "th_high": 255},
        "otsu1": {"method": "otsu", "th_low": 100, "th_high": 255},
        "otsu2": {"method": "otsu", "th_low": 150, "th_high": 255},
        "adaptive": {"method": "adaptive", "th_low": 0, "th_high": 255},
        "adaptive1": {"method": "adaptive", "th_low": 100, "th_high": 255},
        "adaptive2": {"method": "adaptive", "th_low": 150, "th_high": 255},
        "adaptive_gaussian": {"method": "adaptive_gaussian", "th_low": 0, "th_high": 255},
        "adaptive_gaussian1": {"method": "adaptive_gaussian", "th_low": 100, "th_high": 255},
        "adaptive_gaussian2": {"method": "adaptive_gaussian", "th_low": 150, "th_high": 255},
        "simple": {"method": "simple", "th_low": 127, "th_high": 255},
        "simple1": {"method": "simple", "th_low": 100, "th_high": 255},
        "simple2": {"method": "simple", "th_low": 150, "th_high": 255},
        "trunc": {"method": "trunc", "th_low": 127, "th_high": 255},
        "trunc1": {"method": "trunc", "th_low": 100, "th_high": 255},
        "trunc2": {"method": "trunc", "th_low": 150, "th_high": 255},
        "tozero": {"method": "tozero", "th_low": 127, "th_high": 255},
        "tozero1": {"method": "tozero", "th_low": 100, "th_high": 255},
        "tozero2": {"method": "tozero", "th_low": 150, "th_high": 255},
    }

    # establish the methods we are going to use
    methods = [key for key in th_metadata.keys()]
    print(f"Methods: {methods}")

    # ensure the directory exists for the processed images and mask output
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(th_comparison_path, exist_ok=True)

    ########################### threshold comparison ###########################


    equalizes = [False, True]
    blurs = [False, True]

    num_th_methods = len(methods)

    for strip_num in chosen_strip_numbers:
        
        combinations = product(methods, blurs, equalizes)
        strip_filename = roi_metadata[roi_metadata['strip'] == strip_num]['strip_filename'].values[0]
        metadata = roi_metadata[roi_metadata['strip'] == strip_num]
        image = load_gray_image(image_dir + strip_filename)

        # print otu the shape of the images
        print(f"Image shape: {image.shape}")

        # plot them all together in one image so we can compare them. 
        # make sure each has a title so we know what we are looking at
        # the first image is the original image, and will be the only image in row 1

        combo_images = []

        for method, blur, equalize in combinations:
            
            print(f"Method: {method}, blur: {blur}, equalize: {equalize}")

            title = f"{method}, {'blurred' if blur else 'not blurred'}, {'equalized' if equalize else 'not equalized'}"
            isolated_membrane = get_isolated_membrane(image, method, blur=blur, equalize=equalize, metadata=metadata, th_metadata=th_metadata)
            
            combo_images.append((title, isolated_membrane))

        # plot the isolated membranes, with a white background
        fig, axes = plt.subplots(num_th_methods, 4, figsize=(12, num_th_methods * 6), facecolor='white')

        for i, (title, isolated_membrane) in enumerate(combo_images):
            row = i // 4
            col = i % 4
            axes[row, col].imshow(isolated_membrane, cmap='gray')
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
            
        plt.tight_layout()
        plt.savefig(f"{base_path}/threshold_comparison/strip_{strip_num}_isolated_membranes.png", dpi=300, facecolor='white')
        # plt.show()
        plt.close()


