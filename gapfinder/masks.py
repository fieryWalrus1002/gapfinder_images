import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas
import glob

def get_isolated_membrane(image, method, blur=True, equalize=True, metadata=None, th_metadata=None):
    
    if metadata is None:
        raise ValueError("Metadata must be provided")
    
    if th_metadata is None:
        raise ValueError("Thresholding metadata must be provided")
    
    processed = process_image(image, method, blur=blur, equalize=equalize, metadata=metadata, th_metadata=th_metadata[method])
    return processed["overlayed"]

def load_gray_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
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




# function to take blur and non blur and create a plot of them
def plot_images(base_path:str, image_dict:dict, titles=None, display_image=True):
    # image output file name should be the key of the image_dict
    key = image_dict['key']
    
    if titles is None:
        titles = ["Original","Thresholded", "Morphed", "Overlayed"]
    suptitle = f"{key} {image_dict['method']} thresholding, {image_dict['morph_key']} morphological filter"
    images = [image_dict["image"], image_dict["thresholded"], image_dict["morphed"], image_dict["overlayed"]]
    fig, axes = plt.subplots(1, 4, figsize=(12, 5.5), facecolor='white')
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
        ax.set_title(titles[i])
    plt.tight_layout()
    plt.suptitle(suptitle, fontsize=16)

    # save just the membrane image, so we can use it later
    mask_output_path = f"{base_path}/masks/{key}"
    os.makedirs(mask_output_path, exist_ok=True)    
    cv2.imwrite(f"{mask_output_path}/strip_{image_dict['metadata']['strip'].values[0]}.png", image_dict['morphed'])

    # save the comparison of raw and processed images step by step to a file in method_comparison
    comparison_output_path = f"{base_path}/method_comparison"
    os.makedirs(comparison_output_path, exist_ok=True)
    filename = f"{comparison_output_path}/strip_{image_dict['metadata']['strip'].values[0]}_{key}.png"
    plt.savefig(filename)
    
    plt.close()
    

################### Thresholding methods to use #######################################
def simple_threshold(image, options: dict):
    # Simple Thresholding: Applies a fixed threshold value to the image.
    # Pixels above the threshold are set to the maxval, and pixels below are set to zero.
    thresh = options.get("thresh", 0)
    maxval = options.get("maxval", 255)
    _, thresholded = cv2.threshold(image, thresh, maxval, cv2.THRESH_BINARY)
    return thresholded

def trunc_threshold(image, options: dict):
    # Truncation Thresholding: Pixels above the threshold are set to the threshold value.
    # Pixels below the threshold are left unchanged.
    thresh = options.get("thresh", 0)
    maxval = options.get("maxval", 255)
    _, thresholded = cv2.threshold(image, thresh, maxval, cv2.THRESH_TRUNC)
    return thresholded

def tozero_threshold(image, options: dict):
    # To Zero Thresholding: Pixels below the threshold are set to zero.
    # Can use this as a way to remove the background from the image, leaving only the foreground.
    # thresh is the threshold value, maxval is the maximum intensity value that can be assigned to a pixel.
    # Pixels above the threshold are left unchanged.
    thresh = options.get("thresh", 0)
    maxval = options.get("maxval", 255)
    _, thresholded = cv2.threshold(image, thresh, maxval, cv2.THRESH_TOZERO)
    return thresholded


############## three options needed

def adaptive_threshold(image, options: dict):
    # Adaptive Mean Thresholding: Uses the mean of the neighborhood area to calculate the threshold.
    # Takes into account the intensity of the neighborhood area, with a higher weight for the center pixel.
    # maxval is the maximum intensity value that can be assigned to a pixel.
    # blocksize must be an odd number, typically 3, 5, 7, 9, 11, etc.
    # c is a constant subtracted from the weighted sum.
    maxval = options.get("maxval", 255)
    blocksize = options.get("blocksize", 25)
    c = options.get("c", 2)
    thresholded = cv2.adaptiveThreshold(image, maxval, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, c)
    return thresholded

def adaptive_gaussian_threshold(image, options: dict):
    # Adaptive Gaussian Thresholding: Uses the weighted sum of the neighborhood area to calculate the threshold.
    # Takes into account the variance of the neighborhood area, with a higher weight for the center pixel.
    # maxval is the maximum intensity value that can be assigned to a pixel.
    # blocksize must be an odd number, typically 3, 5, 7, 9, 11, etc.
    # c is a constant subtracted from the weighted sum.
    maxval = options.get("maxval", 255)
    blocksize = options.get("blocksize", 25)
    c = options.get("c", 2)
    thresholded = cv2.adaptiveThreshold(image, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, c)
    return thresholded


############ only one option needed
def otsu_threshold(image, options: dict):
    # Otsu's Thresholding: Automatically calculates the optimal threshold value, can be used for bimodal images.
    # maxval is the maximum intensity value that can be assigned to 
    # a pixel.
    # thresh is the threshold value calculated by Otsu's method, not used here. as input
    thresh = options.get("thresh", 0)
    maxval = options.get("maxval", 255)
    _, thresholded = cv2.threshold(image, thresh, maxval, cv2.THRESH_OTSU)
    return thresholded

################ two options, because we add in the offset multiplier

def otsu_then_offset_threshold(image, options: dict):
    """ 
    Calcualte the Otsu threshold value, then add an offset to the threshold value.
    Use that newly offset value to re-threshold using a binary threshold.
    """
    thresh = options.get("thresh", 0)
    maxval = options.get("maxval", 255)
    offset = options.get("offset", 1.0)
    
    otsu_th, thresholded = cv2.threshold(image, thresh, maxval, cv2.THRESH_OTSU)

    new_thresh = int(otsu_th * offset)
    print(f"Otsu threshold: {otsu_th}, new threshold: {new_thresh}")
    
    _, thresholded = cv2.threshold(image, new_thresh, maxval, cv2.THRESH_BINARY)
    
    return thresholded
    


########################### Morph Filter functions ######################################

def morph_close_then_open(image, morph_args: dict):
    """
        Perform opening and closing on the given image, using the specified kernels and iterations in the morph_args dictionary.
    """
    if (not "closing_kernel" in morph_args) or (not "opening_kernel" in morph_args) or (not "closing_iterations" in morph_args) or (not "opening_iterations" in morph_args):
        raise ValueError("Morphological arguments must include closing_kernel, opening_kernel, closing_iterations, and opening_iterations")
    
    closing1 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, morph_args["closing_kernel"], iterations=morph_args["closing_iterations"])
    opening2 = cv2.morphologyEx(closing1, cv2.MORPH_OPEN, morph_args["opening_kernel"], iterations=morph_args["opening_iterations"])
    return opening2


########################################################################################


def process_image(image, key: str, strip_metadata:dict=None, metadata:dict=None):
    """ 
        This function takes an image and processes it using the specified method. Here 
        we need to define the helper methods we use for each of the steps.
    """ 
    if metadata is None:
        raise ValueError("Metadata must be provided")
    
    # pre-process the image with equalization and blurring
    if metadata["equalize"]:
        image = equalize_image(image, clip_low=0, clip_high=255)
    if metadata["blur"]:
        image = gauss_blur(image, ksize=(5, 5), sigmaX=0)
        
    # threshold the image using the specified threshold function
    if metadata["threshold_func"] is None:
        raise ValueError("No threshold function provided")
    
    thresholded = metadata["threshold_func"](image, metadata)
    
    # apply the morphological filter using the specified function
    if metadata["morph_filter_func"] is None:
        raise ValueError("No morphological filter function provided")
    
    morphed =  metadata["morph_filter_func"](thresholded, metadata["morph_args"])
    
    overlayed = add_membrane_overlay(image, morphed)
    
    metadata2 = metadata.copy()
    
    metadata["key"] = key
    metadata2["image"] = image
    metadata2["thresholded"] = thresholded
    metadata2["morphed"] = morphed
    metadata2["overlayed"] = overlayed
    metadata2['metadata'] = strip_metadata
    return metadata2

############################# Helper functions #########################################
# import the product function from itertools
from itertools import product

#thresh, maxval, offset, c, blocksize
# import the product function from itertools
def create_th_dict(threshold_key: str, morph_key: str, equalize: bool, blur: bool, thresh: int, maxval: int,  blocksize: int, c: int, threshold_func, morph_filter_func, morph_args: dict, offset_value: int = 1.0):
    return {
            "method": threshold_key,
            "morph_key": morph_key,
            "equalize": equalize,
            "blur": blur,
            "thresh": thresh,
            "maxval": maxval,
            "blocksize": blocksize,
            "c": c,
            "threshold_func": threshold_func,
            "morph_filter_func": morph_filter_func,
            "morph_args": morph_args,
            "offset" : offset_value
    }

def create_threshold_metadata(base_path: str, image_dir: str, roi_metadata_path: str, options: dict):

    if options is None:
        raise ValueError("Options must be provided")

    mask_path = f"{base_path}/masks/"
    th_comparison_path = f"{base_path}/threshold_comparison/"

    # load the roi metadata
    roi_metadata = pandas.read_csv(roi_metadata_path)
    
    # create a dictionary of dictionaries, where the key is the method_key+index
    all_th_metadata = {}
    simple_th_metadata = {}
    adaptive_th_metadata = {}
    otsu_offset_th_metadata = {}

    ##################### generate th_dicts for various combos #########################

    threshold_functions = {"otsuOffset": otsu_then_offset_threshold,
                           "otsu": otsu_threshold,
                           "simple": simple_threshold,
                           "trunc": trunc_threshold,
                           "tozero": tozero_threshold,
                           "adaptive": adaptive_threshold,
                           "adaptive_gaussian": adaptive_gaussian_threshold}



    morph_filter_functions = {"closeOpen": morph_close_then_open}
    
    # create a list of all the combinations of the thresholding methods
    otsu_offset_combinations = product(options['th_method_keys'], options['morph_keys'], options['opts_equalize'], options['opts_blur'], options['thresh_levels'], options['maxval_levels'], options['blocksize_levels'], options['c_levels'], options['morph_args'], options['offset_values'])


    for i, (th_key, morph_key, equalize, blur, thresh, maxval, blocksize, c, morph_arg, offset) in enumerate(otsu_offset_combinations):

        th_dict = create_th_dict(th_key, morph_key, equalize, blur, thresh, maxval, blocksize, c, threshold_functions[th_key], morph_filter_functions[morph_key], morph_arg, offset)
        
        if th_dict is not None:
            th_dict["key"] = f"{i}_{th_key}"
            otsu_offset_th_metadata[th_dict["key"]] = th_dict
        else:
            
            print(f"Skipping {i}_{th_key}")

    print(f"Created {len(otsu_offset_th_metadata)} otsu offset thresholding metadata dictionaries")
    
    ######################## combine the three metadata dicst into all_th_data ##############################################
    # all_th_metadata = {**simple_th_metadata, **adaptive_th_metadata, **otsu_offset_th_metadata}
    all_th_metadata = otsu_offset_th_metadata
    print(f"Created {len(all_th_metadata)} total thresholding metadata dictionaries")
    
    ###################################################################################################################
    # can we create a dataframe of the thresholding metadata? We need to convert the dictionary to a dataframe
    all_th_metadata_df = pandas.DataFrame.from_dict(all_th_metadata, orient='index')
    
    # ,method,morph_key,thresh,maxval,threshold_func,threshold_name,morph_filter_func,morph_args,morph_filter_name,offset
    # remove the threshold_func, morph_filter_func
    all_th_metadata_df = all_th_metadata_df.drop(columns=["threshold_func", "morph_filter_func"])
    # column 1 should be renamed to method

    print(f"columns {all_th_metadata_df.columns}")
    all_th_metadata_df.to_csv(f"{base_path}/threshold_metadata.csv", index=True)

    return all_th_metadata

# create_masks(mask_path=mask_path, image_dir=image_dir, roi_metadata_path=roi_metadata_path, all_th_metadata=th_metadata)
def create_masks(base_path: str, image_dir: str, roi_metadata_path: str, all_th_metadata: dict):

    roi_metadata = pandas.read_csv(roi_metadata_path)

    mask_path = f"{base_path}/masks/"
    th_comparison_path = f"{base_path}/threshold_comparison/"

    # ensure the directory exists for the processed images and mask output
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(th_comparison_path, exist_ok=True)

    ########################### method comparison ###########################

    # th_metadata combo, do the thing to process
    for key, th_metadata in all_th_metadata.items():
        print(f"Processing with method: {key}")

        for strip_num in roi_metadata['strip'].unique():
            strip_filename = roi_metadata[roi_metadata['strip'] == strip_num]['strip_filename'].values[0]

            print(f"    - image: {os.path.basename(strip_filename.strip(".png"))}")

            # isolate the dataframe row that corresponds to the strip number
            strip_metadata = roi_metadata[roi_metadata['strip'] == strip_num]

            image = load_gray_image(f"{image_dir}/{strip_filename}")
            titles = ["Original","Thresholded", "Morphed", "Isolated Membrane"]
            
            # process the image
            plot_images(base_path=base_path, image_dict=process_image(image, key, strip_metadata, th_metadata), titles=titles, display_image=False)

    print(f"Finished processing masks and method_comparison for strips")