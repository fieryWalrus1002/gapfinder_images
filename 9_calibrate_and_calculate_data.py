
import pandas
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from scipy.signal import find_peaks
from scipy.signal import peak_widths
import pandas as pd
import os
# strip,grana_height,grana_height_nm,num_lumen,repeat_distance,repeat_distance_nm,px_per_nm,nm_per_px, scale, scale_pixels
# Include the scale and scale_pixels in the data export. 
# Also, include for each lumen and membrane: width, 
import pandas as pd
import numpy as np


import pandas
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from scipy.signal import find_peaks
from scipy.signal import peak_widths
import pandas as pd
import os
# strip,grana_height,grana_height_nm,num_lumen,repeat_distance,repeat_distance_nm,px_per_nm,nm_per_px, scale, scale_pixels
# Include the scale and scale_pixels in the data export. 
# Also, include for each lumen and membrane: width, 
import pandas as pd
import numpy as np

def get_image_list(directory:str, image_type:str) -> list:
    filenames = glob.glob(f"{directory}/*.{image_type}")
    
    return [os.path.normpath(f) for f in filenames]

def get_processed_image(image_path, trial, process_name, invert = False):
    filename = os.path.basename(image_path)
    image =  cv2.imread(f"./output/trial_{trial}/processed_images/masks/{process_name}/{os.path.basename(image_path)}", cv2.IMREAD_GRAYSCALE)
    
    if invert:
        image = cv2.bitwise_not(image)
        
    return image

def get_subdirectories(directory):
    return [f.path for f in os.scandir(directory) if f.is_dir()]

def get_process_names(directory):
    return [os.path.basename(f) for f in get_subdirectories(directory)]

def load_images_for_given_process(image_path, trial_number, process_name):
    """
        Load images from the given filenames. Returns raw image and the processed image.
        
        The processed image is inverted so that the lumen/stroma is black and the membrane as white.
        This is to aid in the contouring process, which sees the area of the image as the area within the contour.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    try:
        p_image = cv2.imread(f"./output/trial_{trial_number}/masks/{process_name}/{os.path.basename(image_path)}", cv2.IMREAD_GRAYSCALE)
    except:
        p_image = None

    if p_image is None:
        print(image_path)
        raise ValueError(f"Processed image not found for {image_path}")

    return image, cv2.bitwise_not(p_image)

def get_original_filename(image_name, metadata_filename):
    """
        Search the metadata file for the original filename of the image that the strip was taken from, and return it.
    """
    image_metadata = pd.read_csv(metadata_filename)
    
    image_name = os.path.normpath(image_name)
    print(image_name)
    
    #output\trial_1\rois\strip_101.png
    image_name = os.path.basename(image_name)
    
    #strip_101.png
    strip_number = int(image_name.strip(".png").split("_")[1])
    
    
    # if image_metadata is none, thorw an error
    if image_metadata is None:
        raise ValueError("Image metadata is None")
    
    image_df = image_metadata[image_metadata['strip'] == strip_number].to_dict(orient='records')[0]
    return image_df["filename"]


def get_image_conversion_factors(image_name:str, conversion_df_filename: str, metadata_filename: str) -> dict:
    """ 
        Returns the dict with the nm_per_pixel and pixel_per_nm values for the given image name.
    """

    image_raw_filename = get_original_filename(image_name, metadata_filename)

    conversion_df = pd.read_csv(conversion_df_filename)
    
    conversion_df['filename'] = conversion_df['filename'].map(os.path.normpath)
    
    filename = os.path.normpath(image_raw_filename)#
    
    image_conversion_factors = conversion_df[conversion_df['filename'] == filename].to_dict(orient='records')[0]

    return {"nm_per_pixel": image_conversion_factors['nm_per_pixel'], 
            "pixel_per_nm": image_conversion_factors['pixel_per_nm'], 
            "scale": image_conversion_factors['scale'],
            "scale_pixels": image_conversion_factors['scale_pixels']}


def convert_nm_to_pixel(nm_value, nm_per_pixel):
    return nm_value / nm_per_pixel

def convert_pixel_to_nm(pixel_value, pixel_per_nm):
    return pixel_value / pixel_per_nm

def create_image_dict(image_name, metadata:dict) -> dict:
    """ retrieve the images for the given image name, trial number and process name, and return them in a dictionary """
        
    # p_image has the lumen/stroma as black and the membrane as white. We want to extract the membrane
    image, p_image = load_images_for_given_process(image_name, metadata["trial_number"], metadata["process_name"])

    # start with a black image for saving the membrane contours to
    membrane_image = np.zeros_like(p_image)    
    
    # create the contours based on the processed image
    membrane_contours = get_filtered_contours(p_image, min_area=100, max_area=np.Infinity)
    
    # draw the contours on the image
    cv2.drawContours(membrane_image, membrane_contours, -1, (255, 0, 0), -1)
    
    # invert the image so that the membrane is white and the lumen/stroma is black
    lumen_image = cv2.bitwise_not(membrane_image)

    convert_dict = get_image_conversion_factors(image_name, metadata["conversion_df_filename"], metadata["metadata_filename"])

    strip_name = os.path.basename(image_name).split(".png")[0]

    image_dict = {
        "strip_name": strip_name,
        "image_name": image_name,
        "image": image,
        "p_image": p_image,
        "lumen": lumen_image,
        "membrane": membrane_image,
        "nm_per_pixel": convert_dict["nm_per_pixel"],
        "pixel_per_nm": convert_dict["pixel_per_nm"],
        "scale": convert_dict["scale"],
        "scale_pixels": convert_dict["scale_pixels"],
    }
    
    return image_dict


def get_filtered_contours(image, min_area=0, max_area=np.Infinity, contour_method : int = cv2.RETR_EXTERNAL, contour_approximation : int = cv2.CHAIN_APPROX_SIMPLE):
    """
        Calculate the contours of the white regions of the image, then filter the results
        according to the given min and max area. Return the filtered contours.
        
    """
    
    if image is None:
        raise ValueError("Image is None")
    
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    contours, hierarchy = cv2.findContours(image, contour_method, contour_approximation)


    return [c for c in contours if min_area < cv2.contourArea(c) < max_area]

def calculate_peak_data(image_dict:dict, metadata: dict, peak_type: str = "membrane") -> dict:
    """
        Calculate the peaks and widths of the membrane histogram, and return the data in a dict.
    """
    image_name = image_dict["image_name"]
    nm_per_pixel = image_dict['nm_per_pixel']
    pixel_per_nm = image_dict['pixel_per_nm']
    input_image = image_dict[peak_type]
    chosen_height = metadata["chosen_height"]

    peak_data = {}

    histogram = np.sum(input_image, axis=1)

    peaks, _ = find_peaks(histogram)
    
    if peaks.size == 0:
        print(f"No peaks found for {image_name}")
        return None

    avg_peak_height = np.mean(histogram[peaks])
    half_height = avg_peak_height * chosen_height

    # we calculate the peak width at the half height of each peak, not at the 
    chosen_rel_height = half_height / avg_peak_height
    print(f"Chosen height: {chosen_height}, half height: {half_height}, avg peak height: {avg_peak_height}, chosen_rel_height: {chosen_rel_height}")

    peaks, _ = find_peaks(histogram, height=chosen_rel_height)

    # use a function to get width, width_height, left_ip, right ip for every individual peak
    # Take those and add them into the  lists: widths, width_heights, left_ips, right_ips
    # widths, width_heights, left_ips, right_ips = [], [], [], []
    
    # for p, peak in enumerate(peaks):
    #     # recalculate the peak width at the half height of each peak
    #     # this means we have to run peak_widths again, for each peak at the half height
    
    #     peak_width, width_height, left_ip, right_ip = peak_widths(histogram, [peak], rel_height=chosen_rel_height)
    #     print(f"Peak width: {peak_width}, width height: {width_height}, left ip: {left_ip}, right ip: {right_ip}")
        
    #     widths.append(peak_width[0])
    #     width_heights.append(width_height[0])
    #     left_ips.append(left_ip[0])
    #     right_ips.append(right_ip[0])
    widths, width_heights, left_ips, right_ips = peak_widths(histogram, peaks, rel_height=chosen_rel_height)

    peak_data["peaks"] = peaks
    peak_data["histogram"] = histogram
    peak_data["avg_peak_height"] = avg_peak_height
    peak_data["half_height"] = half_height
    peak_data["chosen_rel_height"] = chosen_rel_height
    peak_data["widths"] = widths
    peak_data["width_heights"] = width_heights
    peak_data["left_ips"] = left_ips
    peak_data["right_ips"] = right_ips
    
    return peak_data


def calculate_grana_height(membrane_data:dict) -> float:
    """
        Calculate the height of the grana stacks in px. Take the min of the left_ips and the
        max of the right_ips.
    """
        # Left ips: [24.36666667 34.17391304 48.41463415 62.39393939 77.22580645 91.818181[82]
    # Right ips: [29.16666667 43.40625    57.33333333 71.04166667 86.8        96.5       ]
    
    if membrane_data["right_ips"] is None or membrane_data["left_ips"] is None:
        print(f"No membrane data for {image_name}")
        return 0
    
    return np.max(membrane_data["right_ips"]) - np.min(membrane_data["left_ips"])


def calculate_repeat_distance(membrane_data:dict) -> float:
    """
        Calculate the distance between the membrane peaks in nm. We need to first calculate the
        float value for the center of the peaks, then calculate the difference between the peaks.
    """
    
    # todo: instead of calculating the diff from peaks, we
    # can calculate the diff from the left_ips and right_ips of the membrane. 
    # aget the center of the peaks using the left_ips and right_ips

    left_ips = membrane_data["left_ips"]
    right_ips = membrane_data["right_ips"]
    
    # zip together as pairs
    peaks = np.array(list(zip(left_ips, right_ips)))

    # the mean of each pair will get us the center of the peak. 
    center_points = np.mean(peaks, axis=1)
    
    # calculate the differences between the peaks
    repeat_distances = np.diff(center_points)
    
    # round them down to two significant figures
    repeat_distances = np.round(repeat_distances, 2)
    
    # flatten the numpy array into a list
    repeat_distances = repeat_distances.flatten()
        
    return repeat_distances

# calculate_values(membrane_data, lumen_data, image_dict, metadata)/
def calculate_grana_values(membrane_data: dict, lumen_data: dict, image_dict: dict) -> dict:
    """ 
        Take the peaks and widths of the membrane and lumen histograms, and return the grana values.
        peaks data:
            peak_data["peaks"] = peaks
            peak_data["histogram"] = histogram
            peak_data["avg_peak_height"] = avg_peak_height
            peak_data["half_height"] = half_height
            peak_data["chosen_rel_height"] = chosen_rel_height
            peak_data["widths"] = widths
            peak_data["width_heights"] = width_heights
            peak_data["left_ips"] = left_ips
            peak_data["right_ips"] = right_ips
        image_dict:
            "image_name": image_name,
            "image": image,
            "p_image": p_image,
            "lumen": lumen_image,
            "membrane": membrane_image,
            "nm_per_pixel": convert_dict["nm_per_pixel"],
            "pixel_per_nm": convert_dict["pixel_per_nm"],
            "scale": convert_dict["scale"],
            "scale_pixels": convert_dict["scale_pixels"],
    """
    # strip,grana_height,grana_height_nm,num_lumen,repeat_distance,repeat_distance_nm,px_per_nm,nm_per_px, scale, scale_pixels
    # Also, include for each lumen and membrane: width, 
    try:
        grana_values = {
            "strip": os.path.basename(image_dict["image_name"]).strip(".png").strip("strip_"),
            "image_name": image_dict["image_name"],
            "image_dict": image_dict,
            "px_per_nm": image_dict['pixel_per_nm'],
            "nm_per_px": image_dict['nm_per_pixel'],
            "scale": image_dict['scale'],
            "scale_pixels": image_dict['scale_pixels'],
            "membrane_data": membrane_data,
            "lumen_data": lumen_data,
            "grana_height": calculate_grana_height(membrane_data),
            "num_lumen": len(lumen_data["peaks"]),
            "repeat_distance": calculate_repeat_distance(membrane_data),
            "lumen_width": lumen_data["widths"],
            "membrane_width": membrane_data["widths"],
            "lumen_width_heights": lumen_data['width_heights'],
            "membrane_width_heights": membrane_data['width_heights'],
        }
    except:
        print(f"Error calculating grana values for {image_dict['image_name']}")
        return None
    
    return grana_values


# def export_grana_values(grana_values: dict) -> dict:
    
#     if grana_values is None:
#         raise ValueError("grana_values is None")
    
#     return {
#         "strip": grana_values['strip_num'],
#         "grana_height" : grana_values['grana_height'],
#         "num_lumen": grana_values['num_lumen'],
#         "repeat_distance": grana_values['repeat_distance'],
#         "px_per_nm": grana_values['image_dict']['pixel_per_nm'],
#         "nm_per_px": grana_values['image_dict']['nm_per_pixel'],
#         "scale": grana_values['image_dict']['scale'],
#         "scale_pixels": grana_values['image_dict']['scale_pixels'],
#         "lumen_width": grana_values['lumen_width'],
#         "membrane_width": grana_values['membrane_width'],
#         "lumen_width_height": grana_values['lumen_data']['width_heights'],
#         "membrane_width_height": grana_values['membrane_data']['width_heights'],
#     }

def plot_histogram(grana_data: dict, metadata:dict, output_directory:str, peak_type: str = "membrane", display: bool = False):

    image_dict = grana_data["image_dict"]
    membrane_data = grana_data["membrane_data"]
    lumen_data = grana_data["lumen_data"]
    strip_name = image_dict["strip_name"]
    process_name = metadata["process_name"]
    trial_number = metadata["trial_number"]
    
    if peak_type == "membrane":
        peaks = membrane_data["peaks"]
        histogram = membrane_data["histogram"]
        half_height = grana_data["membrane_width_heights"]
        left_ips = membrane_data["left_ips"]
        right_ips = membrane_data["right_ips"]
        
    else:
        peaks = lumen_data["peaks"]
        histogram = lumen_data["histogram"]
        half_height = grana_data["lumen_width_heights"]
        left_ips = lumen_data["left_ips"]
        right_ips = lumen_data["right_ips"]

    ################ Plot it ################
    # plot the histogram, but make sure the background is white
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150, facecolor='w')
    plt.plot(histogram)

    # plot the peaks, and then a green line dropping down
    plt.plot(peaks, histogram[peaks], "x")
    for peak in peaks:
        plt.plot([peak, peak], [0, histogram[peak]], "--g")
        
    # use the left and right interpolated points to plot the width of the peak
    for peak_left_ip, peak_right_ip, peak_half_height in zip(left_ips, right_ips, half_height):
        plt.plot([peak_left_ip, peak_right_ip], [peak_half_height, peak_half_height], "-r")
        
        # put a small red x at the peak_left_ip and peak_right_ip at the half height
        plt.plot(peak_left_ip, peak_half_height, "xr")
        plt.plot(peak_right_ip, peak_half_height, "xr")

    plt.title(f"{peak_type} Histogram\n{strip_name}")
    plt.savefig(f"{output_directory}/histograms/{strip_name}_{peak_type}_histogram.png")
    
    if display:
        plt.show()
    else:
        plt.close()


def explode_grana_data(df) -> (pd.DataFrame, pd.DataFrame):
        """
        Explode the grana data! We need one row per lumen_width and repeat_distance in one df (dropping the membrane_width),
        and one row per membrane_width in another df (dropping the lumen_width and repeat_distance).
        """
        
        # Split the dataframes into two separate dataframes
        membrane_df = df.drop(columns=["lumen_width", "repeat_distance", "repeat_distance_nm"])
        lumen_df = df.drop(columns=["membrane_width", "repeat_distance_nm"])
        
        # Explode the membrane_df based on 'membrane_width'
        membrane_df = membrane_df.explode("membrane_width")
        
        # Zip the lumen_width and repeat_distance columns together as a list of tuples
        lumen_df["lumen_repeat_pairs"] = list(zip(lumen_df["lumen_width"], lumen_df["repeat_distance"]))
        
        # Explode the 'lumen_repeat_pairs' column into separate rows
        lumen_df = lumen_df.explode("lumen_repeat_pairs")
        
        # After exploding, split the tuple back into two columns
        lumen_df["lumen_width"], lumen_df["repeat_distance"] = zip(*lumen_df["lumen_repeat_pairs"])
        
        # Drop the temporary 'lumen_repeat_pairs' column
        lumen_df = lumen_df.drop(columns=["lumen_repeat_pairs"])
        
        return membrane_df, lumen_df

def main():
    
    selected_trial = 1
    base_path = f"./output/trial_{selected_trial}"
    process_names = get_process_names(f"{base_path}/masks")
    roi_directory = f"{base_path}/rois"
    conversion_df_filename = "./metadata/image_scale_conversion.csv"
    metadata_filename = f"{base_path}/081624_rois_metadata_bignine.csv"
    processed_image_path = f"{base_path}/processed_images"
    # process = '107_ostuOffset'
        
    process_names = ["106_otsuOffset", "107_otsuOffset", "126_otsuOffset", "127_otsuOffset", "146_otsuOffset", "147_otsuOffset"]
    
    errors_in_processing = []
    
    for process in process_names:
        
        metadata = {
            "trial_number": selected_trial,
            "process_name": process,
            "conversion_df_filename" : conversion_df_filename,
            "metadata_filename" : metadata_filename,
            "images": get_image_list(directory=roi_directory, image_type="png"),
            "chosen_height": 0.5,
            }
        
        print(f"Process: {process}")

        output_directory = f"{processed_image_path}/{process}"
        print(f"Creating output directory {output_directory}")
        os.makedirs(f"{output_directory}/histograms/", exist_ok=True)

        all_grana_data= []

        for image_number, image_name in enumerate(metadata["images"]):

            # load the images in their dict: image, p_image, lumen_image, membrane_image    
            image_dict = create_image_dict(image_name, metadata)

            membrane_data = calculate_peak_data(image_dict, metadata, peak_type="membrane")
            lumen_data = calculate_peak_data(image_dict, metadata, peak_type="lumen")
            
            if membrane_data is None or lumen_data is None:
                print(f"Error processing peak data for process: {process} on image: {image_name}")
                continue
            
            grana_values = calculate_grana_values(membrane_data, lumen_data, image_dict)
            
            if grana_values is not None:
                all_grana_data.append(grana_values)
            else:
                # print(f"Error processing grana values for: {image_name}")
                errors_in_processing.append(f"Error processing grana values for: {process} : {image_name}")
                continue
                
            plot_histogram(grana_values, metadata, output_directory, peak_type="membrane")
            plot_histogram(grana_values, metadata, output_directory, peak_type="lumen")    
            

        # check to see if the all_grana_data is empty
        if len(all_grana_data) == 0:
            # print(f"No grana data found for {process}")
            errors_in_processing.append(f"No grana data found for {process}")
            continue
        else:
            print(f"Found {len(all_grana_data)} grana data items for {process}")

            
        # create a dataframe from the grana data
        grana_df = pd.DataFrame(all_grana_data)

        # Create a dataframe from the grana data
        grana_df = pd.DataFrame(all_grana_data)

        # drop image_dict, lumen_data, and membrane_data
        grana_df = grana_df.drop(columns=["image_dict", "lumen_data", "membrane_data"])

        # add a column for identifying the membrane or lumen type
        grana_df["type"] = "-"

        ########################    membranes only     #################################
        try:
            # Split the dataframes into two separate dataframes
            membrane_df = grana_df.drop(columns=["lumen_width", "repeat_distance", "lumen_width_heights"])

            # set the type for each dataframe
            membrane_df["type"] = "membrane"

            # Add an 'index' column to track the order of each item in the 'membrane_width' list
            membrane_df["index"] = membrane_df.apply(lambda row: list(range(len(row["membrane_width"]))), axis=1)

            # Explode the membrane_df based on 'membrane_width' and the new 'membrane_index' column
            membrane_df = membrane_df.explode(["membrane_width", "membrane_width_heights", "index"])

            # add the process as a column to the dataframe
            membrane_df["process"] = process
        except:
            errors_in_processing.append(f"Error processing membrane data for {process}")
            continue

        ############################## lumen only ######################################
        try:
            # Drop the 'membrane_width' column as not needed here
            lumen_df = grana_df.copy()
            lumen_df["type"] = "lumen"

            # drop all columens with "membrane" in the name
            lumen_df = lumen_df[lumen_df.columns.drop(list(lumen_df.filter(regex='membrane')))]
    
            # Add an 'index' column to track the order of each item in the 'lumen_width' list before exploding
            lumen_df["index"] = lumen_df.apply(lambda row: list(range(len(row["lumen_width"]))), axis=1)
        
            # Explode the membrane_df based on 'membrane_width' and the new 'membrane_index' column

            lumen_df = lumen_df.explode(["lumen_width", "repeat_distance", "lumen_width_heights","index"])
        except:
            errors_in_processing.append(f"Error processing lumen data for {process}")
            continue
        
        ####################   save them both   ########################################

        #save it to csv
        print(f"Saving membrane data to {output_directory}/grana_data_membrane.csv")
        membrane_df.to_csv(f"{output_directory}/grana_data_membrane.csv", index=False)

        print(f"Saving lumen data to {output_directory}/grana_data_lumen.csv")
        lumen_df["process"] = process
        lumen_df.to_csv(f"{output_directory}/grana_data_lumen.csv", index=False)

    for error_process in errors_in_processing:
        print(error_process)

if __name__ == "__main__":
    main()
