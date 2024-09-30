
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

def get_subdirectories(directory):
    return [f.path for f in os.scandir(directory) if f.is_dir()]

def get_process_names(directory):
    return [os.path.basename(f) for f in get_subdirectories(directory)]

def get_original_filename(image_name, metadata_filename):
    """
        Search the metadata file for the original filename of the image that the strip was taken from, and return it.
    """
    image_metadata = pd.read_csv(metadata_filename)
    
    image_name = os.path.normpath(image_name)
    
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

def create_image_dict(image_path: str, trial_number: int,  process_name: str, metadata:dict) -> dict:
    """ retrieve the images for the given image name, trial number and process name, and return them in a dictionary """
        
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    mask = cv2.imread(f"./output/trial_{trial_number}/masks/{process_name}/{os.path.basename(image_path)}", cv2.IMREAD_GRAYSCALE)

    # output\trial_2\processed_images\1_otsuOffset\membrane\strip_101.png
    membrane_image = cv2.imread(f"./output/trial_{trial_number}/processed_images/{process_name}/membrane/{os.path.basename(image_path)}", cv2.IMREAD_GRAYSCALE)
    
    # output\trial_2\processed_images\1_otsuOffset\lumen\strip_101.png
    lumen_image = cv2.imread(f"./output/trial_{trial_number}/processed_images/{process_name}/lumen/{os.path.basename(image_path)}", cv2.IMREAD_GRAYSCALE)
    
    convert_dict = get_image_conversion_factors(image_path, metadata["conversion_df_filename"], metadata["metadata_filename"])

    strip_name = os.path.basename(image_path).split(".png")[0]

    image_dict = {
        "strip_name": strip_name,
        "image_name": image_path,
        "image": image,
        "mask": mask,
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
    print(f"len(peaks): {len(peaks)}, pre-filtering")
    
    if peaks.size == 0:
        print(f"No peaks found for {image_name}")
        return None

    avg_peak_height = np.mean(histogram[peaks])
    half_height = avg_peak_height * chosen_height

    # we calculate the peak width at the half height of each peak, not at the 
    chosen_rel_height = half_height / avg_peak_height

    # # peaks, _ = find_peaks(histogram, distance=10, width=[2, 15], rel_height=chosen_rel_height)
    # peaks, _ = find_peaks(histogram, distance=10, width=[2, 15], rel_height=avg_peak_height * 0.9)
    # print(f"len(peaks): {len(peaks)}, post-filtering")

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
    widths, width_heights, left_ips, right_ips = peak_widths(histogram, peaks, rel_height=0.50)

    peak_data["peaks"] = peaks
    peak_data["histogram"] = histogram
    peak_data["avg_peak_height"] = avg_peak_height
    peak_data["half_height"] = half_height
    peak_data["chosen_rel_height"] = chosen_rel_height
    peak_data["widths"] = widths
    peak_data["width_heights"] = width_heights
    peak_data["left_ips"] = left_ips
    peak_data["right_ips"] = right_ips
    peak_data["num_peaks"] = len(peaks)
    
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


# "lumen_data": calculate_lumen_data(membrane_data, lumen_data),
def calculate_lumen_data(membrane_data: dict, lumen_data: dict) -> dict:
    """
        The Lumen data is not working well, but the membrane data is great. So lets use
        the right and left ips from the membrane data and then calculate the repeat
        distance, lumen width, etc from that.
    """

    lumen_data = {}
    
    if membrane_data["right_ips"] is None or membrane_data["left_ips"] is None:
        print(f"No membrane data for {image_name}")
        return None
    
    return lumen_data


def calculate_repeat_distance(membrane_data:dict) -> float:
    """
        Calculate the distance between the membrane peaks in nm. We need to first calculate the
        float value for the center of the peaks, then calculate the difference between the peaks.
    """
    
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
            "mask": mask,
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
            "num_membrane": len(membrane_data["peaks"]),
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
    
    
def filter_extrawide_membrane_widths(grana_values: dict, width_th: float = 16) -> dict:
    """
        See that while the lumen histogram comes pretty close with the lumen width 
        calculation, the membrane histogram measures the membrane width at our desired 
        height. It just so happens that it merges with another membrane at that height, 
        resulting in both peaks getting a double-width value for their membrane width.

        If I instead take the left and right edges of the lumen, I can instead make a good
        guess as to what the membrane width would be ( as the membrane + gap + membrane 
        would be the distance between them). I have created a method that runs through 
        the membrane widths, and if they are both larger than 12px (or whatever threshold
        I desire) and identical to another membrane width in the set (indicated one of 
        these COMBO membranes widths), it recalculates the membrane width using the lumen 
        data.

        I think this is going to be a more accurate method for calculating the widths, 
        and only kicks in if the membrane widths are way larger than expected. This is 
        an additional level of "common-sense" added into the automation.
        
        Example exported data showing the membrane_width:
        strip,image_name,x_per_nm,nm_per_px,scale,scale_pixels,grana_height,membrane_width,membrane_width_heights,peaks,type,index,process
        135,"",1.095,0.91324200913242,200,219,73.25,8.785714285714285,7650.0,27,membrane,0,4_otsuOffset
        135,"",1.095,0.91324200913242,200,219,73.25,12.415686274509802,7650.0,42,membrane,1,4_otsuOffset
        135,"",1.095,0.91324200913242,200,219,73.25,29.075255102040813,7650.0,61,membrane,2,4_otsuOffset
        135,"",1.095,0.91324200913242,200,219,73.25,29.075255102040813,7650.0,76,membrane,3,4_otsuOffset
        135,"",1.095,0.91324200913242,200,219,73.25,8.13095238095238,7650.0,90,membrane,4,4_otsuOffset
        
        So in this case, we check the membrane_widths, and if we find two that are the same,
        and that they are over the given threshold (example above is 16, which is 1.5* the
        expected membrane width), we will need to recalculate the membrane width using the
        lumen data.
        
        What is in grana_values:
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
        
        We can access the required data from the grana_values dict:

        
    """

    
    membrane_peaks = grana_values["membrane_data"]["peaks"]
    left_ips = grana_values["membrane_data"]["left_ips"]
    right_ips = grana_values["membrane_data"]["right_ips"]
    membrane_widths = grana_values["membrane_width"]
    
    # iterate through the 
    for i, (left_ip, right_ip, peak, membrane_width) in enumerate(zip(left_ips, right_ips, membrane_peaks, membrane_widths)):
        if i < len(membrane_widths) - 1: # don't go out of bounds
            if membrane_width > width_th and membrane_width == membrane_widths[i+1]:
                print(f"Recalculating membrane width for {grana_values['image_name']}")
                print(f"Membrane width: {membrane_width}, left_ip: {left_ip}, right_ip: {right_ip}")
                left_width, right_width = split_membrane(i, grana_values["membrane_data"], grana_values["lumen_data"], left_ip, right_ip)
                
                if left_width == 0 and right_width == 0:
                    print(f"Error recalculating membrane width for {grana_values['image_name']}")
                    continue
                
                # update the membrane_widths list
                membrane_widths[i] = left_width
                membrane_widths[i+1] = right_width    
                
                print(f"New membrane widths: {left_width}, {right_width}")
            
    grana_values["membrane_width"] = membrane_widths
    
    return grana_values
    
def split_membrane(i:int, membrane_data: dict, lumen_data: dict, left_ip: float, right_ip: float) -> (float, float):
    """ 
        Recalculate the membrane width using the lumen data.
        
        The i is the membrane width index of the leftmost membrane peak of the pair. 
        
        The left_ip and right_ip are the left and right ips of the combined membrane peaks.
        We will need to get the lumen with the index value correspodning to the leftmost membrane peak,
        and tet its left and right ips.
        
        The lumen_left_ip - left_ip will give us the left width, and the right_ip - lumen_right_ip will
        give us the right width.
    """
    lumen_left_ip = lumen_data["left_ips"][i]
    lumen_right_ip = lumen_data["right_ips"][i]
    
    
    # assert that left_ip is less than lumen_left_ip, and right_ip is greater than lumen_right_ip
    try:
        assert left_ip < lumen_left_ip, f"left_ip: {left_ip}, lumen_left_ip: {lumen_left_ip}"
        assert right_ip > lumen_right_ip, f"right_ip: {right_ip}, lumen_right_ip: {lumen_right_ip}"
    except:
        print(f"Error in split_membrane for {grana_values['image_name']}")
        return (0, 0)
    
    left_width = lumen_left_ip - left_ip
    right_width = right_ip - lumen_right_ip

    return left_width, right_width
    
def main(trial_number:int):
    
    base_path = f"./output/trial_{trial_number}"
    process_names = get_process_names(f"{base_path}/processed_images")
    roi_directory = f"{base_path}/rois"
    conversion_df_filename = "./metadata/image_scale_conversion.csv"
    metadata_filename = f"{base_path}/081624_rois_metadata_bignine.csv"
    processed_image_path = f"{base_path}/processed_images"
        

    process_names = os.listdir(processed_image_path)
    if (process_names is None) or (len(process_names) == 0):
        print("No processes found")
        exit(1)

    errors_in_processing = []
    

    
    for process in process_names:
        
        search_str = f"{os.path.join(processed_image_path, process, "membrane")}/*.png"
        print(f"Searching for images in {search_str}")
        
        image_list = [os.path.normpath(f) for f in glob.glob(search_str)]
        
    # output\trial_2\processed_images\0_otsuOffset\membrane\strip_101.png
    
        
        if (len(image_list) == 0):
            print(f"No images found for {process}")
            continue
    
        metadata = {
            "trial_number": trial_number,
            "process_name": process,
            "conversion_df_filename" : conversion_df_filename,
            "metadata_filename" : metadata_filename,
            "images": image_list,
            "chosen_height": 0.5,
            }
        
        output_directory = f"{processed_image_path}/{process}"
        os.makedirs(f"{output_directory}/histograms/", exist_ok=True)

        all_grana_data= []

        if (len(metadata["images"]) == 0):
            print(f"No images found for {process}")
            continue

        for image_number, image_name in enumerate(metadata["images"]):
            
            # load the images in their dict: image, mask, lumen_image, membrane_image    
            image_dict = create_image_dict(image_path=image_name, trial_number=trial_number, process_name=process, metadata=metadata)

            membrane_data = calculate_peak_data(image_dict, metadata, peak_type="membrane")
            lumen_data = calculate_peak_data(image_dict, metadata, peak_type="lumen")
                        
            if membrane_data is None or lumen_data is None:
                print(f"Error processing peak data for process: {process} on image: {image_name}")
                continue            
            
            grana_values = calculate_grana_values(membrane_data, lumen_data, image_dict)
            
            grana_values = filter_extrawide_membrane_widths(grana_values)
    
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

        # add the peaks from membrane_data and lumen_data to the dataframe
        grana_df["membrane_peaks"] = grana_df["membrane_data"].apply(lambda x: x["peaks"])
        grana_df["lumen_peaks"] = grana_df["lumen_data"].apply(lambda x: x["peaks"])
        
        # # create a "membrane ips" column, that is a list of tuples of the left and right ips
        # grana_df["m_ips"] = grana_df["membrane_data"].apply(lambda x: list(zip(x["left_ips"], x["right_ips"])))
        # grana_df["l_ips"] = grana_df["lumen_data"].apply(lambda x: list(zip(x["left_ips"], x["right_ips"])))
        
        # drop image_dict, lumen_data, and membrane_data
        grana_df = grana_df.drop(columns=["image_dict", "lumen_data", "membrane_data"])

        # add a column for identifying the membrane or lumen type
        grana_df["type"] = "-"

        ########################    membranes only     #################################
        try:
            # drop repeat distance from the membrane data
            membrane_df = grana_df.drop(columns=["repeat_distance"])

            # drop all columens with "lumen" in the name
            membrane_df = membrane_df[membrane_df.columns.drop(list(membrane_df.filter(regex='lumen')))]
            
            # set the type for each dataframe
            membrane_df["type"] = "membrane"
            
            # rename 'membrane_peaks' to 'peaks'
            membrane_df = membrane_df.rename(columns={"membrane_peaks": "peaks"})
            
            # Add an 'index' column to track the order of each item in the 'membrane_width' list
            membrane_df["index"] = membrane_df.apply(lambda row: list(range(len(row["membrane_width"]))), axis=1)

            # Explode the membrane_df based on 'membrane_width' and the new 'membrane_index' column
            membrane_df = membrane_df.explode(["membrane_width", "peaks","membrane_width_heights", "index"])
        
            # add the process as a column to the dataframe
            membrane_df["process"] = process
            print(f"membrane_df.shape after explode: {membrane_df.shape}")
            # now filter them according to whether they are inner or outer membranes.
            # To be an inner membrane, the index of the membrane row should be greater than 0, and less than the "num_membrane" value
            # To be an outer membrane, the index of the membrane row equal to 0, or the max index for that strip
            # does num_membrane exist in the dataframe?
            if "num_membrane" not in membrane_df.columns:
                print(f"No num_membrane column in membrane_df")
                continue
            
            is_inner = membrane_df.apply(lambda row: row["index"] > 0 and row["index"] < (row["num_membrane"] -1), axis=1)
            
            # how do I add this to the dataframe?
            membrane_df["membrane_type"] = np.where(is_inner, "inner", "outer")
            
            print(f"membrane_df.shape after filtering: {membrane_df.shape}")
            
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
            
            # rename 'lumen_peaks' to 'peaks'
            lumen_df = lumen_df.rename(columns={"lumen_peaks": "peaks"})
            
            # Add an 'index' column to track the order of each item in the 'lumen_width' list before exploding
            lumen_df["index"] = lumen_df.apply(lambda row: list(range(len(row["lumen_width"]))), axis=1)

            # lumen_df.columns: Index(['strip', 'image_name', 'px_per_nm', 'nm_per_px', 'scale',
            #        'scale_pixels', 'grana_height', 'num_lumen', 'repeat_distance',
            #        'lumen_width', 'lumen_width_heights', 'peaks', 'type', 'index'],
            #       dtype='object')
            lumen_df.to_csv(f"{output_directory}/tmp_lumen.csv", index=False)
            
            lumen_df = lumen_df.explode(["lumen_width", "peaks", "repeat_distance", "lumen_width_heights","index"])
        
            # add the process as a column to the dataframe            
            print(f"lumen_df.shape after explode: {lumen_df.shape}")

        
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
    main(trial_number=2)
