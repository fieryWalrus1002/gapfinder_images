import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from scipy.signal import find_peaks
from scipy.signal import peak_widths
import os
import pandas as pd


def get_image_list(directory: str, image_type: str) -> list:
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

    image_name = os.path.basename(image_name)

    strip_number = int(image_name.strip(".png").split("_")[1])

    if image_metadata is None:
        raise ValueError("Image metadata is None")

    image_df = image_metadata[image_metadata["strip"] == strip_number].to_dict(
        orient="records"
    )[0]
    return image_df["filename"]


def get_image_conversion_factors(
    image_name: str, conversion_df_filename: str, metadata_filename: str
) -> dict:
    """
    Returns the dict with the nm_per_pixel and pixel_per_nm values for the given image name.
    """

    image_raw_filename = get_original_filename(image_name, metadata_filename)

    conversion_df = pd.read_csv(conversion_df_filename)

    conversion_df["filename"] = conversion_df["filename"].map(os.path.normpath)

    filename = os.path.normpath(image_raw_filename)  #

    image_conversion_factors = conversion_df[
        conversion_df["filename"] == filename
    ].to_dict(orient="records")[0]

    return {
        "nm_per_pixel": image_conversion_factors["nm_per_pixel"],
        "pixel_per_nm": image_conversion_factors["pixel_per_nm"],
        "scale": image_conversion_factors["scale"],
        "scale_pixels": image_conversion_factors["scale_pixels"],
    }


def convert_nm_to_pixel(nm_value, nm_per_pixel):
    return nm_value / nm_per_pixel


def convert_pixel_to_nm(pixel_value, pixel_per_nm):
    return pixel_value / pixel_per_nm


def create_image_dict(
    image_path: str,
    mask_path: str,
    contour_base_path: str,
    process_name: str,
    metadata: dict,
) -> dict:
    """retrieve the images for the given image name, process name, and return them in a dictionary"""

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    mask = cv2.imread(
        f"{mask_path}/{process_name}/{os.path.basename(image_path)}",
        cv2.IMREAD_GRAYSCALE,
    )

    membrane_image = cv2.imread(
        f"{contour_base_path}/{process_name}/membrane/{os.path.basename(image_path)}",
        cv2.IMREAD_GRAYSCALE,
    )

    lumen_image = cv2.imread(
        f"{contour_base_path}/{process_name}/lumen/{os.path.basename(image_path)}",
        cv2.IMREAD_GRAYSCALE,
    )

    convert_dict = get_image_conversion_factors(
        image_path, metadata["conversion_df_filename"], metadata["metadata_filename"]
    )

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


def get_filtered_contours(
    image,
    min_area=0,
    max_area=np.inf,
    contour_method: int = cv2.RETR_EXTERNAL,
    contour_approximation: int = cv2.CHAIN_APPROX_SIMPLE,
):
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


def calculate_peak_data(
    image_dict: dict,
    metadata: dict,
    peak_type: str = "membrane",
    min_peak_distance: int = 10,
    min_peak_width: float = 5.0,
) -> dict:
    """
    Calculate the peaks and widths of the membrane histogram, and return the data in a dict.
    """
    image_name = image_dict["image_name"]
    nm_per_pixel = image_dict["nm_per_pixel"]
    pixel_per_nm = image_dict["pixel_per_nm"]
    input_image = image_dict[peak_type]
    chosen_height = metadata["chosen_height"]

    peak_data = {}

    histogram = np.sum(input_image, axis=1)

    peaks, _ = find_peaks(histogram, distance=min_peak_distance)

    if peaks.size == 0:
        print(f"No peaks found for {image_name}")
        return None

    avg_peak_height = np.mean(histogram[peaks])
    half_height = avg_peak_height * chosen_height

    chosen_rel_height = half_height / avg_peak_height

    widths, width_heights, left_ips, right_ips = peak_widths(
        histogram, peaks, rel_height=0.50
    )

    # filter out the peaks that have widths less than the min_peak_width
    valid_indices = np.where(widths >= min_peak_width)[0]
    peaks = peaks[valid_indices]
    widths = widths[valid_indices]
    width_heights = width_heights[valid_indices]
    left_ips = left_ips[valid_indices]
    right_ips = right_ips[valid_indices]

    if len(peaks) == 0:
        print(f"No peaks found for {image_name}")
        return None

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


def calculate_grana_height(membrane_data: dict) -> float:
    """
    Calculate the height of the grana stacks in px. Take the min of the left_ips and the
    max of the right_ips.
    """
    if membrane_data["right_ips"] is None or membrane_data["left_ips"] is None:
        print(f"No membrane data for {image_name}")
        return 0

    return np.max(membrane_data["right_ips"]) - np.min(membrane_data["left_ips"])


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


def calculate_repeat_distance(membrane_data: dict) -> float:
    """
    Calculate the distance between the membrane peaks in nm. We need to first calculate the
    float value for the center of the peaks, then calculate the difference between the peaks.
    """

    left_ips = membrane_data["left_ips"]
    right_ips = membrane_data["right_ips"]
    peaks = np.array(list(zip(left_ips, right_ips)))
    center_points = np.mean(peaks, axis=1)
    repeat_distances = np.diff(center_points)
    repeat_distances = np.round(repeat_distances, 2)
    repeat_distances = repeat_distances.flatten()

    return repeat_distances


def calculate_grana_values(
    membrane_data: dict, lumen_data: dict, image_dict: dict
) -> dict:
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
    try:
        grana_values = {
            "strip": os.path.basename(image_dict["image_name"])
            .strip(".png")
            .strip("strip_"),
            "image_name": image_dict["image_name"],
            "image_dict": image_dict,
            "px_per_nm": image_dict["pixel_per_nm"],
            "nm_per_px": image_dict["nm_per_pixel"],
            "scale": image_dict["scale"],
            "scale_pixels": image_dict["scale_pixels"],
            "membrane_data": membrane_data,
            "lumen_data": lumen_data,
            "grana_height": calculate_grana_height(membrane_data),
            "num_membrane": len(membrane_data["peaks"]),
            "num_lumen": len(lumen_data["peaks"]),
            "repeat_distance": calculate_repeat_distance(membrane_data),
            "lumen_width": lumen_data["widths"],
            "membrane_width": membrane_data["widths"],
            "lumen_width_heights": lumen_data["width_heights"],
            "membrane_width_heights": membrane_data["width_heights"],
        }
    except:
        print(f"Error calculating grana values for {image_dict['image_name']}")
        return None

    return grana_values


def plot_histogram(
    grana_data: dict,
    metadata: dict,
    output_directory: str,
    peak_type: str = "membrane",
    display: bool = False,
):

    image_dict = grana_data["image_dict"]
    membrane_data = grana_data["membrane_data"]
    lumen_data = grana_data["lumen_data"]
    strip_name = image_dict["strip_name"]
    process_name = metadata["process_name"]

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
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150, facecolor="w")
    plt.plot(histogram)

    plt.plot(peaks, histogram[peaks], "x")
    for peak in peaks:
        plt.plot([peak, peak], [0, histogram[peak]], "--g")

    for peak_left_ip, peak_right_ip, peak_half_height in zip(
        left_ips, right_ips, half_height
    ):
        plt.plot(
            [peak_left_ip, peak_right_ip], [peak_half_height, peak_half_height], "-r"
        )

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
    membrane_df = df.drop(
        columns=["lumen_width", "repeat_distance", "repeat_distance_nm"]
    )
    lumen_df = df.drop(columns=["membrane_width", "repeat_distance_nm"])

    membrane_df = membrane_df.explode("membrane_width")

    lumen_df["lumen_repeat_pairs"] = list(
        zip(lumen_df["lumen_width"], lumen_df["repeat_distance"])
    )

    lumen_df = lumen_df.explode("lumen_repeat_pairs")

    lumen_df["lumen_width"], lumen_df["repeat_distance"] = zip(
        *lumen_df["lumen_repeat_pairs"]
    )

    lumen_df = lumen_df.drop(columns=["lumen_repeat_pairs"])

    return membrane_df, lumen_df


def filter_extrawide_membrane_widths(grana_values: dict, th: float = 16) -> dict:
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
    """

    membrane_peaks = grana_values["membrane_data"]["peaks"]
    left_ips = grana_values["membrane_data"]["left_ips"]
    right_ips = grana_values["membrane_data"]["right_ips"]
    membrane_widths = grana_values["membrane_width"]

    outliers = []

    for i, (left_ip, right_ip, peak, membrane_width) in enumerate(
        zip(left_ips, right_ips, membrane_peaks, membrane_widths)
    ):
        outliers.append("false")

        if i < len(membrane_widths) - 1:  # don't go out of bounds
            if membrane_width > th and membrane_width == membrane_widths[i + 1]:

                left_width, right_width = split_membrane(
                    i,
                    grana_values["membrane_data"],
                    grana_values["lumen_data"],
                    left_ip,
                    right_ip,
                )

                if left_width == 0 and right_width == 0:
                    outliers[i] = "true"
                    continue

                membrane_widths[i] = left_width
                membrane_widths[i + 1] = right_width

    grana_values["membrane_width"] = membrane_widths
    grana_values["outliers"] = outliers

    return grana_values


def split_membrane(
    i: int, membrane_data: dict, lumen_data: dict, left_ip: float, right_ip: float
) -> (float, float):
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

    if (
        (left_ip > lumen_left_ip)
        or (right_ip < lumen_right_ip)
        or (lumen_left_ip > right_ip)
        or (left_ip > right_ip)
    ):
        return (0, 0)

    left_width = lumen_left_ip - left_ip
    right_width = right_ip - lumen_right_ip

    return left_width, right_width


def filter_narrow_widths(
    grana_values: dict, peak_type: str = "membrane", th: float = 1.0
) -> dict:
    """
    Sometimes we get extraneous peaks in the membrane and lumen histograms. These
    are usually very narrow, and can be filtered out. This function filters out
    any peaks that are less than the minimum width threshold value.

    The grana_values dict should contain the membrane_data and lumen_data, which
    contain the widths of the peaks.

    It should

    Mark any peaks that are filtered out in the outliers list.
    """

    membrane_peaks = grana_values["membrane_data"]["peaks"]
    left_ips = grana_values["membrane_data"]["left_ips"]
    right_ips = grana_values["membrane_data"]["right_ips"]
    membrane_widths = grana_values["membrane_width"]

    if peak_type != "membrane":
        raise ValueError(
            "Only membrane peaks can be filtered, lumen not implemented yet"
        )

    if "outliers" not in grana_values:
        grana_values["outliers"] = ["false" for i in range(len(membrane_peaks))]

    if (
        len(membrane_peaks)
        != len(membrane_widths)
        != len(left_ips)
        != len(right_ips)
        != len(outliers)
    ):
        raise ValueError(
            "Length of peaks, widths, left_ips, right_ips, and outliers are not the same"
        )

    for i, (left_ip, right_ip, peak, membrane_width) in enumerate(
        zip(left_ips, right_ips, membrane_peaks, membrane_widths)
    ):
        if membrane_width < th:
            grana_values["outliers"][i] = "true"
        else:
            grana_values["outliers"][i] = "false"

    return grana_values


def clear_existing_data(contour_base_path: str):
    process_names = get_process_names(contour_base_path)

    for process in process_names:
        membrane_csv = f"{contour_base_path}/{process}/grana_data_membrane.csv"
        lumen_csv = f"{contour_base_path}/{process}/grana_data_lumen.csv"

        if os.path.exists(membrane_csv):
            os.remove(membrane_csv)
        if os.path.exists(lumen_csv):
            os.remove(lumen_csv)

        # remove the histograms
        histograms = glob.glob(f"{contour_base_path}/{process}/histograms/*.png")
        for histogram in histograms:
            os.remove(histogram)

    print("Deleted existing grana data files")


def calculate_repeat_distancecalculate_contour_parameters(
    contour_base_path: str,
    output_folder: str,
    metadata_filename: str,
    conversion_df_filename: str,
    min_lumen_width: float = 1.0,
    min_lumen_peak_distance: int = 10,
    min_membrane_width: float = 6.0,
    min_membrane_peak_distance: int = 10,
    max_membrane_width: float = 16.0,
    clear_existing: bool = False,
):

    if clear_existing:
        clear_existing_data(contour_base_path)

    process_names = get_process_names(contour_base_path)

    if (process_names is None) or (len(process_names) == 0):
        print("No processes found")
        exit(1)

    errors_in_processing = []

    for process in process_names:

        image_list = [
            os.path.normpath(f)
            for f in glob.glob(f"{contour_base_path}/{process}/membrane/*.png")
        ]

        if len(image_list) == 0:
            print(f"No images found for {process}")
            continue

        metadata = {
            "process_name": process,
            "conversion_df_filename": conversion_df_filename,
            "metadata_filename": metadata_filename,
            "images": image_list,
            "chosen_height": 0.5,
        }

        output_directory = f"{contour_base_path}/{process}"
        os.makedirs(f"{output_directory}/histograms/", exist_ok=True)

        all_grana_data = []

        if len(metadata["images"]) == 0:
            print(f"No images found for {process}")
            continue

        for image_number, image_name in enumerate(metadata["images"]):

            image_dict = create_image_dict(
                image_path=image_name,
                mask_path=f"{output_folder}/mask",
                contour_base_path=contour_base_path,
                process_name=process,
                metadata=metadata,
            )

            membrane_data = calculate_peak_data(
                image_dict,
                metadata,
                peak_type="membrane",
                min_peak_width=min_membrane_width,
                min_peak_distance=min_membrane_peak_distance,
            )
            lumen_data = calculate_peak_data(
                image_dict,
                metadata,
                peak_type="lumen",
                min_peak_width=min_lumen_width,
                min_peak_distance=min_lumen_peak_distance,
            )

            if membrane_data is None or lumen_data is None:
                print(
                    f"Error processing peak data for process: {process} on image: {image_name}"
                )
                continue
            elif (len(membrane_data["peaks"]) - 1) != len(lumen_data["peaks"]):
                print(
                    f"More membrane peaks than lumen peaks for {image_name}, skipping"
                )
                print(f"Membrane peaks: {len(membrane_data['peaks'])}")
                print(f"Lumen peaks: {len(lumen_data['peaks'])}")
                continue

            grana_values = calculate_grana_values(membrane_data, lumen_data, image_dict)

            grana_values = filter_extrawide_membrane_widths(
                grana_values, th=max_membrane_width
            )
            # grana_values = filter_narrow_widths(grana_values, th=m)

            if grana_values is not None:
                all_grana_data.append(grana_values)
            else:
                print(f"Error processing grana values for: {image_name}")
                errors_in_processing.append(
                    f"Error processing grana values for: {process} : {image_name}"
                )
                continue

            plot_histogram(
                grana_values, metadata, output_directory, peak_type="membrane"
            )
            plot_histogram(grana_values, metadata, output_directory, peak_type="lumen")

        if len(all_grana_data) == 0:
            print(f"No grana data found for {process}")
            errors_in_processing.append(f"No grana data found for {process}")
            continue
        else:
            print(f"Found {len(all_grana_data)} grana data items for {process}")

        grana_df = pd.DataFrame(all_grana_data)

        grana_df["membrane_peaks"] = grana_df["membrane_data"].apply(
            lambda x: x["peaks"]
        )
        grana_df["lumen_peaks"] = grana_df["lumen_data"].apply(lambda x: x["peaks"])
        grana_df = grana_df.drop(columns=["image_dict", "lumen_data", "membrane_data"])

        grana_df["type"] = "-"

        ########################    membranes only     #################################
        try:
            membrane_df = grana_df.drop(columns=["repeat_distance"])

            membrane_df = membrane_df[
                membrane_df.columns.drop(list(membrane_df.filter(regex="lumen")))
            ]

            membrane_df["type"] = "membrane"

            membrane_df = membrane_df.rename(columns={"membrane_peaks": "peaks"})

            membrane_df["index"] = membrane_df.apply(
                lambda row: list(range(len(row["membrane_width"]))), axis=1
            )

            membrane_df = membrane_df.explode(
                [
                    "membrane_width",
                    "peaks",
                    "outliers",
                    "membrane_width_heights",
                    "index",
                ]
            )

            membrane_df["process"] = process

            if "num_membrane" not in membrane_df.columns:
                print(f"No num_membrane column in membrane_df")
                continue

            is_inner = membrane_df.apply(
                lambda row: row["index"] > 0
                and row["index"] < (row["num_membrane"] - 1),
                axis=1,
            )

            membrane_df["membrane_type"] = np.where(is_inner, "inner", "outer")

        except:
            errors_in_processing.append(f"Error processing membrane data for {process}")
            continue

        ############################## lumen only ######################################
        try:
            lumen_df = grana_df.copy()
            lumen_df["type"] = "lumen"

            lumen_df = lumen_df[
                lumen_df.columns.drop(list(lumen_df.filter(regex="membrane")))
            ]
            lumen_df = lumen_df[
                lumen_df.columns.drop(list(lumen_df.filter(regex="outliers")))
            ]

            lumen_df = lumen_df.rename(columns={"lumen_peaks": "peaks"})

            lumen_df["process"] = process

            # Add an 'index' column to track the order of each item in the 'lumen_width' list before exploding
            lumen_df["index"] = lumen_df.apply(
                lambda row: list(range(len(row["lumen_width"]))), axis=1
            )

            lumen_df = lumen_df.explode(
                [
                    "lumen_width",
                    "peaks",
                    "repeat_distance",
                    "lumen_width_heights",
                    "index",
                ]
            )

        except:
            errors_in_processing.append(f"Error processing lumen data for {process}")
            continue

        ####################   save them both   ########################################

        # save it to csv
        print(f"Saving membrane data to {output_directory}/grana_data_membrane.csv")
        membrane_df.to_csv(f"{output_directory}/grana_data_membrane.csv", index=False)

        print(f"Saving lumen data to {output_directory}/grana_data_lumen.csv")
        lumen_df["process"] = process
        lumen_df.to_csv(f"{output_directory}/grana_data_lumen.csv", index=False)

    for error_process in errors_in_processing:
        print(error_process)


def get_data_files(type: str, search_path: str, process_names: list = None) -> list:
    files = []

    if process_names is None:
        files = glob.glob(f"{search_path}/**/*{type}.csv", recursive=True)
    else:
        for process_name in process_names:
            files.extend(
                glob.glob(
                    f"{search_path}/{process_name}/**/*{type}.csv", recursive=True
                )
            )
    return files


def calculate_mean_lumen_width(lumen_files: list) -> list:
    mean_widths = []
    for file in lumen_files:
        data = pd.read_csv(file)
        widths = data["width"]
        mean_widths.append(np.mean(widths))
    return mean_widths


def combine_data_files(base_path: str, processed_image_path: str, output_folder: str):

    os.makedirs(output_folder, exist_ok=True)

    # get the subdirectories in the processed_images directory
    process_names = os.listdir(processed_image_path)

    if (process_names is None) or (len(process_names) == 0):
        print("No processes found")
        exit(1)

    # get all of the lumen and membrane files
    lumen_files = get_data_files("lumen", processed_image_path, process_names)
    membrane_files = get_data_files("membrane", processed_image_path, process_names)

    if (len(lumen_files) == 0) or (len(membrane_files) == 0):
        print("No lumen or membrane files found")
        exit(1)

    # import all of those files into one big df for each type
    lumen_df = pd.concat([pd.read_csv(file) for file in lumen_files])
    membrane_df = pd.concat([pd.read_csv(file) for file in membrane_files])

    # Handle lumen_width based on its content
    if lumen_df["lumen_width"].apply(type).isin([list]).any():
        # If the column contains lists, compute the mean or extract a value
        lumen_df["lumen_width"] = lumen_df["lumen_width"].apply(
            lambda x: np.mean(x) if isinstance(x, list) and x else np.nan
        )
    else:
        # Attempt to convert to numeric
        lumen_df["lumen_width"] = pd.to_numeric(
            lumen_df["lumen_width"], errors="coerce"
        )

    # Handle NaN values
    nan_count = lumen_df["lumen_width"].isna().sum()
    if nan_count > 0:
        lumen_df = lumen_df.dropna(subset=["lumen_width"])

    # Perform the multiplication
    lumen_df["lumen_width_nm"] = lumen_df["lumen_width"] * lumen_df["nm_per_px"]
    membrane_df["membrane_width_nm"] = (
        membrane_df["membrane_width"] * membrane_df["nm_per_px"]
    )

    # Now write them out to a csv
    lumen_df.to_csv(f"{output_folder}/lumen.csv", index=False)
    membrane_df.to_csv(f"{output_folder}/membrane.csv", index=False)
