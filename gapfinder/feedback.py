import numpy as np
import cv2
import os
import pandas as pd
from matplotlib import pyplot as plt
from itertools import product


def getMembraneImage(process_folder: str, strip_name: str) -> np.ndarray:
    """
    Get the membrane image for a given process.

    Args:
        process_folder: The folder of the process (e.g., ./output/trial_1/processed_images/process_1).
        strip_name: The name of the strip image file (e.g., 'strip_101.png').

    Returns:
        The membrane image as a NumPy array.
    """
    # Get the path to the membrane image
    membrane_image_path = os.path.join(process_folder, "membrane", strip_name)

    # Load the membrane image
    membrane_image = cv2.imread(membrane_image_path, cv2.IMREAD_GRAYSCALE)

    return membrane_image


def getRoiImage(roi_folder: str, strip_name: str) -> np.ndarray:
    """
    Get the roi image for a given strip.

    Args:
        strip_name: The name of the strip.

    Returns:
        The roi image as a NumPy array, converted to RGB.

    """
    # Get the path to the roi image
    roi_image_path = os.path.join(roi_folder, strip_name)

    # Load the roi image
    roi_image = cv2.imread(roi_image_path, cv2.IMREAD_GRAYSCALE)

    # convert to a rgb image
    roi_image = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2RGB)

    return roi_image


def getMembraneDf(poi_folder: str, strip: str) -> pd.DataFrame:
    """
    Get the membrane data for the POI.

    Args:
        poi_folder: The folder of the POI (e.g., ./output/trial_1/poi/process_1).

    Returns:
        A DataFrame containing the membrane data.
    """
    # Get the path to the membrane data file
    membrane_data_path = os.path.join(poi_folder, "grana_data_membrane.csv")
    # print(f"Membrane data path: {membrane_data_path}")
    # Load the membrane data
    membrane_data = pd.read_csv(membrane_data_path)
    # print(f"len(membrane_data): {len(membrane_data)}")
    # isolate the numeric portion of the strip
    strip1 = int(strip.split("_")[-1].strip(".png"))
    # print(f"Strip: {strip1}")

    # filter to only include the strip
    membrane_data = membrane_data[membrane_data["strip"] == strip1]

    return membrane_data


def drawContours(
    membrane_image,
    roi_image: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 1,
    df: pd.DataFrame = None,
) -> np.ndarray:
    """
    Draw the contours of the membrane on the roi image.

    Args:
        membrane_image: The membrane image as a NumPy array.
        roi_image: The roi image as a NumPy array.
        color: The color of the contours.

    Returns:
        The roi image with the contours drawn on it.
    """

    # Find the contours in the membrane image. Membrane is white, background is black
    contours, _ = cv2.findContours(
        membrane_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw the contours on the roi image (-1 means all contours)
    roi_image_with_contours = cv2.drawContours(
        roi_image.copy(), contours, -1, color, thickness
    )

    if df is not None:
        # get the peaks from the df
        peaks = df["peaks"].values
        # for each peak, plot the peak number at the y value of the peak
        for i, peak in enumerate(peaks):

            # draw a dashed blue line at the peak
            cv2.line(
                roi_image_with_contours,
                (0, peak),
                (roi_image_with_contours.shape[1], peak),
                (0, 0, 255),
                1,
                cv2.LINE_8,
                0,
            )

            # add a text label for the peak, in red
            cv2.putText(
                roi_image_with_contours,
                f"{i}",
                (roi_image_with_contours.shape[1] - 10, peak),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

    return roi_image_with_contours


import shutil
import os


def delete_existing_files(folder: str):
    if not os.path.exists(folder):
        return

    for f in os.scandir(folder):
        if os.path.exists(f.path):
            os.remove(f.path)


def delete_folder(folder: str):
    try:
        shutil.rmtree(folder)
    except OSError as e:
        print(f"Error: {e.strerror}")


# # now do that for each process folder, for each strip name
def create_roi_vs_contours_images(
    metadata_csv: str, roi_folder: str, poi_folder: str, clear_existing: bool = False
):
    """
    Iterate through the process folders, and for each strip in the metadata csv,
    create a plot that shows two images:
    1. the roi image on its own
    2. The roi images with the contours of the membrane drawn on it in green,
        and the peaks from the membrane data drawn on it in red,
        and the peak width drawn on it in blue

    Save these out to the process_folder/roi_vs_contours folder
    """

    metadata = pd.read_csv(metadata_csv)

    process_folders = [f.path for f in os.scandir(poi_folder) if f.is_dir()]

    strip_filenames = metadata["strip_filename"].unique()

    combinations = list(product(process_folders, strip_filenames))

    if clear_existing:
        for process in process_folders:
            if os.path.exists(os.path.join(process, "roi_vs_contours")):
                print(f"Deleting existing roi_vs_contours folder in {process}")
                delete_folder(os.path.join(process, "roi_vs_contours"))

    for process, strip in combinations:

        membrane_image = getMembraneImage(process, strip)
        roi_image = getRoiImage(roi_folder, strip)
        df = getMembraneDf(process, strip)
        roi_image_with_contours = drawContours(
            membrane_image, roi_image, color=(0, 255, 0), thickness=1
        )

        # create an output folder for the roi_vs_contours images
        roi_vs_contours_folder = os.path.join(process, "roi_vs_contours")

        os.makedirs(roi_vs_contours_folder, exist_ok=True)

        roi_vs_contours_path = os.path.join(roi_vs_contours_folder, strip)

        # use opencv to write the image to the process_folder/contours folder
        contours_folder = os.path.join(process, "contours")
        os.makedirs(contours_folder, exist_ok=True)
        contours_image_path = os.path.join(contours_folder, strip)

        cv2.imwrite(contours_image_path, roi_image_with_contours)
        # print(f"Image written to {contours_image_path}")

        fig, axes = plt.subplots(1, 2, figsize=(8, 5))

        # Plot roi_image on the left
        axes[0].imshow(roi_image, cmap="gray")
        axes[0].axis("off")
        axes[0].set_title("roi Image")

        # Plot roi_image_with_contours on the right
        axes[1].imshow(roi_image_with_contours)
        axes[1].axis("off")
        axes[1].set_title("roi Image with Contours")

        if df is not None:
            # get the peaks from the df
            peaks = df["peaks"].values
            # for each peak, plot the peak number at the y value of the peak
            for i, peak in enumerate(peaks):
                # draw a dashed line at the peak
                axes[1].plot([0, roi_image_with_contours.shape[1]], [peak, peak], "r--")
                # add a text label for the peak, in red
                axes[1].text(
                    roi_image_with_contours.shape[1] + 1,
                    peak,
                    f"{i}",
                    color="red",
                    fontsize=12,
                )

                # add the vertical line showing the peak width, in blue, at center of peak
                peak_width = df["membrane_width"].values[i]
                peak_center = roi_image.shape[1] // 2
                axes[1].plot(
                    [peak_center, peak_center],
                    [peak - peak_width / 2, peak + peak_width / 2],
                    "b",
                )

                # draw a horizontal line at the top and bottom of each of those peaks
                axes[1].plot(
                    [peak_center - 2.5, peak_center + 2.5],
                    [peak - peak_width / 2, peak - peak_width / 2],
                    "b",
                )
                axes[1].plot(
                    [peak_center - 2.5, peak_center + 2.5],
                    [peak + peak_width / 2, peak + peak_width / 2],
                    "b",
                )

        plt.suptitle(f"Process: {process}, Strip: {strip}")
        plt.tight_layout()

        plt.savefig(roi_vs_contours_path)
        plt.close()
