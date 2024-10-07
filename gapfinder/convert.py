import os
import glob
from PIL import Image
import csv
import cv2
import pytesseract
import re
import os
import glob
import re
import argparse

def print_metadata(metadata):
    # print this out in a pretty way. 
    print(f"Filename: {metadata['filename']}")
    print(f"Genotype: {metadata['genotype']}")
    print(f"Condition: {metadata['condition']}")
    print(f"Date: {metadata['date']}")
    print(f"Block: {metadata['block']}")
    print(f"Layer number: {metadata['layer_number']}")
    print(f"K value: {metadata['k_value']}")
    print(f"Scale: {metadata['scale']}")
    print("\n")

def process_image(filename, input_folder='./input_images'):
    # Normalize paths to avoid issues with different OS path separators
    input_folder = os.path.normpath(input_folder)
    filename = os.path.normpath(filename)
    
    print(f"Filename: {filename}")
    
    
    # Extract metadata from folder structure
    folder_parts = filename.strip().split(os.sep)  # Use os.sep to split based on the OS-specific separator
    
    # Check if input_folder is a prefix of the filename path
    if os.path.commonpath([input_folder, filename]) == input_folder:
        folder_parts = folder_parts[len(input_folder.split(os.sep)):]
    
    # Extract metadata from folder structure
    # images\tif_images\Wild Type - 500uE 1 hour\2022-1-10\Block 1\13 WT-500uE 29k.tif
    
    # use os to get the folder parts
    # folder_parts would be: ['images', 'tif_images', 'Wild Type - 500uE 1 hour', '2022-1-10', 'Block 1', '13 WT-500uE 29k.tif]
    folder_parts = filename.strip().split(os.sep)
    
    # starting from the back
    # filename would be: '13 WT-500uE 29k.tif'
    basename = os.path.basename(folder_parts.pop()).split('.tif')[0]
    
    # block would be: '1'
    block = folder_parts.pop().split(' ')[-1]
    
    # date would be: '2022-1-10'
    date = folder_parts.pop()
    
    # genotype_condition would be: ['Wild Type', '500uE 1 hour']
    # genotype would be: 'Wild Type'
    # condition would be '500uE 1 hour'
    genotype, condition = folder_parts.pop().split('-')
    
    # now process the filename
    filename_parts = basename.split(' ')
    
    # layer_number would be: '13'
    layer_number = filename_parts[0]
    
    # k_value would be: '29'
    k_value = filename_parts[-1].split('.')[0]
    
    # scale we get from ocr on the image
    scale = scan_for_text(filename)

    # Create dict to store metadata
    metadata = {
        'filename': basename,
        'genotype': genotype.strip(),
        'condition': condition.strip(),
        'date': date.strip(),
        'block': block,
        'layer_number': layer_number,
        'k_value': k_value,
        'scale': scale
        }
    
    print_metadata(metadata)

    return metadata

def scan_for_text(filename):#
        # load the image using opencv
        bgr_image = cv2.imread(filename)

        # Convert the image to grayscale
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # take only the lowest band  of pixels of the image
        lowest_30_pixels = gray[-65:, :]
        
        # Preprocess the image (e.g., apply thresholding, denoising)
        _, thresh = cv2.threshold(lowest_30_pixels, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processed_image = cv2.medianBlur(thresh, 3)

        text = pytesseract.image_to_string(processed_image)
        
        # Extract numeric values using regular expressions
        numeric_values = [int(num) for num in re.findall(r'\d+', text)]

        # Find the greatest numeric value
        return max(numeric_values)


def convert_tif_to_png(input_folder='./input_images', output_folder='./raw_images', csv_filename='./metadata/image_index.csv'):
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # print out to console what we are doing, what image folders we are looking at, etc
    if os.path.exists(csv_filename):
        print(f"CSV filename: {csv_filename} exists, appending to file")
    else:
        print(f"CSV filename: {csv_filename} does not exist, creating new file")

    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['filename', 'genotype', 'condition', 'date', 'block', 'layer_number', 'rep_number', 'k_value', 'scale']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If CSV file doesn't exist, write header row
        if not os.path.exists(csv_filename):
            writer.writeheader()

        # Iterate through files using glob
        for filename in glob.iglob('*/**/*.tif', recursive=True):

            # Process image and get metadata
            metadata = process_image(filename)
        
            # Save image as PNG with appropriate name in raw_images folder
            image = Image.open(filename)
            new_filename = f"{output_folder}/{metadata['date']}_{metadata['genotype']}_{metadata['condition']}_{metadata['block']}_{metadata['layer_number']}_{metadata['k_value']}.png"
            image.save(new_filename)

            # Write metadata to CSV
            metadata['filename'] = new_filename
            writer.writerow(metadata)
            
        print(f"Finished processing images, saved output to CSV file at {csv_filename}")