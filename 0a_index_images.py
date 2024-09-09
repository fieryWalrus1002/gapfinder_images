import os
import glob
from PIL import Image
import csv
import cv2
# https://github.com/UB-Mannheim/tesseract/wiki to install tesseract
# pip install pillow opencv-python pytesseract
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

    # print(f"Folder parts: {folder_parts}")
    genotype_condition = folder_parts[0].split('-')
    # print(f"Genotype and condition: {genotype_condition}")
    genotype = genotype_condition[0].strip(' ')
    # print(f"Genotype: {genotype}")

    condition = genotype_condition[1]

    # print(f"Condition: {condition}")
    date = folder_parts[1]
    # print(f"Date: {date}")
    block = folder_parts[2].split(' ')[-1]
    # print(f"Block: {block}")


    basename = os.path.basename(filename).split('.tif')[0]
    filename_parts = basename.split(' ')
    layer_number = filename_parts[0]
    k_value = filename_parts[-1].split('.')[0]
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

# Main function
def main(args):  
    os.makedirs(args.input_folder, exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    # Open or create CSV file for image metadata
    csv_filename = './metadata/image_index.csv'
    csv_exists = os.path.exists(csv_filename)
    
    # print out to console what we are doing, what image folders we are looking at, etc
    if csv_exists:
        print(f"CSV filename: {csv_filename} exists, appending to file")
    else:
        print(f"CSV filename: {csv_filename} does not exist, creating new file")

    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['filename', 'genotype', 'condition', 'date', 'block', 'layer_number', 'rep_number', 'k_value', 'scale']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If CSV file doesn't exist, write header row
        if not csv_exists:
            writer.writeheader()

        # Iterate through files using glob
        for filename in glob.iglob('*/**/*.tif', recursive=True):

            # Process image and get metadata
            metadata = process_image(filename)
        
            # Save image as PNG with appropriate name in raw_images folder
            image = Image.open(filename)
            new_filename = f"raw_images/{metadata['date']}_{metadata['genotype']}_{metadata['condition']}_{metadata['block']}_{metadata['layer_number']}_{metadata['k_value']}.png"
            image.save(new_filename)

            # Write metadata to CSV
            metadata['filename'] = new_filename
            writer.writerow(metadata)
            
        print(f"Finished processing images, saved output to CSV file at {csv_filename}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and extract metadata")
    parser.add_argument('-i', '--input_folder', type=str, default='./input_images', help='The folder containing the images to process')
    parser.add_argument('-o', '--output_folder', type=str, default='./raw_images', help='The folder to save the processed images')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print out extra information')

    args = parser.parse_args()

    if args.verbose:
        print(f"Arguments: {args}")

    main(args)
