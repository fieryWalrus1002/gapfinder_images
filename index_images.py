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

def process_image(filename):
    # Extract metadata from folder structure
    folder_parts = filename.strip().split('\\')
    genotype_condition = folder_parts[0].split('-')
    genotype = genotype_condition[0].strip(' ')
    condition = genotype_condition[1]
    date = folder_parts[1]
    block = folder_parts[2].split(' ')[-1]

    basename = os.path.basename(filename).split('.tif')[0]
    filename_parts = basename.split(' ')
    layer_number = filename_parts[0]
    k_value = filename_parts[-1].split('.')[0]

    scale = scan_for_text(filename)
    print(scale)

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

    return metadata

def scan_for_text(filename):
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

def main():
    os.makedirs('raw_images', exist_ok=True)
    os.makedirs('ocr_images', exist_ok=True)

    # Open or create CSV file for image metadata
    csv_filename = 'image_index.csv'
    csv_exists = os.path.exists(csv_filename)

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
        
            # Save image as PNG with appropriate name
            image = Image.open(filename)
            new_filename = f"raw_images/{metadata['date']}_{metadata['genotype']}_{metadata['condition']}_{metadata['block']}_{metadata['layer_number']}_{metadata['k_value']}.png"
            image.save(new_filename)

            # Write metadata to CSV
            metadata['filename'] = new_filename
            writer.writerow(metadata)

if __name__ == "__main__":
    main()
