import os
import csv
import glob

def generate_data(filenames):
    data = []
    counter = 0  # Initialize counter
    
    # Iterate through filenames
    for filename in filenames:
        basename = os.path.basename(filename)

        # Generate incrementing numbers for each row and add to data
        for i in range(1, 4):
            counter += 1  # Increment counter
            strip_filename = f"strip_{counter:04d}_{i}"
            row = [filename, strip_filename, f'{counter:04d}', i, '', '']  # Format counter with leading zeros
            data.append(row)

    return data

def write_to_csv(data, output_file):
    # Write data to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'strip_filename', 'Counter', 'Rep', 'X', 'Y'])  # Write header row
        writer.writerows(data)

if __name__ == "__main__":
    folder_path = "raw_images\\*.png"  # Specify the folder path
    output_file = 'output.csv'  # Specify the output CSV file name

    # Get list of filenames in the folder
    filenames = glob.glob(folder_path)
    
    # Generate data
    data = generate_data(filenames)

    # Write data to CSV
    write_to_csv(data, output_file)
