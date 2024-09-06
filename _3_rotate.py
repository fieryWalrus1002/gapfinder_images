import cv2
import numpy as np
import glob
import os
import shutil

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def load_image_bgr(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def equalize_histogram(image):
    return cv2.equalizeHist(image)

def invert_image(image):
    # Invert the binary image
    return cv2.bitwise_not(image)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    # Apply Gaussian blur to reduce noise
    return cv2.GaussianBlur(image, kernel_size, 0)

def fixed_threshold(image, threshold_value=170, max_value=255):
    # Use fixed thresholding to binarize the image
    _, thresholded_image = cv2.threshold(image, threshold_value, max_value, cv2.THRESH_BINARY_INV)
    return thresholded_image

def adaptive_threshold(image):
    # Use adaptive thresholding to binarize the image
    return cv2.adaptiveThreshold(image, 170, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)

def morphological_operations(image):
    # Use morphological operations to enhance lines
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    inversed = invert_image(dilated)
    return inversed

def detect_edges(image, low_threshold=50, high_threshold=255):
    return cv2.Canny(image, low_threshold, high_threshold, apertureSize=3)

def calculate_rotation_angle(lines):
    angles = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
    median_angle = np.median(angles)
    return median_angle

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def preprocess_image(image_path):
    image = load_image(image_path)
    blurred_image = apply_gaussian_blur(image, kernel_size=(5, 5))
    equalized_image = equalize_histogram(blurred_image)
    thresholded_image = fixed_threshold(equalized_image, threshold_value=150, max_value=255)
    processed_image = morphological_operations(thresholded_image)
    return processed_image

def find_and_draw_contours(edges, original_image, size_threshold=1000):
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(original_image)

    for contour in contours:
        if cv2.contourArea(contour) > size_threshold:
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
            # draw the contour unfilled
            # cv2.drawContours(mask, [contour], -1, (255), thickness=4)

    return mask

def write_angle_to_file(image_path, output_path, rotation_angle):
    file_path = os.path.join(output_path, "rotation_angles.txt")
    with open(file_path, "a") as file:
        file.write(f"{image_path}: {rotation_angle}\n")

def detect_lines(image, threshold=100, min_line_length=50, max_line_gap=10):
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines

def draw_lines_on_image(lines, image):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2) # Drawing lines in green
    return image

def main(image_path, example=False):
    processed_image = preprocess_image(image_path)
    original_image = load_image(image_path)
    
    # Display the filtered (processed) image
    if example:
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
    
    edges = detect_edges(processed_image)

    # contours
    contour_image = find_and_draw_contours(edges, processed_image, size_threshold=100)
    cv2.imwrite(f"./contour_images/{os.path.basename(image_path)}", contour_image)

    if example:
        cv2.imshow("Contours", contour_image)
        cv2.waitKey(0)
    
    lines = detect_lines(contour_image, threshold=100, min_line_length=25, max_line_gap=5)
    
    if lines is None:
        print("No lines were detected")
        return

   # Draw lines on the original image and save
    bgr_image = load_image_bgr(image_path)
    line_image = draw_lines_on_image(lines, bgr_image)
    line_image_path = f"./line_images/{os.path.basename(image_path)}"
    cv2.imwrite(line_image_path, line_image)

    # now do the rotation
    rotation_angle = calculate_rotation_angle(lines)
    rotated_image = rotate_image(original_image, rotation_angle)
    
    if example:
        cv2.imshow("Rotated Image", rotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    output_path = f"./rotated_images/{os.path.basename(image_path)}"
    print(f"Saving rotated image to: {output_path}")
    cv2.imwrite(output_path, rotated_image)
    
    write_angle_to_file(image_path, "./rotated_images", rotation_angle)

if __name__ == "__main__":
    clear_directory("./contour_images")
    clear_directory("./rotated_images")
    clear_directory("./line_images")
    
    image_paths = glob.glob("./strip_images/*.png")
    for image_path in image_paths:
        main(image_path)
