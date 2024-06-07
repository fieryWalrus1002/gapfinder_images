import cv2
import os
import csv
import glob

# Global variables
current_image_index = 0
images = []
image_folder = "./raw_images"
roi_list = []
rect_size = 200  # Size of the rectangle (fixed size)
drawing = False
ix, iy = -1, -1
resize_factor = 0.25  # Factor to resize the image for display
coord_text = "(0, 0)"


def load_image(index):
    """Loads an image from the images list based on the index."""
    image_path = images[index]
    image = cv2.imread(image_path)
    return image, os.path.basename(image_path)

def draw_rectangle(img, center, base_rect_size=200, scale_factor=1.0):
    """Draws a rectangle centered at the given coordinates."""
    # Scale the rectangle size based on the resize factor
    scaled_rect_size = int(base_rect_size * scale_factor)
    x, y = center
    top_left = (x - scaled_rect_size // 2, y - scaled_rect_size // 2)
    bottom_right = (x + scaled_rect_size // 2, y + scaled_rect_size // 2)
    cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, roi_list, coord_text

    original_image, display_image = param

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = int(x / resize_factor), int(y / resize_factor)
        roi_list.append((ix, iy))
        # cv2.circle(display_image, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow("Image", display_image)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = display_image.copy()
            draw_rectangle(temp_img, (x, y), rect_size, resize_factor)
            # Display current coordinates in the lower left corner
            coord_text = f"({x}, {y})"
            # cv2.putText(temp_img, coord_text, (10, temp_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Image", temp_img)
        else:
            # Display current coordinates in the lower left corner even when not drawing
            temp_img = display_image.copy()
            coord_text = f"({x}, {y})"
            # cv2.putText(temp_img, coord_text, (10, temp_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Image", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        draw_rectangle(display_image, (x, y), rect_size, resize_factor)
        roi_list[-1] = (int(x / resize_factor), int(y / resize_factor))  # Update the last ROI with final position
        cv2.imshow("Image", display_image)

def save_roi(image_name, roi_list, folder):
    """Saves the ROI information to a csv file."""
    csv_path = os.path.join(folder, 'roi_data.csv')
    with open(csv_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for roi in roi_list:
            csv_writer.writerow([image_name, roi[0], roi[1], rect_size])
            
def clear_roi_data(folder):
    """Clears the ROI data file."""
    csv_path = os.path.join(folder, 'roi_data.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)

def main():
    global current_image_index, images, roi_list, coord_text

    # clear_roi_data(image_folder)

    # Load all image paths
    images = glob.glob(f"{image_folder}/*.png")
    if not images:
        print("No images found in the folder.")
        return
    else:
        print(f"Found {len(images)} images in the folder.")

    cv2.namedWindow("Image")

    while True:
        image, image_name = load_image(current_image_index)
        roi_list = []

        # Resize image for display
        resized_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
        display_image = resized_image.copy()

        # Set mouse callback with original and display image
        cv2.setMouseCallback("Image", mouse_callback, (image, display_image))

        while True:
            temp_img = display_image.copy()
            
            # display the coordinates of the current ROI
            cv2.putText(temp_img, image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # display the current image index, out of the max number of images
            cv2.putText(temp_img, f"Image {current_image_index + 1}/{len(images)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(temp_img, coord_text, (10, temp_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Image", temp_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                save_roi(image_name, roi_list, image_folder)
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                save_roi(image_name, roi_list, image_folder)
                current_image_index = (current_image_index + 1) % len(images)
                break
            elif key == 9:  # Tab key does the same as 'n'
                save_roi(image_name, roi_list, image_folder)
                current_image_index = (current_image_index + 1) % len(images)
                break
            elif key == ord('r'):
                roi_list = []
                display_image = resized_image.copy()  # Reset to the original resized image
                break



if __name__ == "__main__":
    main()
