import cv2
import os
import glob
import random
import matplotlib.pyplot as plt

def load_images(image_paths):
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting
        images.append(image)
    return images

def plot_and_save_roi_context(context_folder, roi_folder, output_path):
    # List images in context folder and roi folder
    context_images = glob.glob(os.path.join(context_folder, '*.png'))
    roi_images = glob.glob(os.path.join(roi_folder, '*.png'))

    # Ensure there are enough images
    if len(context_images) < 4 or len(roi_images) < 4:
        print("Not enough images to plot.")
        return

    # Select 4 random images
    selected_context_images = random.sample(context_images, 4)
    selected_roi_images = [os.path.join(roi_folder, os.path.basename(img)) for img in selected_context_images]

    # Load the images
    context_imgs = load_images(selected_context_images)
    roi_imgs = load_images(selected_roi_images)

    # Plot images in a row
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for i in range(4):
        axs[0, i].imshow(context_imgs[i])
        axs[0, i].axis('off')
        axs[0, i].set_title('Context Image')
        axs[1, i].imshow(roi_imgs[i])
        axs[1, i].axis('off')
        axs[1, i].set_title('ROI Image')

    # Save the plot
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    context_folder = './roi_images_context'  # Folder containing context images with ROI
    roi_folder = './roi_images'  # Folder containing ROI images
    plot_output_file = './roi_context_plot.png'  # Output file for the plot

    plot_and_save_roi_context(context_folder, roi_folder, plot_output_file)
