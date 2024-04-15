import os
import cv2
import numpy as np
import random

def augment_image(image_path, output_folder):
    # Read the image
    image = cv2.imread(image_path)

    # Get the image name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save the original image in the output folder
    output_path = os.path.join(output_folder, f"{image_name}_raw.png")
    cv2.imwrite(output_path, image)

    # Flip augmentation
    flip_horizontal = cv2.flip(image, 1)
    output_path = os.path.join(output_folder, f"{image_name}_flip_horizontal.png")
    cv2.imwrite(output_path, flip_horizontal)

    flip_vertical = cv2.flip(image, 0)
    output_path = os.path.join(output_folder, f"{image_name}_flip_vertical.png")
    cv2.imwrite(output_path, flip_vertical)

    # Rotation augmentation
    angle = random.uniform(-45, 45)
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    output_path = os.path.join(output_folder, f"{image_name}_rotation_{int(angle)}.png")
    cv2.imwrite(output_path, rotated_image)

    # Grayscale augmentation (applied to 15% of images)
    if random.random() < 0.15:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output_path = os.path.join(output_folder, f"{image_name}_grayscale.png")
        cv2.imwrite(output_path, grayscale_image)

    # Saturation augmentation
    saturation_factor = random.uniform(0.8, 1.2)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = hsv_image[:, :, 1] * saturation_factor
    saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    output_path = os.path.join(output_folder, f"{image_name}_saturation_{saturation_factor:.2f}.png")
    cv2.imwrite(output_path, saturated_image)

    # Blur augmentation
    blur_kernel_size = random.randint(1, 3) * 2 + 1  # Ensure odd kernel size
    blurred_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
    output_path = os.path.join(output_folder, f"{image_name}_blur_{blur_kernel_size}px.png")
    cv2.imwrite(output_path, blurred_image)

    # Noise augmentation
    noise_percentage = random.uniform(0, 0.0101)
    noise_image = image.copy()
    num_noise_pixels = int(noise_percentage * image.shape[0] * image.shape[1])
    for _ in range(num_noise_pixels):
        i = random.randint(0, image.shape[0] - 1)
        j = random.randint(0, image.shape[1] - 1)
        noise_image[i, j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    output_path = os.path.join(output_folder, f"{image_name}_noise_{noise_percentage:.4f}.png")
    cv2.imwrite(output_path, noise_image)

# Specify the input and output folders
input_folder = "/home/jann/Documents/uni/research/CV-10-midjourney-image-splitter/cut_images"
output_folder = "/home/jann/Documents/uni/research/CV-test-11-Image Augmentation/midjourney_augmented_images"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith((".png", ".jpg", ".jpeg")):  # Check for PNG and JPEG extensions
        image_path = os.path.join(input_folder, filename)
        augment_image(image_path, output_folder)