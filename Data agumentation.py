import os
from PIL import Image, ImageOps
import random


# Function to create augmented images
def augment_image(image):
    # Apply random rotation between -30 and 30 degrees
    rotated_image = image.rotate(random.uniform(-30, 30))

    # Apply horizontal flip
    flipped_image = ImageOps.mirror(image)

    # Apply scaling (zooming in) by resizing the image
    scale_factor = random.uniform(1.1, 1.5)
    w, h = image.size
    scaled_image = image.resize((int(w * scale_factor), int(h * scale_factor)))

    # Apply translation (shifting image)
    max_shift = 10  # Maximum shift in pixels
    dx, dy = random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)
    translated_image = image.transform(image.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))

    # Apply shearing
    shear_factor = random.uniform(-0.2, 0.2)
    shear_matrix = (1, shear_factor, 0, shear_factor, 1, 0)
    sheared_image = image.transform(image.size, Image.AFFINE, shear_matrix)

    return [rotated_image, flipped_image, scaled_image, translated_image, sheared_image]


# Function to augment dataset and save the original + augmented images
def augment_dataset(input_folder, output_folder):
    # Check if input folder exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist.")

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Modify for other formats if needed
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # Save the original image to the output folder
            image.save(os.path.join(output_folder, f'original_{filename}'))

            # Apply augmentations
            augmented_images = augment_image(image)

            # Save augmented images
            for i, aug_image in enumerate(augmented_images):
                aug_image.save(os.path.join(output_folder, f'{i}augmented{filename}'))

    print(f"Dataset augmentation complete! Augmented images saved to: {output_folder}")


# Set your folder paths
input_folder = r'C:\Users\HEMANTH KUMAR U\PycharmProjects\Nithin\C-NMC_Leukemia\training_data\fold_0'  # Input folder path
output_folder = r'C:\Users\HEMANTH KUMAR U\Desktop\Training'  # Output folder path

# Run the augmentation
augment_dataset(input_folder, output_folder)