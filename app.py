import os
import requests
from PIL import Image
import imagehash
import pandas as pd
from io import BytesIO
from tqdm import tqdm

def download_image(image_url):
    """Download an image from a URL."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")
        return None

def get_image_hash(image):
    """Calculate the perceptual hash of an image."""
    return imagehash.phash(image)

def check_repeated_images(row):
    """Check if the main image is repeated in the additional images."""
    main_image_url = row['main_image']
    additional_images = row['additional_image']

    # Check if additional_image is missing or not a string
    if pd.isna(additional_images) or not isinstance(additional_images, str):
        return "No additional images"

    additional_image_urls = additional_images.split(", ")

    # Download the main image
    main_image = download_image(main_image_url)
    if main_image is None:
        return "Main image download failed"

    # Get the hash of the main image
    main_image_hash = get_image_hash(main_image)

    # Check for repeated images in the additional images
    for additional_image_url in additional_image_urls:
        additional_image = download_image(additional_image_url)
        if additional_image is None:
            continue

        additional_image_hash = get_image_hash(additional_image)

        # If hashes match, the images are considered similar
        if main_image_hash == additional_image_hash:
            return "Repeated"

    return "Not Repeated"

def check_images_for_repeats(file_path):
    """Check for repeated images in a given CSV file."""
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Use tqdm to display a progress bar
    tqdm.pandas(desc="Checking images for repeats")

    # Check each row for repeated images
    df['is_repeated'] = df.progress_apply(check_repeated_images, axis=1)

    # Save the results to a new CSV file
    output_file_path = os.path.splitext(file_path)[0] + "_checked.csv"
    df.to_csv(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")

# Example usage
input_file_path = '/Users/ravindra_modi/Documents/elgrocer/URL Checker/test.csv'  
check_images_for_repeats(input_file_path)
