import os
import cv2
import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import pillow_avif  # Required for AVIF support

def download_image(image_url):
    """Download an image from a URL."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(image)
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")
        return None

def extract_features(image):
    """Extract features using ORB feature detector."""
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def compare_images(main_image, additional_image):
    """Compare two images using feature matching."""
    # Extract features from both images
    main_keypoints, main_descriptors = extract_features(main_image)
    additional_keypoints, additional_descriptors = extract_features(additional_image)

    # If no descriptors are found, the images are too different
    if main_descriptors is None or additional_descriptors is None:
        return 0

    # Use FLANN-based matcher
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                        table_number=6,  # 12
                        key_size=12,     # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # or pass an empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(main_descriptors, additional_descriptors, k=2)

    # Apply ratio test to count good matches
    good_matches = [m_n[0] for m_n in matches if len(m_n) == 2 and m_n[0].distance < 0.7 * m_n[1].distance]

    # Return the number of good matches
    return len(good_matches)

def check_repeated_images(row, match_threshold=70):
    """Check if the main image is repeated in the additional images using feature matching."""
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

    # Convert the main image to grayscale for feature extraction
    main_image_gray = cv2.cvtColor(main_image, cv2.COLOR_RGB2GRAY)

    # Check for repeated images in the additional images
    for additional_image_url in additional_image_urls:
        additional_image = download_image(additional_image_url)
        if additional_image is None:
            continue

        # Convert the additional image to grayscale
        additional_image_gray = cv2.cvtColor(additional_image, cv2.COLOR_RGB2GRAY)

        # Compare images using feature matching
        num_matches = compare_images(main_image_gray, additional_image_gray)

        # If the number of good matches exceeds the threshold, the images are considered similar
        if num_matches >= match_threshold:
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
input_file_path = '/Users/ravindra_modi/Documents/elgrocer/URL Checker/test_200.csv'  # Update with your file path
check_images_for_repeats(input_file_path)
