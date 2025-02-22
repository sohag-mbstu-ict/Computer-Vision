import cv2
import numpy as np

"""
Histogram-based Similarity
This method compares the color distribution between two images using their histograms. It is invariant to small changes in image content but fails when images differ significantly in composition.
Histogram Intersection: Compares histograms of two images to calculate the similarity.
"""
def resize_image(image, new_size):
    """
    Resize an image using OpenCV.

    Parameters:
    - input_path (str): Path to the input image.
    - output_path (str): Path to save the resized image.
    - new_size (tuple): New size as (width, height).
    """
    # Resize the image
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image

def calculate_histogram_similarity(image1, image2):
    image1 = resize_image(image1,(400,534))
    image2 = resize_image(image2,(400,534))

    """
    Calculate the similarity between two images based on histogram comparison.

    Parameters:
    - image1: First image as a NumPy array.
    - image2: Second image as a NumPy array.

    Returns:
    - similarity (float): A value between 0 and 1, where 1 means identical histograms.
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1)
    hist2 = cv2.normalize(hist2, hist2)

    # Compare histograms using correlation
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return similarity,image1,image2

def find_duplicates(image_folder, threshold=0.9):
    """
    Find and display duplicates in an image folder using histogram similarity.

    Parameters:
    - image_folder (str): Path to the folder containing images.
    - threshold (float): Similarity threshold to consider images as duplicates.
    """
    import os

    # Load all images
    images = []
    image_paths = []
    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            if image is not None:
                images.append(image)
                image_paths.append(file_path)

    duplicates = set()

    # Compare images
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            if image_paths[j] not in duplicates:  # Avoid re-checking duplicates
                similarity,image1,image2 = calculate_histogram_similarity(images[i], images[j])
                print(f"Similarity between {image_paths[i]} and {image_paths[j]}: {similarity}")
                cv2.imshow("Img1", image1)
                cv2.imshow("Img2", image2)
                cv2.waitKey(8000)
                cv2.destroyAllWindows()
                
                if similarity > threshold:
                    try:
                        if os.path.exists(file_path):
                            os.remove(image_paths[j])
                            print(f"Removed duplicate: {image_paths[j]}")
                        else:
                            print(f"{image_paths[j]} File have already been deleted")
                    except Exception as e:
                        print(f"Error removing file {image_paths[j]}: {e}")

# Example usage
find_duplicates("/home/mtl/Documents/R_and_D/img_folder", threshold=0.95)


