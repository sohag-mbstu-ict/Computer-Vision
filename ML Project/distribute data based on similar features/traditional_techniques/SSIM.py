import os
import cv2
from skimage.metrics import structural_similarity as ssim

"""
Structural Similarity Index (SSIM): Measures the structural similarity between two images by focusing on luminance, contrast, and structure. SSIM is more perceptually relevant compared to MSE."""

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


def calculate_ssim(image1, image2):
    image1 = resize_image(image1,(400,534))
    image2 = resize_image(image2,(400,534))
    cv2.imshow("Img1", image1)
    cv2.imshow("Img2", image2)
    cv2.waitKey(8000)
    """
    Calculate the Structural Similarity Index (SSIM) between two images.

    Parameters:
    - image1: First image as a NumPy array.
    - image2: Second image as a NumPy array.

    Returns:
    - ssim_score: A float representing SSIM similarity (1.0 means identical).
    """
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    ssim_score, _ = ssim(gray1, gray2, full=True)
    return ssim_score

def find_and_remove_duplicates(image_folder, threshold=0.15):
    """
    Detect and remove duplicate images based on SSIM.

    Parameters:
    - image_folder: Path to the folder containing images.
    - threshold: SSIM similarity threshold for considering images as duplicates.

    Returns:
    - None
    """
    images = []
    image_paths = []

    # Load all images into memory
    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)
        if os.path.isfile(file_path):
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(image)
                    image_paths.append(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Compare images and find duplicates
    duplicates = set()
    kept_images = set() 
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            if image_paths[j] not in duplicates:  # Skip already marked duplicates
                similarity = calculate_ssim(images[i], images[j])
                
                # print("similarity between two images (i,j) : ", image_paths[i], " ",image_paths[j], "  ",similarity)
                cv2.destroyAllWindows()
                if similarity > threshold:
                    print("Remove : ",image_paths[j]," ",similarity)
                    os.remove(image_paths[j])


# Example usage
find_and_remove_duplicates("/home/mtl/Documents/R_and_D/img_folder", threshold=0.16)

a=5
a