import cv2
import os

def extract_sift_features(image):
    """
    Extract SIFT keypoints and descriptors from an image.

    Parameters:
    - image: Input image as a NumPy array.

    Returns:
    - keypoints: List of keypoints.
    - descriptors: NumPy array of descriptors.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return keypoints, descriptors


def calculate_sift_similarity(descriptors1, descriptors2, keypoints1, keypoints2, image1, image2):
    """
    Calculate similarity between two images using SIFT descriptors and visualize keypoint matches.

    Parameters:
    - descriptors1: Descriptors of the first image.
    - descriptors2: Descriptors of the second image.
    - keypoints1: Keypoints of the first image.
    - keypoints2: Keypoints of the second image.
    - image1: First image as a NumPy array.
    - image2: Second image as a NumPy array.

    Returns:
    - similarity (float): Match percentage between the two images.
    - matches_img: Image showing the matches between the two images.
    """
    if descriptors1 is None or descriptors2 is None:
        return 0.0, None

    # Use FLANN-based matcher for descriptor matching
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches
    matches_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Similarity as the ratio of good matches to total keypoints
    similarity = len(good_matches) / max(len(descriptors1), len(descriptors2))
    return similarity, matches_img


def find_duplicates_with_sift(image_folder, threshold=0.7):
    """
    Find and remove duplicate images in a folder using SIFT-based similarity.

    Parameters:
    - image_folder (str): Path to the folder containing images.
    - threshold (float): Similarity threshold to consider images as duplicates.
    """
    images = []
    image_paths = []

    # Load images
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
        keypoints1, descriptors1 = extract_sift_features(images[i])
        for j in range(i + 1, len(images)):
            if image_paths[j] not in duplicates:  # Avoid re-checking duplicates
                keypoints2, descriptors2 = extract_sift_features(images[j])
                similarity, matches_img = calculate_sift_similarity(descriptors1, descriptors2, keypoints1, keypoints2, images[i], images[j])
                print(f"SIFT similarity between {image_paths[i]} and {image_paths[j]}: {similarity:.2f}")
                print("similarity : ",similarity)
                # Display the matched keypoints
                if matches_img is not None:
                    cv2.imshow("Matches", matches_img)
                    cv2.waitKey(15000)  # Display for 5 seconds
                    cv2.destroyAllWindows()

                # Remove duplicate if similarity exceeds the threshold
                if similarity > threshold:
                    try:
                        if os.path.exists(image_paths[j]):
                            # os.remove(image_paths[j])
                            print(f"Removed duplicate: {image_paths[j]}")
                        else:
                            print(f"{image_paths[j]} has already been deleted.")
                    except Exception as e:
                        print(f"Error removing file {image_paths[j]}: {e}")
                    duplicates.add(image_paths[j])

    print("Duplicate removal complete!")


# Example usage
find_duplicates_with_sift("/home/mtl/Documents/R_and_D/img_folder", threshold=0.7)


