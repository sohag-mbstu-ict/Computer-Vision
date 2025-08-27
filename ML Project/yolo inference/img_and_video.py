import cv2
import os

class ImageAndVideoProcessing:
    def __init__(self):
        pass

    def get_merged_image(self, input_img, output_img):
        # Resize images to 640x640
        input_img = cv2.resize(input_img, (640, 640))
        output_img = cv2.resize(output_img, (640, 640))
        # Add text on both images
        # cv2.putText(input_img, "input", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(output_img, "output", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA)
        # Merge the two images horizontally
        merged_image = cv2.hconcat([input_img, output_img])
        # Resize the merged image to 1280x720
        merged_image_resized = cv2.resize(merged_image, (1280, 720))
        return merged_image_resized

    def get_image_from_folder(self,image_folder):
        images = sorted([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
        return images