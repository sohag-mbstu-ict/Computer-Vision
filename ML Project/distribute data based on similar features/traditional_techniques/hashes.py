
from PIL import Image
import os
import imagehash

def remove_similar_images(image_folder):
    hashes = {}
    duplicates = []

    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)
        
        if os.path.isfile(file_path):
            image = Image.open(file_path)
            resized_img = image.resize((400,534), Image.Resampling.LANCZOS)
            file_hash = imagehash.average_hash(image)
            
            if file_hash in hashes:
                duplicates.append(file_path)
            else:
                hashes[file_hash] = file_path

    # Remove duplicates
    for duplicate in duplicates:
        # os.remove(duplicate)
        print(f"Removed: {duplicate}")
        print()



# Example usage
remove_similar_images("/home/mtl/Documents/R_and_D/img_folder")
a=5
a
