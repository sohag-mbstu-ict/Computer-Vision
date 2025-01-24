import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.shortcuts import render
import numpy as np
from django.conf import settings
from .models import UploadedImage
from .serializers import UploadedImageSerializer

# Load the model once when the server starts
# MODEL_PATH = "/home/mtl/Music/DRF_keras/check_point"
MODEL_PATH = "/home/mtl/Music/DRF_keras/check_point/Betel_Leaf_inception_Fine_tuned_180_acc99.472_1032_20January2025.h5"
model = load_model(MODEL_PATH)
# Define the labels (adjust based on your model)
# LABELS = ['anthracnose', 'early_blight', 'curl_virus', 'late_blight']
# LABELS = ['foot_rot_disease','white_and_black_flies_insect', 'leaf_rot_disease',  'white_flies_insect']
LABELS = ['UNKNOWN','foot_rot_disease', 'leaf_rot_disease', 'white_and_black_flies_insect', 'white_flies_insect']


def upload_image_form(request):
    """Render the HTML upload form."""
    return render(request, 'upload_image.html')

class UploadAndClassifyImageView(APIView):
    def post(self, request, *args, **kwargs):
        """Handle the POST request for uploading and classifying the image."""
        serializer = UploadedImageSerializer(data=request.data)
        if serializer.is_valid():
            # Save the uploaded image to the database
            uploaded_image = serializer.save()

            # Preprocess the image for classification
            image_path = uploaded_image.image.path
            img = image.load_img(image_path, target_size=(299, 299))  # Resize to your model's input size
            img_array = image.img_to_array(img) / 255.0  # Normalize the image
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Get predictions from the pretrained model
            predictions = model.predict(img_array)
            confidence = float(np.max(predictions))
            predicted_label = LABELS[np.argmax(predictions)]

            # Generate percentages for all labels
            percentages_dict = {
                LABELS[i]: f"{predictions[0][i] * 100:.2f}%" for i in range(len(LABELS))
            }

            # Update the database with classification results
            uploaded_image.label = predicted_label
            uploaded_image.confidence = confidence
            uploaded_image.percentages_dict = percentages_dict
            uploaded_image.save()

            # Return the response with classification results
            return Response(
                {
                    "message": "Image uploaded and classified successfully.",
                    "image_id": uploaded_image.id,
                    "predicted_label": predicted_label,
                    "confidence": f"{confidence * 100:.2f}%",
                    "confidence of all labels": percentages_dict,
                },
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class UploadImageView(APIView):
    def get(self, request, *args, **kwargs):
        # You can return a message or a dummy response for testing GET requests
        return Response({"message": "GET method is not supported for image upload. Use POST to upload images."}, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        # Handle the POST request for uploading the image
        serializer = UploadedImageSerializer(data=request.data)
        if serializer.is_valid():
            uploaded_image = serializer.save()  # Save the image to the database
            return Response(
                {"message": "Image uploaded successfully.", "image_id": uploaded_image.id},
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class GetImagesView(APIView):
    def get(self, request, *args, **kwargs):
        images = UploadedImage.objects.all()  # Retrieve all uploaded images
        serializer = UploadedImageSerializer(images, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

class GetImageByIdView(APIView):
    def get(self, request, id, *args, **kwargs):
        try:
            image = UploadedImage.objects.get(id=id)
            serializer = UploadedImageSerializer(image)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except UploadedImage.DoesNotExist:
            return Response({"error": "Image not found."}, status=status.HTTP_404_NOT_FOUND)



# LABELS = ['anthracnose' 'early_blight' 'curl_virus' 'late_blight']

class ClassifyImageView(APIView):
    def get(self, request, id, *args, **kwargs):
        try:
            # Retrieve the uploaded image by ID
            uploaded_image = UploadedImage.objects.get(id=id)
        except UploadedImage.DoesNotExist:
            return Response({"error": "Image not found."}, status=status.HTTP_404_NOT_FOUND)

        # Preprocess the image for the model
        image_path = uploaded_image.image.path
        img = image.load_img(image_path, target_size=(299, 299))  # Resize for Inception
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Get predictions from the pretrained model
        predictions = model.predict(img_array)    #[[0.1, 0.2, 0.6, 0.1]]
        # predicted_label = LABELS[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        print("predictions : ",predictions)
        itemindex = np.where(predictions == np.max(predictions))
        percentages_dict = {
                                label: f"{prob * 100:.2f}%" for label, prob in zip(LABELS, predictions[0])
                            }
        print("percentages_dict : ",percentages_dict)
        # Update the database with the classification results
        uploaded_image.label = LABELS[itemindex[1][0]]
        uploaded_image.percentages_dict = percentages_dict
        uploaded_image.confidence = confidence
        uploaded_image.save()

        # Serialize the updated image object and return it
        serializer = UploadedImageSerializer(uploaded_image)
        return Response(serializer.data, status=status.HTTP_200_OK)
