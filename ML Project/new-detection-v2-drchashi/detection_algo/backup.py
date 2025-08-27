import os
import numpy as np
import tensorflow as tf
import logging
import asyncio
from typing import Tuple, Dict
from datetime import datetime
from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView
from drf_spectacular.utils import extend_schema
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.backend import clear_session

from .models import ModelStore, ModelLabel

logger = logging.getLogger(__name__)
handler = logging.FileHandler(os.path.join(settings.BASE_DIR, 'logs', 'model_prediction.log'))
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(handler)


class ImagePreprocessor:
    @staticmethod
    def preprocess(image_path: str, target_size=(300, 300)) -> np.ndarray:
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)


class ModelMetadataService:
    @staticmethod
    def get_model_info(model_name: str, model_path_param: str = None) -> Tuple[str, Dict, Dict]:
        cache_key = f"model_meta:{model_name}"
        cached_data = cache.get(cache_key)

        if cached_data:
            if model_path_param:
                cached_data['model_path'] = os.path.join(settings.BASE_DIR, model_path_param)
            model_info = dict(cached_data)
            model_info['source'] = 'cache'
            return model_info['model_path'], model_info, model_info['label_map']

        try:
            model_obj = ModelStore.objects.get(name=model_name)

            if model_path_param:
                model_path = os.path.join(settings.BASE_DIR, model_path_param)
            elif model_obj.model_file:
                model_path = model_obj.model_file.path
            elif model_obj.path:
                model_path = os.path.join(settings.BASE_DIR, model_obj.path)
            else:
                raise ValueError("Model path is not configured in DB.")

            label_qs = ModelLabel.objects.filter(model_name=model_obj)
            label_map = {str(label.label_id): label.label_name for label in label_qs}

            if not label_map:
                raise ValueError("No labels found for the selected model.")

            model_info = {
                'model_id': model_obj.id,
                'model_name': model_obj.name,
                'source': 'db',
                'model_path': model_path,
                'label_map': label_map
            }

            cache.set(cache_key, model_info, timeout=86400)
            return model_path, model_info, label_map

        except ModelStore.DoesNotExist:
            raise LookupError("Model not found in database.")


class ModelPredictor:
    def __init__(self, model_path: str):
        clear_session()  # Prevents memory buildup when loading multiple models
        self.model = load_model(model_path)

    def predict(self, img_array: np.ndarray) -> np.ndarray:
        raw_predictions = self.model.predict(img_array, batch_size=1, verbose=0)
        return tf.nn.softmax(raw_predictions).numpy()


class ImageDetectionAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [AllowAny]

    @extend_schema(
        request={
            "multipart/form-data": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "format": "binary"},
                    "model_name": {"type": "string"},
                    "model_path": {"type": "string"}
                },
                "required": ["file"]
            }
        },
        summary="Image Detection Using Dynamic Model",
        description="Upload an image with either model name (from DB) or direct path to a model file."
    )
    def post(self, request: HttpRequest, *args, **kwargs):
        uploaded_file = request.FILES.get('file')
        model_name = request.data.get('model_name')
        model_path_param = request.data.get('model_path')

        if not uploaded_file:
            return Response({"error": "No file uploaded."}, status=status.HTTP_400_BAD_REQUEST)
        if not model_name:
            return Response({"error": "model_name is required."}, status=status.HTTP_400_BAD_REQUEST)

        temp_file_path = f"temp_{uploaded_file.name}"
        try:
            with open(temp_file_path, "wb+") as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            model_path, model_info, label_map = ModelMetadataService.get_model_info(model_name, model_path_param)

            predictor = ModelPredictor(model_path)
            img_array = ImagePreprocessor.preprocess(temp_file_path)
            predictions = predictor.predict(img_array)

            predicted_class_index = int(np.argmax(predictions, axis=1)[0])
            predicted_label = label_map.get(str(predicted_class_index), "UNKNOWN")
            predicted_accuracy = float(predictions[0][predicted_class_index]) * 100

            all_predictions = [
                {
                    "label": label_map.get(str(i), f"Class_{i}"),
                    "accuracy": float(predictions[0][i]) * 100
                }
                for i in range(len(predictions[0]))
            ]

            logger.info(f"Prediction | Source: {model_info['source']} | Model: {model_info['model_name']} | "
                        f"Prediction: {predicted_label} | Accuracy: {predicted_accuracy:.2f}%")

            return Response({
                "model_used": model_info,
                "predicted_label": predicted_label,
                "predicted_accuracy": f"{predicted_accuracy:.2f}%",
                "all_predictions": all_predictions
            })

        except LookupError as e:
            return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            return Response({"error": f"Unexpected error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
