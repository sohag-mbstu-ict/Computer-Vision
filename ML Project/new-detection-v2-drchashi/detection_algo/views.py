import os
import logging
import asyncio
from datetime import datetime
from uuid import uuid4
# start 6:37   18:32
import aiofiles
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict
import h5py
import shutil

from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest
from asgiref.sync import async_to_sync, sync_to_async

from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from drf_spectacular.utils import extend_schema

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.backend import clear_session

from .models import ModelStore, ModelLabel
from datetime import datetime, timedelta, timezone
from detection_algo.yolo_model import YOLOPredictor

gmt_plus_6 = timezone(timedelta(hours=6))
detected_at = datetime.now(gmt_plus_6).strftime("%Y-%m-%d %H:%M:%S")

# Logger setup
logger = logging.getLogger(__name__)
handler = logging.FileHandler(os.path.join(settings.BASE_DIR, 'logs', 'model_prediction.log'))
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(handler)


# for classification model
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This line disables GPU, remove if you want GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("Memory growth failed:", e)
except Exception as e:
    print("Outer setup failed:", e)

# Device detection
gpus = tf.config.list_physical_devices('GPU')
USING_DEVICE = "GPU" if gpus else "CPU"
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logger.info(f"Using GPU: {gpus[0].name}")
    except Exception as e:
        logger.warning(f"GPU setup failed: {e}")
        USING_DEVICE = "CPU"
else:
    logger.info("No GPU available. Running on CPU.")

class ImagePreprocessor:
    @staticmethod
    def preprocess(image_path: str, target_size=(299, 299)) -> np.ndarray:
        img = image.load_img(image_path, target_size=target_size)  # Resize to your model's input size
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

class ModelMetadataService:
    @staticmethod
    @sync_to_async
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

            model_path = (
                os.path.join(settings.BASE_DIR, model_path_param)
                if model_path_param else (
                    model_obj.model_file.path if model_obj.model_file else os.path.join(settings.BASE_DIR, model_obj.path)
                )
            )

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
    @staticmethod
    @sync_to_async
    def get_model_info_for_yolo_model(model_name: str, model_path_param: str = None) -> Tuple[str, Dict, Dict]:
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

            model_path = (
                os.path.join(settings.BASE_DIR, model_path_param)
                if model_path_param else (
                    model_obj.model_file.path if model_obj.model_file else os.path.join(settings.BASE_DIR, model_obj.path)
                )
            )
            model_info = {
                'model_id': model_obj.id,
                'model_name': model_obj.name,
                'source': 'db',
                'model_path': model_path,
            }
            cache.set(cache_key, model_info, timeout=86400)
            return model_path, model_info

        except ModelStore.DoesNotExist:
            raise LookupError("Model not found in database.")

class ModelPredictor:
    def __init__(self, model_path: str):
        clear_session()
        print("model_path : ",model_path)
        with tf.device('/GPU:0' if USING_DEVICE == 'GPU' else '/CPU:0'):
            self.model = load_model(model_path)
       
        # shutil.copy(model_path, model_path.replace(".h5", "_backup.h5"))

        # # Open the HDF5 file and rename layers
        # with h5py.File(model_path, 'r+') as f:
        #     layer_names = f['model_weights'].keys()
        #     print("Original Layer Names:")
        #     for name in layer_names:
        #         print(f" - {name}")
            
        #     renamed_layers = {}
        #     for name in list(f['model_weights'].keys()):
        #         if '/' in name:
        #             new_name = name.replace('/', '_')
        #             renamed_layers[name] = new_name
        #             f['model_weights'].move(name, new_name)
            
        #     print("\nRenamed Layers:")
        #     for old, new in renamed_layers.items():
        #         print(f"{old} → {new}")


    def predict(self, img_array: np.ndarray) -> np.ndarray:
        raw_predictions = self.model.predict(img_array)
        return raw_predictions

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
        return async_to_sync(self._async_post)(request, *args, **kwargs)

    async def _async_post(self, request: HttpRequest, *args, **kwargs):
        uploaded_file = request.FILES.get('file')
        model_name = request.data.get('model_name')
        model_path_param = request.data.get('model_path')

        if not uploaded_file:
            return Response({"error": "No file uploaded."}, status=status.HTTP_400_BAD_REQUEST)
        if not model_name:
            return Response({"error": "model_name is required."}, status=status.HTTP_400_BAD_REQUEST)

        temp_file_path = f"temp_{uploaded_file.name}"
        try:
            async with aiofiles.open(temp_file_path, "wb+") as f:
                for chunk in uploaded_file.chunks():
                    await f.write(chunk)

            if(model_name=='Brinjal_Det'): # For detection model
                model_path, model_info = await ModelMetadataService.get_model_info_for_yolo_model(model_name, model_path_param)
                yolo_object = YOLOPredictor()
                predicted_label,predicted_accuracy,all_predictions  = yolo_object.disease_detection_using_yolo_model(temp_file_path,model_path)

            else: # For classification model
                model_path, model_info, label_map = await ModelMetadataService.get_model_info(model_name, model_path_param)
                predictor = ModelPredictor(model_path)
                img_array = ImagePreprocessor.preprocess(temp_file_path)
                predictions = predictor.predict(img_array)

                predicted_class_index = int(np.argmax(predictions, axis=1)[0])
                predicted_label = label_map.get(str(predicted_class_index), "UNKNOWN")
                predicted_accuracy = float(predictions[0][predicted_class_index]) * 100

                all_predictions = [
                    {
                        # "label": label_map.get(str(i), f"Class_{i}"),
                        "accuracy": float(predictions[0][i]) * 100
                    }
                    for i in range(len(predictions[0]))
                ]

            # Generate unique ID and timestamp
            detection_id = str(uuid4())
            # detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Log with extended info
            logger.info(
                f"[{detection_id}] Prediction | Time: {detected_at} | "
                f"Model: {model_info['model_name']} | "
                f"Accuracy: {predicted_accuracy:.2f}%"
            )

            # Extended API response
            return Response({
                "detection_id": detection_id,
                "detected_at": detected_at,
                "model_used": model_info['model_name'],
                "predicted_label": predicted_label,
                "predicted_accuracy": f"{predicted_accuracy:.2f}%",
                # "device_used": USING_DEVICE,
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
