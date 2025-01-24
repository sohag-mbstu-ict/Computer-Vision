# classifier/urls.py
from django.urls import path
from .views import UploadImageView, ClassifyImageView, GetImagesView, GetImageByIdView,upload_image_form,UploadAndClassifyImageView

urlpatterns = [
    path('', upload_image_form, name='upload-form'), # connected with frontend
    path("upload-and-classify/", UploadAndClassifyImageView.as_view(), name="classify-image"), # connected with frontend


    path("upload-image/", UploadImageView.as_view(), name="upload-image"),
    path("classify-image/<int:id>/", ClassifyImageView.as_view(), name="classify-image"),
    path('get-images/', GetImagesView.as_view(), name='get-images'),
    path('get-images/<int:id>/', GetImageByIdView.as_view(), name='get-image-by-id'),
]
