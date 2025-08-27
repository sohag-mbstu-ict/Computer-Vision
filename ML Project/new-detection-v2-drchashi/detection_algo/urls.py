from django.urls import path
from .views import ImageDetectionAPIView

urlpatterns = [
    path('detection/', ImageDetectionAPIView.as_view(), name='image-detection'),
]
