# classifier/models.py
from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to="uploaded_images/")
    label = models.CharField(max_length=255, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    percentages_dict = models.JSONField(blank=True, null=True) 
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image {self.id}: {self.label or 'Not Classified'}"
