from django.db import models

# Create your models here.
class ModelStore(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=100, unique=True)
    version = models.CharField(max_length=100)
    path = models.CharField(max_length=100, blank=True, null=True)
    model_file = models.FileField(upload_to='models', null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return self.name

    class Meta:
        db_table = 'model_store'
        ordering = ['id']


class ModelLabel(models.Model):
    id = models.IntegerField(primary_key=True)
    model_name = models.ForeignKey(ModelStore, on_delete=models.CASCADE, related_name='labels')
    label_id = models.IntegerField()
    label_name = models.CharField(max_length=100)
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.model_name.name} - {self.label_name}"

    class Meta:
        db_table = 'model_label'
        ordering = ['model_name', 'label_id']
        unique_together = ('model_name', 'label_id')