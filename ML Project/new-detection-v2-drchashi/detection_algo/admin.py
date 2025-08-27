from django.contrib import admin
from .models import ModelStore, ModelLabel

@admin.register(ModelStore)
class ModelStoreAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'version', 'path', 'updated_at', 'created_at')
    search_fields = ('name', 'version')
    list_filter = ('updated_at', 'created_at')
    ordering = ('id',)
    readonly_fields = ('created_at', 'updated_at')

    fieldsets = (
        (None, {
            'fields': ('id', 'name', 'version', 'path', 'model_file')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
        }),
    )


@admin.register(ModelLabel)
class ModelLabelAdmin(admin.ModelAdmin):
    list_display = ('id', 'model_name', 'label_id', 'label_name', 'updated_at', 'created_at')
    search_fields = ('label_name', 'model_name__name')
    list_filter = ('model_name', 'updated_at', 'created_at')
    ordering = ('model_name', 'label_id')
    readonly_fields = ('created_at', 'updated_at')

    fieldsets = (
        (None, {
            'fields': ('id', 'model_name', 'label_id', 'label_name')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
        }),
    )
