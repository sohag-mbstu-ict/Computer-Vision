# Generated by Django 5.1.5 on 2025-01-22 08:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('classifier', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadedimage',
            name='percentages_dict',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
