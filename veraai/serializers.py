from rest_framework import serializers 
from .models import *


class BMSAPISerializer(serializers.ModelSerializer):
    class Meta:
        model = BMSSENSOR
        fields = '__all__' 



class CsvOrExcelUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = CsvOrExcelUpload
        fields = '__all__' 