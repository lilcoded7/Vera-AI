from rest_framework import serializers 
from .models import *


class VINSINCBUILDINGDATASerializer(serializers.ModelSerializer):
    class Meta:
        model = VINSINCBUILDINGDATA
        fields = '__all__' 



class BuildingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Building
        fields = '__all__' 
