from rest_framework.generics import CreateAPIView
from rest_framework import status, generics
from rest_framework.permissions import AllowAny
from .models import *
from .serializers import *
from rest_framework.response import Response


class VINSINCBUILDINGDATACreateUpdateView(CreateAPIView):
    serializer_class = VINSINCBUILDINGDATASerializer
    queryset = VINSINCBUILDINGDATA.objects.all()
    def perform_create(self, serializer):
        unique_identifier = serializer.validated_data.get('timestamp')
        existing_instance = VINSINCBUILDINGDATA.objects.filter(timestamp=unique_identifier).first()
        if existing_instance:
            serializer.update(existing_instance, serializer.validated_data)
        else:
            serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class BuildingAPIVIew(generics.GenericAPIView):
    serializer_class = BuildingSerializer
    permission_classes = [AllowAny]
    def post(self, request):
        data = self.serializer_class(data=request.data)
        if data.is_valid():
            data.save()
            return Response({'message':'Building added!'})
        return Response({'message':'builing faild to add'})


class UploadBuildingDataCreateUpdateView(CreateAPIView):
    serializer_class = UploadBuildingDataSerializer
    queryset = UploadBuildingData.objects.all()
    def perform_create(self, serializer):
        building = serializer.validated_data.get('building')
        existing_instance = UploadBuildingData.objects.filter(building=building).first()
        if existing_instance:
            serializer.update(existing_instance, serializer.validated_data)
        else:
            serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    

from django.shortcuts import render
from django.http import HttpResponse
from io import BytesIO
import base64
import pandas as pd
import matplotlib.pyplot as plt
from django.conf import settings
import os

def optimize_lighting(df):
    # Replace this with your actual optimization algorithm
    # Example: Reduce lighting by 20% during non-working hours
    working_hours = (8, 18)  # Assuming 8 AM to 6 PM are working hours
    df['Lighting(W/m2)'] = df.apply(lambda row: row['Lighting(W/m2)'] * 0.8 if not working_hours[0] <= row['Date/Time'].hour < working_hours[1] else row['Lighting(W/m2)'], axis=1)
    return df

def plot_energy_data(request):
    current_user = request.user
    user_buildings = Building.objects.filter(user=current_user)
    
    if not user_buildings.exists():
        return render(request, 'no-building-found.html')
    
    df = pd.DataFrame()
    
    for user_building in user_buildings:
        upload_data = UploadBuildingData.objects.filter(building=user_building).first()
        if upload_data and upload_data.upload_file:
            file_path = upload_data.upload_file.path
            building_df = pd.read_excel(file_path)
        else:
            energy_data = VINSINCBUILDINGDATA.objects.filter(building=user_building)
            if energy_data:
                building_df = pd.DataFrame(list(energy_data.values()))
            else:
                building_df = pd.DataFrame()
        
        df = pd.concat([df, building_df], ignore_index=True)
    
    relevant_columns = [
        "Date/Time",
        "Lighting(W/m2)",
        "General Lighting(W/m2)",
        "Radiant Temperature(°C)",
        "Air Temperature(°C)",
    ]
    
    if not df.empty:
        df = df[relevant_columns]
        df = optimize_lighting(df)
        
        plt.figure(figsize=(12, 6))
        for column in relevant_columns[1:]:
            plt.plot(df["Date/Time"], df[column], label=column)
        plt.xlabel("Date/Time")
        plt.ylabel("Values")
        plt.title("Energy Consumption Data with Lighting Optimization")
        plt.legend()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        image_data = buffer.getvalue()
        base64_image = base64.b64encode(image_data).decode("utf-8")
        context = {"image": base64_image}
        
        return render(request, "code.html", context)
    else:
        return render(request, 'no-data-found.html')




