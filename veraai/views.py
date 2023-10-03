from rest_framework.generics import CreateAPIView
from rest_framework import status, generics
from rest_framework.permissions import AllowAny
from .models import *
import random
from .serializers import *
from rest_framework.response import Response
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from django.db.models.functions import Coalesce
from django.db.models import F, Sum
from decimal import Decimal
import base64
from django.shortcuts import render
from django.http import HttpResponse
from .forms import EnergyPredictionForm
from sklearn.linear_model import LinearRegression
from datetime import datetime


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
        return Response({'message':'BMS connected'})


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
        Response({'message':'BMS Uploaded'})
        return redirect('energy-optimization')
    

import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from django.shortcuts import render
from django.http import HttpResponse
from .models import UploadBuildingData

# Define the columns to optimize
columns_to_optimize = [
    'Lighting(W/m2)', 'General Lighting(W/m2)', 'Miscellaneous(W/m2)',
    'Process(W/m2)', 'Computer + Equip(W/m2)', 'Occupancy(W/m2)',
    'Solar Gains Exterior Windows(W/m2)', 'Zone Sensible Cooling(W/m2)',
    'Air Temperature(°C)',
    'Radiant Temperature(°C)', 'Operative Temperature(°C)',
    'Room Electricity(W/m2)', 'Lighting(W/m2)', 'System Fans(W/m2)',
    'System Pumps(W/m2)', 'Cooling (Electricity)(W/m2)', 'Exterior lighting(W/m2)',
    'Outside Dry-Bulb Temperature(°C)'
]

def read_excel_data(file):
    df = pd.read_excel(file)
    return df

def randomize_energy_data(df):
    for column in columns_to_optimize:
        df[column] = df[column] * random.uniform(0.8, 1.2)
    return df

def calculate_optimization_cost(initial_df, optimized_df):
    initial_energy = sum(initial_df[columns_to_optimize].sum()) * random.uniform(0.8, 1.2)
    optimized_energy = sum(optimized_df[columns_to_optimize].sum())
    cost = round(initial_energy - optimized_energy, 2)
    return cost

def create_energy_plot(df, initial_df):
    plt.figure(figsize=(10, 6))
    for column in columns_to_optimize:
        color = "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        plt.plot(df['Date/Time'], df[column], label=column, color=color)
    df['HVAC Adjustment'] = df['Zone Sensible Cooling(W/m2)'] - initial_df['Zone Sensible Cooling(W/m2)']
    plt.plot(df['Date/Time'], df['HVAC Adjustment'], label='HVAC Adjustment', linestyle='--')
    plt.xlabel('Date/Time')
    plt.ylabel('Energy Consumption')
    plt.legend()
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    return img_data, df

def energy_optimization_view(request):
    building_data = UploadBuildingData.objects.first()

    if not building_data:
        return HttpResponse("No data available.")
    
    initial_df = read_excel_data(building_data.upload_file)
    optimized_df = randomize_energy_data(initial_df.copy())
    img_data, optimized_df = create_energy_plot(optimized_df, initial_df)
    optimization_cost = calculate_optimization_cost(initial_df, optimized_df)
    initial_total_energy = sum(initial_df[columns_to_optimize].sum()) * random.uniform(0.8, 1.2)
    optimized_total_energy = sum(optimized_df[columns_to_optimize].sum())

    context = {
        'image': img_data,
        'optimization_cost': optimization_cost,
        'initial_total_energy': initial_total_energy,
        'optimized_total_energy': optimized_total_energy,
    }
    return render(request, 'bms_optimization.html', context)


# views.py



def predict_energy_consumption(request):
    if request.method == 'POST':
        form = EnergyPredictionForm(request.POST)
        if form.is_valid():
            building_name = form.cleaned_data['building_name']
            future_date = form.cleaned_data['future_date']
            building_data = UploadBuildingData.objects.filter(building__name=building_name).first()
            if building_data:
                df = pd.read_excel(building_data.upload_file)
                df['Date/Time'] = pd.to_datetime(df['Date/Time'])
                df['Hour'] = df['Date/Time'].dt.hour
                df['Day'] = df['Date/Time'].dt.day
                df['Month'] = df['Date/Time'].dt.month
                df['Year'] = df['Date/Time'].dt.year
                X = df[['Hour', 'Day', 'Month', 'Year']]
                y = df['General Lighting(W/m2)'] 
                model = LinearRegression()
                model.fit(X, y)
                future_data = {
                    'Hour': future_date.hour,
                    'Day': future_date.day,
                    'Month': future_date.month,
                    'Year': future_date.year
                }
                predicted_energy = model.predict([list(future_data.values())])[0]
                initial_energy = y.mean() 
                optimization_cost = initial_energy - predicted_energy
                optimization_cost = round(optimization_cost, 2)

                # Create a graph
                plt.figure(figsize=(10, 6))
                plt.plot(df['Date/Time'], y, label='Historical Energy Consumption')
                plt.axvline(x=future_date, color='red', linestyle='--', label='Future Date and Time')
                plt.xlabel('Date/Time')
                plt.ylabel('Energy Consumption')
                plt.legend()

                # Save the graph to an image
                img_buffer = BytesIO()
                plt.savefig(img_buffer, format='png')
                img_buffer.seek(0)
                img_data = base64.b64encode(img_buffer.read()).decode()
                plt.close()

                context = {
                    'form': form,
                    'predicted_energy': predicted_energy,
                    'optimization_cost': optimization_cost,
                    'image': img_data,
                }
                return render(request, 'demand_response.html', context)
    else:
        form = EnergyPredictionForm()

    context = {
        'form': form,
    }
    return render(request, 'demand_response.html', context)
