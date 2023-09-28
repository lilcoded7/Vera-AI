import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, Http404
from rest_framework import generics, mixins, viewsets, status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from django.contrib.auth.decorators import login_required
from io import BytesIO
import requests
from datetime import datetime
from decimal import Decimal

import base64
from .models import *
from .serializers import *
from .forms import EnergyPredictionForm
from .utils import scrape_data_from_file_and_save
from .models import VINSINCBUILDINGDATA
from django.db.models import F
from django.db.models import Sum
from django.db.models.functions import Coalesce
import io




import base64
import requests
import io
import datetime
from decimal import Decimal
import matplotlib.pyplot as plt

WEATHER_API_KEY = 'your_weather_api_key'  # Replace with your actual weather API key

def energy_optimization(request):
    # Retrieve historical data for optimization
    historical_data = VINSINCBUILDINGDATA.objects.all()

    # Optimize energy consumption based on historical data
    optimization_feedback = optimize_energy_consumption(historical_data)

    # Calculate energy consumption cost
    energy_consumption_cost = calculate_energy_cost(historical_data)

    # Generate and save the energy consumption plot
    plot_data = generate_energy_plot(historical_data)

    feedback_message = ""

    latest_data = VINSINCBUILDINGDATA.objects.order_by('-timestamp').first()
    if latest_data:
        current_temperature = latest_data.outside_dry_bulb_temperature
        occupancy = latest_data.occupancy  # Replace with the actual attribute name

        desired_setpoint = calculate_desired_setpoint(current_temperature, occupancy)

        # Fetch the weather forecast for the next day
        forecast = get_weather_forecast()

        if forecast:
            if is_extreme_weather(forecast):
                desired_setpoint = adjust_setpoint_for_extreme_weather(desired_setpoint, forecast)

            if is_peak_demand_period(forecast):
                feedback_message = reduce_energy_consumption(latest_data)

            latest_data.operative_temperature = desired_setpoint
            latest_data.save()

            feedback_message = f"HVAC setpoint adjusted to {desired_setpoint} based on optimization. {feedback_message}"
        else:
            feedback_message = "Weather forecast data unavailable."

    return render(request, 'bms_optimization.html', {
        'feedback': optimization_feedback,
        'energy_cost': f'{energy_consumption_cost:.2f}',  # Format to 2 decimal places
        'plot': plot_data,
        'feedback_message': feedback_message  # Include feedback message in the template
    })

def optimize_energy_consumption(historical_data):
    feedback_message = "Energy optimization applied based on historical data."

    for data in historical_data:
        # Implement your energy optimization logic here
        if data.outside_dry_bulb_temperature > Decimal('28.0'):
            data.zone_sensible_cooling = Decimal('0.9')
            data.lighting *= Decimal('0.8')
        data.save()

    return feedback_message

def calculate_energy_cost(historical_data):
    total_cost = sum([data.cooling_electricity + data.lighting for data in historical_data])
    unit_cost = Decimal('0.15')  # Define your unit cost per kWh

    energy_consumption_cost = total_cost * unit_cost
    return energy_consumption_cost

def generate_energy_plot(historical_data):
    timestamps = [data.timestamp for data in historical_data]
    energy_consumption = [float(data.cooling_electricity + data.lighting) for data in historical_data]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, energy_consumption, marker='o', linestyle='-', color='b')
    plt.xlabel('Timestamp')
    plt.ylabel('Energy Consumption')
    plt.title('Energy Consumption Over Time')
    plt.grid(True)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()

    return plot_data

def get_weather_forecast():
    try:
        api_key = '9da376adc9cd8d7f7194fde24d3bd472'  # Replace with your OpenWeatherMap API key
        api_endpoint = 'https://api.openweathermap.org/data/2.5/weather'
        params = {
            'q': 'YourCity,YourCountry',  # Replace with your city and country
            'appid': api_key,
        }

        response = requests.get(api_endpoint, params=params)
        forecast = response.json()
        return forecast
    except Exception as e:
        return None

def is_extreme_weather(forecast):
    if forecast is None:
        return False

    lower_threshold = 0
    upper_threshold = 35
    temperature = forecast.get('main', {}).get('temp')
    return temperature is not None and (temperature < lower_threshold or temperature > upper_threshold)

def is_peak_demand_period(forecast):
    if forecast is None:
        return False

    forecast_time = forecast.get('dt')
    
    # Ensure that forecast_time is a valid integer
    if forecast_time is not None and isinstance(forecast_time, int):
        forecast_time = datetime.datetime.utcfromtimestamp(forecast_time)
        
        peak_start_time = datetime.time(14, 0)
        peak_end_time = datetime.time(18, 0)
        forecast_time = forecast_time.time()
        
        if forecast_time is not None and (peak_start_time <= forecast_time <= peak_end_time):
            return True

    return False

    return peak_start_time <= forecast_time <= peak_end_time
def adjust_setpoint_for_extreme_weather(desired_setpoint, forecast):
    temperature = forecast.get('main', {}).get('temp')
    if temperature < 0:
        desired_setpoint += 5
    return desired_setpoint

def reduce_energy_consumption(latest_data):
    feedback_message = ""

    if latest_data.outside_dry_bulb_temperature > Decimal('28.0'):
        latest_data.zone_sensible_cooling = 0
        feedback_message += "HVAC set to energy-saving mode due to high temperature. "

    if latest_data.lighting > Decimal('100.0'):
        latest_data.lighting *= Decimal('0.7')
        feedback_message += "Lighting reduced by 30% to save energy. "

    if latest_data.occupancy == 0:  # Replace with the actual attribute name
        latest_data.system_fans = 0
        latest_data.system_pumps = 0
        feedback_message += "Non-essential equipment turned off due to no occupancy. "

    latest_data.save()

    if not feedback_message:
        feedback_message = "No energy reduction actions taken."

    return feedback_message

def calculate_desired_setpoint(temperature, occupancy):
    # Your logic for calculating the desired setpoint based on temperature and occupancy goes here
    if occupancy:
        return 22  # Example setpoint when occupancy is detected
    else:
        return 25




class VINSINCBUILDINGDATAAPIView(generics.CreateAPIView):
    queryset = VINSINCBUILDINGDATA.objects.all()
    serializer_class = VINSINCBUILDINGDATASerializer
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user =request.user 
        scrape_data_from_file_and_save(user, 'coded.xlsx')
        return Response({'message':'connect bms api '})
    def create(self, request, *args, **kwargs):
        data = request.data
        user = request.user
        
        try:
            bms_data = VINSINCBUILDINGDATA.objects.get(user=user, timestamp=data.get('timestamp'))
        except VINSINCBUILDINGDATA.DoesNotExist:
            serializer = self.get_serializer(data=data)
            if serializer.is_valid():
                serializer.save(user=user)
                return Response({'message': 'VINSINCBUILDINGDATA Created'}, status=status.HTTP_201_CREATED)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            serializer = self.get_serializer(bms_data, data=data)
            if serializer.is_valid():
                serializer.save(user=user)
                return Response({'message': 'VINSINCBUILDINGDATA Updated'}, status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)




class BuildingAPIView(generics.CreateAPIView):
    queryset = Building.objects.all()
    serializer_class = BuildingSerializer
    permission_classes = [IsAuthenticated]

    def create(self, request, *args, **kwargs):
        data = request.data
        user = request.user

        try:
            building = Building.objects.get(user=user, name=data.get('name'))
        except Building.DoesNotExist:
            serializer = self.get_serializer(data=data)
            if serializer.is_valid():
                serializer.save(user=user)
                return Response({'message': 'Building Created'}, status=status.HTTP_201_CREATED)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            serializer = self.get_serializer(building, data=data)
            if serializer.is_valid():
                serializer.save(user=user)
                return Response({'message': 'Building Updated'}, status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)




# goog one 

def energy_prediction(request):
    if request.method == 'POST':
        form = EnergyPredictionForm(request.POST)
        if form.is_valid():
            building_name = form.cleaned_data['building_name']
            future_date = form.cleaned_data['future_date']
            
            # Fetch relevant data from the VINSINCBUILDINGDATA model
            building_data = VINSINCBUILDINGDATA.objects.filter(user=request.user, timestamp__lte=future_date).order_by('-timestamp').first()
            
            if building_data:
                # Calculate predicted energy
                predicted_energy = calculate_predicted_energy(building_data)
                
                # Calculate total cost based on predicted energy
                total_cost = calculate_total_cost(predicted_energy)
                
                # Generate the energy consumption plot
                plot = generate_energy_plot(building_data)

                return render(request, 'demand_response.html', {
                    'form': form,
                    'predicted_energy': predicted_energy,
                    'total_cost': total_cost,
                    'plot': plot
                })
            else:
                error_message = "No data found for the provided building and date."
                return render(request, 'demand_response.html', {
                    'form': form,
                    'error_message': error_message
                })

    else:
        form = EnergyPredictionForm()

    return render(request, 'demand_response.html', {'form': form})

def calculate_predicted_energy(building_data):
    # Implement your energy prediction logic here based on historical data
    # Example: Calculating energy as a sum of room electricity and lighting
    predicted_energy = building_data.room_electricity + building_data.lighting
    return predicted_energy

def calculate_total_cost(predicted_energy):
    # Implement your cost calculation logic here based on predicted energy
    # Example: Assuming a fixed rate per unit (replace with your actual rate)
    unit_cost = Decimal('0.15')  # Currency per unit as Decimal
    predicted_energy = Decimal(predicted_energy)  # Convert to Decimal if not already
    
    total_cost = predicted_energy * unit_cost
    
    # Round the total cost to two decimal places
    total_cost = total_cost.quantize(Decimal('0.00'))

    return total_cost

def generate_energy_plot(building_data):
    # Generate an energy consumption plot based on historical data
    queryset = VINSINCBUILDINGDATA.objects.filter(
        user=building_data.user,
        timestamp__lte=building_data.timestamp
    ).annotate(
        total_energy_consumption=Coalesce(
            F('room_electricity') + F('lighting'), Decimal('0.0')
        )
    ).order_by('timestamp')

    timestamps = queryset.values_list('timestamp', flat=True)
    energy_consumption = queryset.values_list('total_energy_consumption', flat=True)

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, energy_consumption, marker='o', linestyle='-', color='b')
    plt.xlabel('Timestamp')
    plt.ylabel('Energy Consumption')
    plt.title('Energy Consumption Over Time')
    plt.grid(True)

    # Save the plot as a base64 image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()

    return plot_data


