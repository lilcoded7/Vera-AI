import pandas as pd
import numpy as np 
from django.shortcuts import render, redirect
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from .models import *
from rest_framework import generics, mixins, viewsets, status
from .serializers import *
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from django.http import JsonResponse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.contrib.auth.decorators import login_required
from rest_framework.response import Response
import io
import base64
from datetime import datetime
from django.http import HttpResponse, Http404
from io import BytesIO
import requests 
from .forms import EnergyPredictionForm


class VINSINCBUILDINGDATAAPIView(generics.CreateAPIView):
    queryset = VINSINCBUILDINGDATA.objects.all()
    serializer_class = VINSINCBUILDINGDATASerializer
    permission_classes = [IsAuthenticated]

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


from io import BytesIO
import base64
import pandas as pd
import matplotlib.pyplot as plt
from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import VINSINCBUILDINGDATA

# Sample data structure to represent zones and their occupancy
zone_data = {
    'zone1': {
        'occupancy_column': 'occupancy',
        'lighting_column': 'general_lighting',
        'power_outlet_column': 'system_pumps',
        'hvac_column': 'cooling_electricity'
    },
    'zone2': {
        'occupancy_column': 'occupancy',
        'lighting_column': 'general_lighting',
        'power_outlet_column': 'system_pumps',
        'hvac_column': 'cooling_electricity'
    },
    # Add more zones as needed
}

@login_required
def bms_optimization(request, pk):
    user = request.user
    file_data = VINSINCBUILDINGDATA.objects.get(id=pk, user=user)  # Use VINSINCBUILDINGDATA model
    df = pd.DataFrame([vars(file_data)])  # Convert the model instance to a DataFrame

    cost_per_unit_lighting = calculate_cost_per_unit(df, 'general_lighting')
    cost_per_unit_hvac = calculate_cost_per_unit(df, 'cooling_electricity')
    cost_per_unit_power_outlet = calculate_cost_per_unit(df, 'system_pumps')

    optimized_df = optimize_energy(df, cost_per_unit_lighting, cost_per_unit_hvac)
    optimized_df = adjust_power_outlets(optimized_df, 'occupancy', 'system_pumps', cost_per_unit_power_outlet)
    optimized_df = calculate_energy_cost(optimized_df)

    total_cost = round(optimized_df[['zone1_cost', 'zone2_cost']].sum().sum(), 2)  # Round to two decimal places

    plt.plot(df['timestamp'], optimized_df['general_lighting'], label='Optimized Lighting(W/m2)')
    plt.xlabel('Timestamp')
    plt.ylabel('Energy Consumption(W/m2)')
    plt.title('Energy Optimization Results')
    plt.xticks(rotation=45)
    plt.legend()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.read()).decode()

    context = {
        'plot_image': f'data:image/png;base64,{img_base64}',
        'data_table': optimized_df.to_html(classes='table table-striped table-bordered table-hover'),
        'total_cost': total_cost
    }

    return render(request, 'bms_optimization.html', context)

def optimize_energy(data, cost_per_unit_lighting, cost_per_unit_hvac):
    # Simplified energy optimization logic
    occupancy_column = 'occupancy'
    lighting_column = 'general_lighting'
    hvac_column = 'cooling_electricity'

    data[lighting_column] = data.apply(lambda row: 0 if row[occupancy_column] == 0 else row[lighting_column], axis=1)
    data[hvac_column] = data.apply(lambda row: row[hvac_column] - 10 if row['air_temperature'] > 25 else row[hvac_column], axis=1)

    return data

def calculate_cost_per_unit(data, column_name):
    total_cost = data[column_name].sum()
    total_energy = data[column_name].count()
    return total_cost / total_energy if total_energy != 0 else 0.0

def adjust_power_outlets(data, occupancy_column, power_outlet_column, cost_per_unit_power_outlet):
    # Example: Control power outlets based on occupancy
    data[power_outlet_column] = data.apply(lambda row: turn_off_outlets(row[occupancy_column], row[power_outlet_column], cost_per_unit_power_outlet), axis=1)
    return data

def turn_off_outlets(occupancy, current_power_outlet, cost_per_unit_power_outlet):
    # Simple AI logic: Turn off power outlets if no occupancy
    if occupancy == 0:
        return 0
    else:
        return current_power_outlet

def calculate_energy_cost(data):
    # Calculate energy cost for each zone based on the adjusted data
    for zone, columns in zone_data.items():
        lighting_column = columns['lighting_column']
        power_outlet_column = columns['power_outlet_column']
        hvac_column = columns['hvac_column']

        data[f'{zone}_cost'] = (
            data[lighting_column] +
            data[power_outlet_column] +
            data[hvac_column]
        )

    return data

# def optimize_lighting_based_on_sensor(data, sensor_data):
#     for index, row in data.iterrows():
#         timestamp = row['Date/Time']
#         sensor_entry = sensor_data.get(timestamp)
#         if sensor_entry is not None:
#             energy_level = sensor_entry.energy_level
#             temperature = sensor_entry.temperature

#             # Adjust lighting energy based on energy level and temperature (modify as needed)
#             lighting_energy = row['General Lighting(W/m2)']
#             lighting_energy *= energy_level  # Adjust based on energy level
#             if temperature > 25:
#                 lighting_energy *= 0.9  # Reduce lighting energy if temperature is high

#             data.at[index, 'General Lighting(W/m2)'] = lighting_energy

#     return data

# def calculate_daily_energy_and_cost(data, cost_per_unit):
#     data['Date'] = pd.to_datetime(data['Date/Time']).dt.date
#     daily_energy = data.groupby('Date')['General Lighting(W/m2)'].sum()
#     daily_cost = daily_energy * cost_per_unit
#     return daily_energy, daily_cost

# def lighting_optimization(request, pk):
#     file_data = CsvOrExcelUpload.objects.get(id=pk)
#     cost_per_unit = 0.1  # Replace with the actual cost per unit of energy

#     if file_data.upload_file.name.endswith(('.csv', '.xls', '.xlsx')):
#         df = pd.read_csv(file_data.upload_file) if file_data.upload_file.name.endswith('.csv') else pd.read_excel(file_data.upload_file)

#         # Fetch sensor data from the BMSSENSOR model
#         sensor_data = BMSSENSOR.objects.filter(timestamp__in=df['Date/Time']).values('timestamp', 'energy_level', 'temperature')
#         sensor_data = {entry['timestamp']: entry for entry in sensor_data}

#         # Create a DataFrame with the same columns as in your file
#         optimized_df = df.copy()

#         # Apply the updated lighting optimization logic
#         optimized_df = optimize_lighting_based_on_sensor(optimized_df, sensor_data)

#         # Calculate daily energy consumption and cost
#         daily_energy, daily_cost = calculate_daily_energy_and_cost(optimized_df, cost_per_unit)

#         # Create a daily graph for lighting energy
#         plt.figure(figsize=(12, 6))
#         plt.plot(daily_energy.index, daily_energy, label='Daily Lighting Energy(W/m2)')
#         plt.xlabel('Date')
#         plt.ylabel('Lighting Energy(W/m2)')
#         plt.title('Daily Lighting Energy Optimization Results')
#         plt.xticks(rotation=45)
#         plt.legend()

#         # Save the daily graph to a BytesIO object
#         img = BytesIO()
#         plt.savefig(img, format='png')
#         img.seek(0)
#         img_base64 = base64.b64encode(img.read()).decode()

#         # Calculate the total cost
#         total_cost = daily_cost.sum()

#         # Render the template with data
#         context = {
#             'img_base64': img_base64,
#             'total_cost': total_cost,
#             'optimization_data': optimized_df.to_html(classes='table table-striped table-bordered table-hover'),
#             'is_sensor_connected': True,  # Replace with the actual sensor status
#         }
#         return render(request, 'lighting_optimization.html', context)
#     else:
#         print("Unsupported file format")
#         return HttpResponse("Unsupported file format")


# # Sample data structure to represent zones and their occupancy
# zone_data = {
#     'zone1': {
#         'occupancy_column': 'Occupancy(W/m2)',
#         'lighting_column': 'General Lighting(W/m2)',
#         'power_outlet_column': 'System Pumps(W/m2)',
#         'hvac_column': 'Cooling (Electricity)(W/m2)'
#     },
#     'zone2': {
#         'occupancy_column': 'Occupancy(W/m2)',
#         'lighting_column': 'General Lighting(W/m2)',
#         'power_outlet_column': 'System Pumps(W/m2)',
#         'hvac_column': 'Cooling (Electricity)(W/m2)'
#     },
#     # Add more zones as needed
# }


# def connect_to_bms(office_data, request, file_data):
#     try:
#         # Get BMS sensor data based on user and timestamp
#         bms_sensor = BMSSENSOR.objects.get(user=request.user, timestamp=file_data.timestamp)
#         bms_data = {
#             'energy_level': bms_sensor.energy_level,
#             'temperature': bms_sensor.temperature
#         }

#         # Clean column names by stripping leading and trailing spaces
#         df_columns = [col.strip() for col in office_data.columns]

#         # Update office_data with BMS data
#         for key, value in bms_data.items():
#             if key in df_columns:
#                 office_data[key] = value

#     except BMSSENSOR.DoesNotExist:
#         # Handle the case where the BMSSENSOR with the specified user and timestamp does not exist
#         print("BMSSENSOR does not exist.")
#     except Exception as e:
#         # Handle any other unexpected exceptions
#         print(f"An error occurred: {str(e)}")

#     # Return the updated office_data
#     return office_data

# def optimize_energy_ai(data, cost_per_unit_lighting, cost_per_unit_hvac, cost_per_unit_power_outlet):
#     for zone, columns in zone_data.items():
#         occupancy_column = columns['occupancy_column']
#         lighting_column = columns['lighting_column']
#         power_outlet_column = columns['power_outlet_column']
#         hvac_column = columns['hvac_column']

#         data = adjust_lighting(data, occupancy_column, lighting_column, cost_per_unit_lighting)
#         data = adjust_hvac(data, hvac_column, cost_per_unit_hvac)
#         data = adjust_power_outlets(data, occupancy_column, power_outlet_column, cost_per_unit_power_outlet)

#     return data

# def adjust_lighting(data, occupancy_column, lighting_column, cost_per_unit_lighting):
#     # Example: Adjust lighting based on occupancy
#     data[lighting_column] = data.apply(lambda row: adjust_brightness(row[occupancy_column], row[lighting_column], cost_per_unit_lighting), axis=1)
#     return data

# def adjust_brightness(occupancy, current_lighting, cost_per_unit_lighting):
#     # Simple AI logic: Reduce lighting if no occupancy, increase if occupancy
#     if occupancy == 0:
#         return current_lighting - 10 * cost_per_unit_lighting
#     else:
#         return current_lighting + 10 * cost_per_unit_lighting

# def adjust_hvac(data, hvac_column, cost_per_unit_hvac):
#     # Example: Adjust HVAC based on room temperature
#     data[hvac_column] = data.apply(lambda row: adjust_temperature(row['Air Temperature(°C)'], row[hvac_column], cost_per_unit_hvac), axis=1)
#     return data

# def adjust_temperature(temperature, current_hvac, cost_per_unit_hvac):
#     # Simple AI logic: Decrease HVAC if too warm, increase if too cold
#     if temperature > 25:
#         return current_hvac - 10 * cost_per_unit_hvac
#     elif temperature < 20:
#         return current_hvac + 10 * cost_per_unit_hvac
#     else:
#         return current_hvac

# def adjust_power_outlets(data, occupancy_column, power_outlet_column, cost_per_unit_power_outlet):
#     # Example: Control power outlets based on occupancy
#     data[power_outlet_column] = data.apply(lambda row: turn_off_outlets(row[occupancy_column], row[power_outlet_column], cost_per_unit_power_outlet), axis=1)
#     return data

# def turn_off_outlets(occupancy, current_power_outlet, cost_per_unit_power_outlet):
#     # Simple AI logic: Turn off power outlets if no occupancy
#     if occupancy == 0:
#         return 0
#     else:
#         return current_power_outlet

# def calculate_energy_cost(data):
#     # Calculate energy cost for each zone based on the adjusted data
#     for zone, columns in zone_data.items():
#         lighting_column = columns['lighting_column']
#         power_outlet_column = columns['power_outlet_column']
#         hvac_column = columns['hvac_column']

#         data[f'{zone}_cost'] = (
#             data[lighting_column] +
#             data[power_outlet_column] +
#             data[hvac_column]
#         )

#     return data

# def calculate_cost_per_unit(data, column_name):
#     # Calculate cost per unit for a given column (e.g., 'General Lighting(W/m2)')
#     total_cost = data[column_name].sum()
#     total_energy = data[column_name].count()
#     return total_cost / total_energy if total_energy != 0 else 0.0

# @login_required
# def bms_optimization(request, pk):
#     user = request.user
#     file_data = CsvOrExcelUpload.objects.get(id=pk, user=user)
#     if file_data.upload_file.name.endswith(('.csv', '.xls', '.xlsx')):
#         df = pd.read_csv(file_data.upload_file) if file_data.upload_file.name.endswith('.csv') else pd.read_excel(file_data.upload_file)

#         # Calculate COST_PER_UNIT_LIGHTING, COST_PER_UNIT_HVAC, and COST_PER_UNIT_POWER_OUTLET
#         cost_per_unit_lighting = calculate_cost_per_unit(df, 'General Lighting(W/m2)')
#         cost_per_unit_hvac = calculate_cost_per_unit(df, 'Cooling (Electricity)(W/m2)')
#         cost_per_unit_power_outlet = calculate_cost_per_unit(df, 'System Pumps(W/m2)')

#         # Connect to the BMS to get power distribution information
#         df = connect_to_bms(df, request, file_data)
#         # Apply AI-based energy optimization logic to the data
#         optimized_df = optimize_energy_ai(df.copy(), cost_per_unit_lighting, cost_per_unit_hvac, cost_per_unit_power_outlet)
#         # Calculate energy cost for each zone
#         optimized_df = calculate_energy_cost(optimized_df)

#         # Calculate total cost for all zones
#         total_cost = optimized_df[['zone1_cost', 'zone2_cost']].sum().sum()  # Adjust column names if needed

#         # Create a line plot to visualize the results
#         plt.plot(df['Date/Time'], optimized_df['General Lighting(W/m2)'], label='Optimized Lighting(W/m2)')
#         plt.xlabel('Date/Time')
#         plt.ylabel('Energy Consumption(W/m2)')
#         plt.title('Energy Optimization Results')
#         plt.xticks(rotation=45)
#         plt.legend()

#         # Save the plot to a BytesIO object
#         img = BytesIO()
#         plt.savefig(img, format='png')
#         img.seek(0)
#         img_base64 = base64.b64encode(img.read()).decode()

#         # Pass the plot as a base64-encoded image to the template
#         context = {
#             'plot_image': f'data:image/png;base64,{img_base64}',
#             'data_table': optimized_df.to_html(classes='table table-striped table-bordered table-hover'),
#             'total_cost': total_cost
#         }

#         return render(request, 'bms_optimization.html', context)
#     else:
#         print("Unsupported file format")
#         return HttpResponse("Unsupported file format")

# @login_required
# def energy_prediction(request):
#     form = EnergyPredictionForm(request.POST or None)
#     predicted_energy = None
#     total_cost = None
#     plot = None

#     if request.method == 'POST' and form.is_valid():
#         building_name = form.cleaned_data['building_name']
#         future_date = form.cleaned_data['future_date']

#         try:
#             csv_excel_upload = CsvOrExcelUpload.objects.get(building_name=building_name)
#             file_path = csv_excel_upload.upload_file.path
#         except CsvOrExcelUpload.DoesNotExist:
#             return render(request, 'demand_response.html', {'form': form, 'error': 'Building not found'})

#         data = pd.read_excel(file_path)  # Use pd.read_csv if it's a CSV file

#         X = data[['External Infiltration (W/m2)', 'Task Lighting(W/m2)', 'General Lighting(W/m2)', 'Miscellaneous(W/m2)']]
#         y = data['Zone Sensible Cooling(W/m2)']

#         model = LinearRegression()
#         model.fit(X, y)

#         future_data = pd.DataFrame({
#             'External Infiltration (W/m2)': [0],  # Replace with your future data
#             'Task Lighting(W/m2)': [0],
#             'General Lighting(W/m2)': [0],
#             'Miscellaneous(W/m2)': [0]
#         })

#         predicted_energy = model.predict(future_data)[0]

#         # Define the cost rate (cost per unit of energy)
#         cost_rate = 0.10  # Change this to the appropriate rate (e.g., $0.10 per kWh)

#         # Calculate the total consumption cost
#         total_cost = predicted_energy * cost_rate

#         # Create a plot for the predicted energy consumption
#         plt.figure(figsize=(8, 6))
#         plt.plot(data.index, data['Zone Sensible Cooling(W/m2)'], label='Actual Energy Consumption', marker='o')
#         plt.axvline(x=len(data), color='r', linestyle='--', label='Future Date')
#         plt.scatter(len(data), predicted_energy, color='g', marker='s', s=100, label='Predicted Energy Consumption')
#         plt.xlabel('Time (Data Points)')
#         plt.ylabel('Energy Consumption (W/m2)')
#         plt.title('Predicted Energy Consumption Over Time')
#         plt.legend()

#         # Save the plot to a BytesIO object
#         buffer = BytesIO()
#         plt.savefig(buffer, format='png')
#         buffer.seek(0)
#         plot = base64.b64encode(buffer.read()).decode()
#         buffer.close()

#     return render(request, 'demand_response.html', {'form': form, 'predicted_energy': predicted_energy, 'total_cost': total_cost, 'plot': plot})
