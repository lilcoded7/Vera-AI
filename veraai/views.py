import pandas as pd
import numpy as np 
from django.shortcuts import render, redirect
from .forms import TimestampForm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from .models import CsvOrExcelUpload, EnergyData, BMSAPI
from rest_framework import generics, mixins, viewsets
from .serializers import *
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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


@login_required
def predict_energy_for_selected_timestamp(selected_timestamp):
    # Load the historical combined data Excel file
    df = pd.read_excel('veraai.xlsx')

    # Feature engineering: Extract relevant features from the timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Select relevant features and target variable
    X = df[['temperature', 'humidity', 'occupancy', 'energy_cost']]
    y = df['energy_consumption']

    # Standardize the feature values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Initialize and train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict energy consumption for the selected timestamp
    selected_timestamp = pd.to_datetime(selected_timestamp)
    selected_data = pd.DataFrame({'timestamp': [selected_timestamp]})

    # Select relevant features for the selected timestamp
    X_selected = selected_data.assign(temperature=0, humidity=0, occupancy=0, energy_cost=0)[['temperature', 'humidity', 'occupancy', 'energy_cost']]

    # Standardize the selected data (use the same scaler as before)
    X_selected = scaler.transform(X_selected)

    # Make predictions for energy consumption at the selected timestamp
    energy_prediction = model.predict(X_selected)[0]

    # Determine whether energy consumption is high or low based on a threshold (you can adjust this threshold)
    threshold = 50  # You can adjust this threshold based on your data
    consumption_category = "High" if energy_prediction > threshold else "Low"

    # Generate the energy consumption graph
    plt.figure(figsize=(10, 6))
    # Modify this section to plot the graph based on your data (e.g., actual consumption data here).
    # For example, you can plot historical consumption data here.
    plt.plot([selected_timestamp], [energy_prediction], marker='o', linestyle='-', color='b', label=f'Predicted Energy Consumption ({consumption_category})')
    plt.axhline(y=50, color='r', linestyle='--', label='Threshold (High/Low)')
    plt.title('Predicted Energy Consumption and Threshold')
    plt.xlabel('Timestamp')
    plt.ylabel('Energy Consumption')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    
    # Save the graph to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    buffer.close()

    return energy_prediction, consumption_category, plot_data

@login_required
def home(request):
    form = TimestampForm(request.POST or None)
    energy_prediction = None
    consumption_category = None
    graph_data = None

    if form.is_valid():
        selected_timestamp = form.cleaned_data['timestamp']
        energy_prediction, consumption_category, graph_data = predict_energy_for_selected_timestamp(selected_timestamp)

    form = TimestampForm()
    context = {
        'form': form,
        'energy_prediction': energy_prediction,
        'consumption_category': consumption_category,
        'graph_data': graph_data,
    }
    return render(request, 'home.html')




def optimize_energy(data):
    # Simplified energy optimization logic (e.g., control lighting, HVAC, and power outlets)
    occupancy_column = 'Occupancy(W/m2)'
    lighting_column = 'General Lighting(W/m2)'
    hvac_column = 'Cooling (Electricity)(W/m2)'

    # Example: Turn off lights if no occupancy
    data[lighting_column] = data.apply(lambda row: 0 if row[occupancy_column] == 0 else row[lighting_column], axis=1)

    # Example: Adjust HVAC based on room temperature
    temperature_column = 'Air Temperature(°C)'
    data[hvac_column] = data.apply(lambda row: row[hvac_column] - 10 if row[temperature_column] > 25 else row[hvac_column], axis=1)

    return data


@login_required
def index(request, pk):
    file_data = CsvOrExcelUpload.objects.get(id=pk)

    if file_data.upload_file.name.endswith(('.csv', '.xls', '.xlsx')):
        df = pd.read_csv(file_data.upload_file) if file_data.upload_file.name.endswith('.csv') else pd.read_excel(file_data.upload_file)

        # Apply energy optimization logic to the data
        optimized_df = optimize_energy(df.copy())

        # Create a line plot to visualize the results
        plt.plot(df['Date/Time'], optimized_df['General Lighting(W/m2)'], label='Optimized Lighting(W/m2)')
        plt.plot(df['Date/Time'], optimized_df['Cooling (Electricity)(W/m2)'], label='Optimized HVAC(W/m2)')
        plt.xlabel('Date/Time')
        plt.ylabel('Energy Consumption(W/m2)')
        plt.title('Energy Optimization Results')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Save the plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.read()).decode()

        # Pass the plot as a base64-encoded image to the template
        context = {
            'plot_image': f'data:image/png;base64,{img_base64}',
            'data_table': optimized_df.to_html(classes='table table-striped table-bordered table-hover')
        }

        return render(request, 'index.html', context)

    else:
        print("Unsupported file format")
        return HttpResponse("Unsupported file format")


# Sample data structure to represent zones and their occupancy
zone_data = {
    'zone1': {
        'occupancy_column': 'Occupancy(W/m2)',
        'lighting_column': 'General Lighting(W/m2)',
        'power_outlet_column': 'System Pumps(W/m2)',
        'hvac_column': 'Cooling (Electricity)(W/m2)'
    },
    'zone2': {
        'occupancy_column': 'Occupancy(W/m2)',
        'lighting_column': 'General Lighting(W/m2)',
        'power_outlet_column': 'System Pumps(W/m2)',
        'hvac_column': 'Cooling (Electricity)(W/m2)'
    },
    # Add more zones as needed
}



class BMSApiView(generics.GenericAPIView):
    queryset = BMSAPI.objects.all()
    serializer_class = BMSAPISerializer
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # Get the data from the request
        data = request.data
        bms_api_url = data.get('bms_api_url')
        try:
            bms_api = BMSAPI.objects.get(bms_api_url=bms_api_url)
        except BMSAPI.DoesNotExist:
            serializer = self.serializer_class(data=data)
            if serializer.is_valid():
                serializer.save()
                Response({'message': 'BMS Connected'})
                return redirect('energy-optimization', request.user.id)
            else:
                return Response({'message': 'Connection fail!'})
        else:
            serializer = self.serializer_class(instance=bms_api, data=data)
            if serializer.is_valid():
                serializer.save()
                Response({'message': 'BMS Updated'})
                return redirect('energy-optimization', request.user.id )
            else:
                return Response({'message': 'Update failed!'})

        return Response({'message': 'Unknown error occurred'})


class CsvOrExcelUploadpiView(generics.GenericAPIView):
    queryset = CsvOrExcelUpload.objects.all()
    serializer_class = CsvOrExcelUploadSerializer
    permission_classes = [IsAuthenticated]
    def post(self, request):
        data = request.data
        building_name = data.get('building_name')
        try:
            csv_upload = CsvOrExcelUpload.objects.get(building_name=building_name)
        except CsvOrExcelUpload.DoesNotExist:
            serializer = self.serializer_class(data=data)
            if serializer.is_valid():
                serializer.save()
                Response({'message': 'Upload created'})
                return redirect('energy-optimization', request.user.id)
            else:
                return Response(serializer.errors, status=400)
        else:
            serializer = self.serializer_class(instance=csv_upload, data=data)
            if serializer.is_valid():
                serializer.save()
                Response({'message': 'Upload updated'})
                return redirect('energy-optimization', request.user.id)
            else:
                return Response(serializer.errors, status=400)
        return Response({'message': 'Unknown error occurred'})



def connect_to_bms(office_data, request, pk):
    try:
        BMS_API_URL = BMSAPI.objects.get(id=pk, user=request.user)
        response = requests.get(BMS_API_URL.bms_api_url)
        response.raise_for_status()
        bms_data = response.json()
        
        # Update office_data with BMS data
        for key, value in bms_data.items():
            office_data[key] = value

        return office_data
    except BMSAPI.DoesNotExist:
        # Handle the case where the BMSAPI with the specified ID does not exist
        print("BMSAPI does not exist.")
    except requests.exceptions.RequestException as e:
        # Handle any request-related errors
        print(f"Request error: {str(e)}")
    except Exception as e:
        # Handle any other unexpected exceptions
        print(f"An error occurred: {str(e)}")

    # Return the original office_data (unchanged) or handle errors as needed
    return office_data


def optimize_energy_bms(data):
    for zone, columns in zone_data.items():
        occupancy_column = columns['occupancy_column']
        lighting_column = columns['lighting_column']
        power_outlet_column = columns['power_outlet_column']
        hvac_column = columns['hvac_column']

        # Example: Turn off lights if no occupancy
        data[lighting_column] = data.apply(lambda row: 0 if row[occupancy_column] == 0 else row[lighting_column], axis=1)

        # Example: Adjust HVAC based on room temperature
        data[hvac_column] = data.apply(lambda row: row[hvac_column] - 10 if row['Air Temperature(°C)'] > 25 else row[hvac_column], axis=1)

        # Example: Control power outlets based on occupancy
        data[power_outlet_column] = data.apply(lambda row: 0 if row[occupancy_column] == 0 else row[power_outlet_column], axis=1)

    return data


@login_required
def bms_optimization(request, pk):
    user = request.user
    file_data = CsvOrExcelUpload.objects.get(id=pk, user=user)
    if file_data.upload_file.name.endswith(('.csv', '.xls', '.xlsx')):
        df = pd.read_csv(file_data.upload_file) if file_data.upload_file.name.endswith('.csv') else pd.read_excel(file_data.upload_file)
        # Connect to the BMS to get power distribution information
        df = connect_to_bms(df, request, pk)
        # Apply energy optimization logic to the data
        optimized_df = optimize_energy_bms(df.copy())

        # Create a line plot to visualize the results
        plt.plot(df['Date/Time'], optimized_df['General Lighting(W/m2)'], label='Optimized Lighting(W/m2)')
        plt.xlabel('Date/Time')
        plt.ylabel('Energy Consumption(W/m2)')
        plt.title('Energy Optimization Results')
        plt.xticks(rotation=45)
        plt.legend()

        # Save the plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.read()).decode()

        # Pass the plot as a base64-encoded image to the template
        context = {
            'plot_image': f'data:image/png;base64,{img_base64}',
            'data_table': optimized_df.to_html(classes='table table-striped table-bordered table-hover')
        }

        return render(request, 'bms_optimization.html', context)
    else:
        print("Unsupported file format")
        return HttpResponse("Unsupported file format")
    
@login_required
def monitor_energy(request):
    # Load the historical combined data Excel file
    file_data = CsvOrExcelUpload.objects.get(id=1)
    if file_data.upload_file.name.endswith('.csv'):
        df = pd.read_csv(file_data.upload_file)

    elif file_data.upload_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_data.upload_file)
    else:
        print("Unsupported file format")

    # Assuming the data is sorted by timestamp in descending order, get the latest data point
    latest_data = df.iloc[0]

    # Train an isolation forest model on your historical data
    X_train = df[['temperature', 'humidity', 'occupancy', 'energy_cost']]
    isolation_forest = IsolationForest(contamination=0.05)  # You can adjust the contamination parameter
    isolation_forest.fit(X_train)

    # Retrieve the latest real-time data from the Excel file
    current_data = {
        'timestamp': latest_data['timestamp'],
        'temperature': latest_data['temperature'],
        'humidity': latest_data['humidity'],
        'occupancy': latest_data['occupancy'],
        'energy_cost': latest_data['energy_cost'],
    }

    # Predict whether the current data is an anomaly
    X_current = np.array([[current_data['temperature'], current_data['humidity'], current_data['occupancy'], current_data['energy_cost']]])
    is_anomaly = isolation_forest.predict(X_current)
    
    # Convert is_anomaly to a string
    is_anomaly = str(int(is_anomaly[0]))  # Convert to a string after converting to an integer

    # Generate a graph for the latest data
    plt.figure(figsize=(10, 6))
    plt.plot([current_data['timestamp']], [latest_data['energy_consumption']], marker='o', linestyle='-', color='b', label=f'Energy Consumption')
    plt.title('Real-Time Energy Consumption')
    plt.xlabel('Timestamp')
    plt.ylabel('Energy Consumption')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Save the graph to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    buffer.close()

    # Create a context dictionary with the data to pass to the template
    context = {
        'timestamp': current_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
        'is_anomaly': is_anomaly, 
        'data': current_data,
        'graph_data': plot_data,
    }
    
    return render(request, 'monitor_energy.html')


@login_required
def energy_optimization(request):
    df = pd.read_excel('veraai.xlsx')

    features = ['temperature', 'humidity', 'occupancy', 'energy_cost']

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    num_clusters = 3  
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_pca)
    recommendations = []

    for cluster_id in range(num_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        avg_energy_cost = cluster_data['energy_cost'].mean()
        recommendation = f"Cluster {cluster_id + 1}: Adjust HVAC settings or lighting schedules to optimize energy usage. "
        recommendation += f"Average Energy Cost: ${avg_energy_cost:.2f}"
        recommendations.append(recommendation)

    current_occupancy = "Occupied"  
    system_status = "Optimized" if current_occupancy == "Occupied" else "Optimizing"
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_clusters), [cluster_data['energy_cost'].mean() for cluster_id, cluster_data in df.groupby('cluster')])
    plt.xticks(range(num_clusters), [f'Cluster {cluster_id + 1}' for cluster_id in range(num_clusters)])
    plt.xlabel('Cluster')
    plt.ylabel('Average Energy Cost')
    plt.title('Energy Cost by Cluster')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    buffer.close()

    context = {
        'recommendations': recommendations,
        'current_occupancy': current_occupancy,
        'system_status': system_status,
        'plot_data': plot_data,
    }

    return render(request, 'energy_optimization.html')

@login_required
def demand_response(request):
    df = pd.read_excel('veraai.xlsx')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    peak_demand_hours = [16, 17, 18]  
    current_hour = datetime.now().hour
    is_peak_demand = current_hour in peak_demand_hours
    if is_peak_demand:
        message = f"It's a peak demand period. The AI is automatically reducing energy consumption."
        action_message = "Implement energy reduction actions here."
    else:
        message = f"It's not a peak demand period. No action is required."
        action_message = "No energy reduction actions are needed at this time."
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['energy_consumption'], marker='o', linestyle='-')
    plt.xlabel('Timestamp')
    plt.ylabel('Energy Consumption')
    plt.title('Historical Energy Consumption')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    buffer.close()

    context = {
        'is_peak_demand': is_peak_demand,
        'current_hour': current_hour,
        'message': message,
        'action_message': action_message,
        'plot_data': plot_data,
    }

    return render(request, 'demand_response.html')





@login_required
def adjust_hvac(request):
    historical_data = pd.read_excel('veraai.xlsx')
    weather_forecast = {
        'temperature': 28, 
        'humidity': 60,
    }
    hvac_adjustment = "Optimize"
    if weather_forecast['temperature'] > 30:
        hvac_adjustment = "Cooling Mode"
    elif weather_forecast['temperature'] < 10:
        hvac_adjustment = "Heating Mode"
    plt.figure(figsize=(10, 6))
    plt.plot(historical_data['timestamp'], historical_data['temperature'], label='Temperature (°C)', color='blue')
    plt.axhline(y=weather_forecast['temperature'], color='red', linestyle='--', label='Weather Forecast')
    plt.title('HVAC Adjustment Based on Weather Forecast')
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature (°C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    buffer.close()
    historical_data['hvac_adjustment'] = hvac_adjustment
    context = {
        'hvac_adjustment': hvac_adjustment,
        'plot_data': plot_data,
        'historical_data': historical_data.iterrows(),
    }
    return render(request, 'adjust_hvac.html')
