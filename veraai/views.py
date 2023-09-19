import pandas as pd
import numpy as np 
from django.shortcuts import render
from .forms import TimestampForm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from django.http import JsonResponse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import io
import base64
from datetime import datetime
from django.http import HttpResponse
from io import BytesIO



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
    return render(request, 'home.html', context)


def index(request):
    return render(request, 'index.html')


def monitor_energy(request):
    # Load the historical combined data Excel file
    df = pd.read_excel('veraai.xlsx')

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
    
    return render(request, 'monitor_energy.html', context)



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

    return render(request, 'energy_optimization.html', context)


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

    return render(request, 'demand_response.html', context)






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
    return render(request, 'adjust_hvac.html', context)
