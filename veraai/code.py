# views.py
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
from django.http import HttpResponse
from django.shortcuts import render
from .models import BMSSENSOR
from .models import CsvOrExcelUpload
from .forms import FuturePredictionForm
from django.contrib.auth.decorators import login_required

@login_required
def energy_prediction(request, pk):
    user = request.user
    file_data = CsvOrExcelUpload.objects.get(id=pk, user=user)

    if request.method == 'POST':
        form = FuturePredictionForm(request.POST)
        if form.is_valid():
            future_date = form.cleaned_data['future_date']
            future_time = form.cleaned_data['future_time']
            future_datetime = datetime.combine(future_date, future_time)
            if file_data.upload_file.name.endswith(('.csv', '.xls', '.xlsx')):
                df = pd.read_csv(file_data.upload_file) if file_data.upload_file.name.endswith('.csv') else pd.read_excel(file_data.upload_file)

                # Calculate COST_PER_UNIT_LIGHTING, COST_PER_UNIT_HVAC, and COST_PER_UNIT_POWER_OUTLET
                cost_per_unit_lighting = calculate_cost_per_unit(df, 'General Lighting(W/m2)')
                cost_per_unit_hvac = calculate_cost_per_unit(df, 'Cooling (Electricity)(W/m2)')
                cost_per_unit_power_outlet = calculate_cost_per_unit(df, 'System Pumps(W/m2)')

                # Connect to the BMS to get power distribution information
                df = connect_to_bms(df, request, file_data)

                # Create a prediction DataFrame for the selected future date and time
                future_prediction_df = create_future_prediction_df(df, future_datetime)

                # Apply AI-based energy optimization logic to the future prediction data
                optimized_future_df = optimize_energy_ai(future_prediction_df.copy(), cost_per_unit_lighting, cost_per_unit_hvac, cost_per_unit_power_outlet)

                # Calculate energy cost for each zone in the future
                optimized_future_df = calculate_energy_cost(optimized_future_df)

                # Calculate total cost for all zones in the future
                total_future_cost = optimized_future_df[['zone1_cost', 'zone2_cost']].sum().sum()  # Adjust column names if needed

                # Create a line plot to visualize the future prediction
                plt.plot(future_prediction_df['Date/Time'], optimized_future_df['General Lighting(W/m2)'], label='Optimized Lighting(W/m2)')
                plt.xlabel('Date/Time')
                plt.ylabel('Energy Consumption(W/m2)')
                plt.title('Future Energy Cost Prediction')
                plt.xticks(rotation=45)
                plt.legend()

                # Save the plot to a BytesIO object
                img = BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                img_base64 = base64.b64encode(img.read()).decode()

                # Pass the plot as a base64-encoded image to the template
                context = {
                    'form': form,
                    'plot_image': f'data:image/png;base64,{img_base64}',
                    'data_table': optimized_future_df.to_html(classes='table table-striped table-bordered table-hover'),
                    'total_cost': total_future_cost
                }

                return render(request, 'energy_prediction.html', context)
    else:
        form = FuturePredictionForm()

    context = {'form': form}
    return render(request, 'energy_prediction.html', context)

def calculate_cost_per_unit(data, column_name):
    # Calculate cost per unit for a given column (e.g., 'General Lighting(W/m2)')
    total_cost = data[column_name].sum()
    total_energy = data[column_name].count()
    return total_cost / total_energy if total_energy != 0 else 0.0

def connect_to_bms(data, request, file_data):
    # Placeholder logic to simulate BMS sensor data retrieval
    # You should replace this with actual data retrieval logic
    try:
        # Get BMS sensor data based on user and timestamp (placeholder)
        bms_sensor = BMSSENSOR.objects.get(user=request.user, timestamp=file_data.timestamp)
        bms_data = {
            'energy_level': bms_sensor.energy_level,
            'temperature': bms_sensor.temperature
        }

        # Update office_data with BMS data (placeholder)
        data.update(bms_data)

    except BMSSENSOR.DoesNotExist:
        # Handle the case where the BMSSENSOR with the specified user and timestamp does not exist
        print("BMSSENSOR does not exist.")
    except Exception as e:
        # Handle any other unexpected exceptions
        print(f"An error occurred: {str(e)}")

    return data

def optimize_energy_ai(data, cost_per_unit_lighting, cost_per_unit_hvac, cost_per_unit_power_outlet):
    # Placeholder logic for energy optimization (adjust as needed)
    for zone, columns in zone_data.items():
        occupancy_column = columns['occupancy_column']
        lighting_column = columns['lighting_column']
        power_outlet_column = columns['power_outlet_column']
        hvac_column = columns['hvac_column']

        data = adjust_lighting(data, occupancy_column, lighting_column, cost_per_unit_lighting)
        data = adjust_hvac(data, hvac_column, cost_per_unit_hvac)
        data = adjust_power_outlets(data, occupancy_column, power_outlet_column, cost_per_unit_power_outlet)

    return data

def calculate_energy_cost(data):
    # Placeholder logic to calculate energy cost for each zone (adjust as needed)
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

# Define your 'zone_data' as you did before

# Define your BMSSENSOR model

# Define other helper functions (e.g., adjust_lighting, adjust_hvac, adjust_power_outlets) as you did before
