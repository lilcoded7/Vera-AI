import pandas as pd
from .models import VINSINCBUILDINGDATA
from datetime import datetime

def scrape_data_from_file_and_save(user_object, file_path):
    try:
        # Determine the file type based on the file extension
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print("Unsupported file format")
            return False

        # Iterate over each row in the DataFrame and create or update VINSINCBUILDINGDATA objects
        for index, row in df.iterrows():
            timestamp = datetime.strptime(str(row['Date/Time']), '%Y-%m-%d %H:%M:%S')
            
            # Check if a record with the same timestamp already exists
            data, created = VINSINCBUILDINGDATA.objects.get_or_create(timestamp=timestamp, user=user_object)

            # Update the fields with new values
            data.mech_vent_nat_vent_infiltration = row['Mech Vent + Nat Vent + Infiltration(ac/h)']
            data.external_infiltration = row['External Infiltration (W/m2)']
            data.task_lighting = row['Task Lighting(W/m2)']
            data.general_lighting = row['General Lighting(W/m2)']
            data.miscellaneous = row['Miscellaneous(W/m2)']
            data.process = row['Process(W/m2)']
            data.computer_equip = row['Computer + Equip(W/m2)']
            data.occupancy = row['Occupancy(W/m2)']
            data.solar_gains_exterior_windows = row['Solar Gains Exterior Windows(W/m2)']
            data.zone_sensible_cooling = row['Zone Sensible Cooling(W/m2)']
            data.air_temperature = row['Air Temperature(°C)']
            data.radiant_temperature = row['Radiant Temperature(°C)']
            data.operative_temperature = row['Operative Temperature(°C)']
            data.room_electricity = row['Room Electricity(W/m2)']
            data.lighting = row['Lighting(W/m2)']
            data.system_fans = row['System Fans(W/m2)']
            data.system_pumps = row['System Pumps(W/m2)']
            data.cooling_electricity = row['Cooling (Electricity)(W/m2)']
            data.exterior_lighting = row['Exterior lighting(W/m2)']
            data.outside_dry_bulb_temperature = row['Outside Dry-Bulb Temperature(°C)']
            
            
            data.save()

        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False
