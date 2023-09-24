def optimize_energy_data(data):
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