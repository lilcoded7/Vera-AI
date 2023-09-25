from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class CsvOrExcelUpload(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    building_name = models.CharField(max_length=100)
    upload_file = models.FileField()

    def __str__(self):
        return self.building_name


class BMSAPI(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    bms_api_url = models.CharField(max_length=255)

    def __str__(self):
        return self.bms_api_url

class EnergyData(models.Model):
    date_time = models.DateTimeField()
    mech_nat_vent_infiltration = models.DecimalField(max_digits=8, decimal_places=5)
    external_infiltration = models.DecimalField(max_digits=8, decimal_places=5)
    task_lighting = models.DecimalField(max_digits=8, decimal_places=5)
    general_lighting = models.DecimalField(max_digits=8, decimal_places=5)
    miscellaneous = models.DecimalField(max_digits=8, decimal_places=5)
    process = models.DecimalField(max_digits=8, decimal_places=5)
    computer_equip = models.DecimalField(max_digits=8, decimal_places=5)
    occupancy = models.DecimalField(max_digits=8, decimal_places=5)
    solar_gains = models.DecimalField(max_digits=8, decimal_places=5)
    zone_sensible_cooling = models.DecimalField(max_digits=8, decimal_places=5)
    air_temperature = models.DecimalField(max_digits=8, decimal_places=5)
    radiant_temperature = models.DecimalField(max_digits=8, decimal_places=5)
    operative_temperature = models.DecimalField(max_digits=8, decimal_places=5)
    room_electricity = models.DecimalField(max_digits=8, decimal_places=5)
    lighting = models.DecimalField(max_digits=8, decimal_places=5)
    system_fans = models.DecimalField(max_digits=8, decimal_places=5)
    system_pumps = models.DecimalField(max_digits=8, decimal_places=5)
    cooling_electricity = models.DecimalField(max_digits=8, decimal_places=5)
    exterior_lighting = models.DecimalField(max_digits=8, decimal_places=5)
    outside_dry_bulb_temperature = models.DecimalField(max_digits=8, decimal_places=5)

    def __str__(self):
        return self.data_time
        
