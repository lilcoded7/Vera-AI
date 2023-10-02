from django.db import models
from django.contrib.auth.models import User



class Building(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name


class VINSINCBUILDINGDATA(models.Model):
    building = models.ForeignKey(Building, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()
    mech_vent_nat_vent_infiltration = models.DecimalField(max_digits=10, decimal_places=2)
    external_infiltration = models.DecimalField(max_digits=10, decimal_places=2)
    task_lighting = models.DecimalField(max_digits=10, decimal_places=2)
    general_lighting = models.DecimalField(max_digits=10, decimal_places=2)
    miscellaneous = models.DecimalField(max_digits=10, decimal_places=2)
    process = models.DecimalField(max_digits=10, decimal_places=2)
    computer_equip = models.DecimalField(max_digits=10, decimal_places=2)
    occupancy = models.DecimalField(max_digits=10, decimal_places=2)
    solar_gains_exterior_windows = models.DecimalField(max_digits=10, decimal_places=2)
    zone_sensible_cooling = models.DecimalField(max_digits=10, decimal_places=2)
    air_temperature = models.DecimalField(max_digits=5, decimal_places=2)
    radiant_temperature = models.DecimalField(max_digits=5, decimal_places=2)
    operative_temperature = models.DecimalField(max_digits=5, decimal_places=2)
    room_electricity = models.DecimalField(max_digits=10, decimal_places=2)
    lighting = models.DecimalField(max_digits=10, decimal_places=2)
    system_fans = models.DecimalField(max_digits=10, decimal_places=2)
    system_pumps = models.DecimalField(max_digits=10, decimal_places=2)
    cooling_electricity = models.DecimalField(max_digits=10, decimal_places=2)
    exterior_lighting = models.DecimalField(max_digits=10, decimal_places=2)
    outside_dry_bulb_temperature = models.DecimalField(max_digits=5, decimal_places=2)


    def __str__(self):
        return f"{self.building.user.username} - {self.timestamp}"


class UploadBuildingData(models.Model):
    building = models.ForeignKey(Building, on_delete=models.CASCADE)
    upload_file = models.FileField()

    def __str__(self) -> str:
        return self.building.name 
