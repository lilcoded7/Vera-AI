from django.contrib import admin
from .models import VINSINCBUILDINGDATA, Building, UploadBuildingData
# Register your models here.


admin.site.register(VINSINCBUILDINGDATA)
admin.site.register(Building)
admin.site.register(UploadBuildingData)