from django.contrib import admin
from .models import CsvOrExcelUpload, EnergyData, BMSSENSOR
# Register your models here.


admin.site.register(CsvOrExcelUpload)
admin.site.register(EnergyData)
admin.site.register(BMSSENSOR)