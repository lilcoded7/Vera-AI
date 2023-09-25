from django.contrib import admin
from .models import CsvOrExcelUpload, EnergyData, BMSAPI
# Register your models here.


admin.site.register(CsvOrExcelUpload)
admin.site.register(EnergyData)
admin.site.register(BMSAPI)