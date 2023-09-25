from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='energy-prediction'),
    path('home/', views.energy_optimization, name='home'),
    path('detect/anomalies/', views.monitor_energy, name='anomalies'),
    path('energy/optimization/<int:pk>', views.index, name='energy-optimization'),
    path('demand/response/status/', views.demand_response, name='demand-response-status'),
    path('adjust/hvac/', views.adjust_hvac, name='adjust_hvac'),
    path('energy/optimization/bms/<int:pk>', views.bms_optimization, name='bms-optimization'),
    path('energy/optimization/bms/api/', views.BMSApiView.as_view(), name='bms-api'),
    path('api/energy/consumption/data/', views.CsvOrExcelUploadpiView.as_view(), name='upload-file'),


]
