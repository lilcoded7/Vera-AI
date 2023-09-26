from django.urls import path
from . import views

urlpatterns = [
    # path('energy/optimization/bms/<int:pk>', views.bms_optimization_lit, name='energy-prediction'),
    path('lighting/optimization/<int:pk>', views.lighting_optimization, name='bsm-lighting-optimization'), #2
    path('energy/optimization/bms/<int:pk>', views.bms_optimization, name='bms-optimization'), #1
    path('energy/optimization/bms/api/', views.BMSApiView.as_view(), name='bms-api'),#5


    
    
    
    path('api/energy/consumption/data/', views.CsvOrExcelUploadpiView.as_view(), name='upload-file'),#4
    path('energy/prediction/<int:pk>', views.energy_prediction, name='energy-prediction')#3


]
