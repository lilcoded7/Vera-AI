from django.urls import path
from . import views

urlpatterns = [
    path('connect/bms/api/', views.VINSINCBUILDINGDATAAPIView.as_view(), name='bms-data-api'),#5
    path('bms/api/v1/', views.BuildingAPIView.as_view(), name='bms-api'),#5
    path('energy/optimization/bms/<int:pk>', views.bms_optimization, name='bms-optimization'), #1



    # path('lighting/optimization/<int:pk>', views.lighting_optimization, name='bsm-lighting-optimization'), #2
    
    
    
    # path('api/energy/consumption/data/', views.CsvOrExcelUploadpiView.as_view(), name='upload-file'),#4
    # path('energy/prediction/', views.energy_prediction, name='energy-prediction'),#3
    

]
