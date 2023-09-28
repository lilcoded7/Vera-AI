from django.urls import path
from . import views

urlpatterns = [
    path('connect/bms/api/', views.VINSINCBUILDINGDATAAPIView.as_view(), name='bms-data-api'),#5
    path('bms/api/v1/', views.BuildingAPIView.as_view(), name='bms-api'),#5



    path('lighting/optimization/', views.energy_optimization, name='bsm-optimization'), #2
    path('energy/prediction/', views.energy_prediction, name='energy-prediction'),#3
    

]
