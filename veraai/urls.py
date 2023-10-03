from django.urls import path
from . import views

urlpatterns = [
    path('connect/bms/api/', views.VINSINCBUILDINGDATACreateUpdateView.as_view(), name='bms-data-api'),#1
    path('bms/api/v1/api/', views.BuildingAPIVIew.as_view(), name='add-bms'),#2
    path('upload/file/data/api/', views.UploadBuildingDataCreateUpdateView.as_view(), name='upload-file-data-api'),
    path('energy/consumption/prediction/', views.predict_energy_consumption, name='predic-energy'),
    path('bms/energy/optimization/', views.energy_optimization_view, name='energy-optimization'),

]
