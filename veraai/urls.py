from django.urls import path
from . import views

urlpatterns = [
    path('connect/bms/api/', views.VINSINCBUILDINGDATACreateUpdateView.as_view(), name='bms-data-api'),#1
    path('bms/api/v1/', views.BuildingAPIVIew.as_view(), name='register-building'),#2
    path('upload/file/data/', views.UploadBuildingDataCreateUpdateView.as_view(), name='upload-file-data-api'),
    path('lighting/optimization/', views.plot_energy_data)

]
