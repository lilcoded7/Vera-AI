from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('home/', views.index),
    path('detect/anomalies/', views.monitor_energy, name='anomalies'),
    path('energy/optimization/', views.energy_optimization, name='energy-optimization'),
    path('demand/response/status/', views.demand_response, name='demand-response-status'),
    path('adjust/hvac/', views.adjust_hvac, name='adjust_hvac')
]
