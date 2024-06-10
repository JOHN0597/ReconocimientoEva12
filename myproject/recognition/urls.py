from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_biometric, name='login_biometric'),
    path('process_video/', views.process_video, name='process_video'),
    path('sign/', views.Sign, name='sign')
]