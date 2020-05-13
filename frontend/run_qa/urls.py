from django.urls import path
from . import views

app_name = 'run_qa'

urlpatterns = [
    path('', views.index, name='index'),
]