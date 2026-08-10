from django.urls import path
from .views import upload_view, statistics_view
from .views import upload_view, statistics_view, ml_view

urlpatterns = [
    path('', upload_view, name='upload'),
    path('statistics/', statistics_view, name='statistics'),
    path("ml/", ml_view, name="ml"),
]