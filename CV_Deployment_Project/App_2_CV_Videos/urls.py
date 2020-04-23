from django.urls import path

from . import views

urlpatterns = [
    path('', views.base, name='base'),
    path('Semantic_Segmentation', views.semantic_segmentation, name='Semantic_Segmentation'),
    path('Object_Detection', views.object_detection, name='Object_Detection'),
]