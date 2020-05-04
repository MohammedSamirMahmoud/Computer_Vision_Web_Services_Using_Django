from django.urls import path

from . import views

urlpatterns = [
    path('', views.base, name='base'),
    path('Classification', views.classification, name='Classification'),
    path('Semantic_Segmentation', views.semantic_segmentation, name='Semantic_Segmentation'),
    path('Object_Detection', views.object_detection, name='Object_Detection'),
    path('feedback_form', views.feedback_form, name="feedback_form"),
]
