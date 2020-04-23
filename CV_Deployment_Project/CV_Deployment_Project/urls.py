"""CV_Deployment_Project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('App_1_CV_Images/', include('App_1_CV_Images.urls')),
    path('App_2_CV_Videos/', include('App_2_CV_Videos.urls')),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
'''
The last line is particularly important, since the MEDIA_URL is the path to the /media/ directory, 
which has all the image files (input and output), that the code will use. So it’s impartant to have that URL configured 
so that the templates can find those images when passed from the backend. In other words, the /media/ directory is the 
link or shared space the links the backend and the front end.
'''