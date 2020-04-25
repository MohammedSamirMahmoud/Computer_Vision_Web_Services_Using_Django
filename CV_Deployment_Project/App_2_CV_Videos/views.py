# Importing All required packages and libraries (Django + ML)

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

import torch
from torchvision import models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2


# Create your views here.


def base(request):
    return render(request, 'App_2_CV_Videos/base.html')