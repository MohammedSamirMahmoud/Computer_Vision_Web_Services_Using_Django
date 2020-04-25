""""
Configuring the 3 CV tasks call backs. This where all the action happens (Backend)

Each of one those does the following:

    A - Handles the POST request of the upload button. If not file uploaded, the basic template (without predictions) is rendered.
    B - Passes the uploaded image file to the prediction model.
    C - Renders back the html template with the returned prediction from the model.
        In cases of object detection and segmentation, those are the paths of the saved images.
"""

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
    return render(request, 'App_1_CV_Images/base.html')


# Image Classification Part

def classification(request):
    if request.method == 'POST' and request.FILES['myfile']:
        my_file = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(my_file.name, my_file)
        img_file = fs.url(filename)

        # Img is a PIL image of size 224*224
        img_file = settings.BASE_DIR + '/' + img_file
        img = image.load_img(img_file, target_size=(224, 224))
        # Converting img to a numpy array --> `img_as_a_numpy_array` is a float32 Numpy array of shape (224, 224, 3)
        img_as_a_numpy_array = image.img_to_array(img)

        # Transforming Array into Batch of size (1,244,244,3) -> by adding extra dim
        img_as_a_numpy_array = np.expand_dims(img_as_a_numpy_array, axis=0)

        # Image Pre_processing
        img_as_a_numpy_array = preprocess_input(img_as_a_numpy_array)  # Channel-Wise color normalization

        # Applying Model (Pre_trained Model)
        model = VGG16(weights='imagenet', include_top=True)

        # Making Predictions on passed image (img_as_a_numpy_array)
        predictions = model.predict(img_as_a_numpy_array)
        print('Predicted:', decode_predictions(predictions, top=3)[0])
        prediction = decode_predictions(predictions, top=1)[0][0][1]

        # Returning Final output to the frond end
        return render(request, 'App_1_CV_Images/classification.html', {'original_img': img_file,
                                                                       'prediction': prediction})

    return render(request, 'App_1_CV_Images/classification.html')


# Image Segmentation Part

def load_segmentation_model():
    # Load Segmentation model for inference
    seg_model = models.segmentation.fcn_resnet101(pretrained=True).eval()
    return seg_model


def get_segmentation(img_file, model):
    input_image = Image.open(img_file)
    pre_process = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = pre_process(input_image)
    input_batch = input_tensor.unsqueeze(0)  # creating a mini-batch as expected by the model

    # For GPU Availability ( Do all the Job in GPU )
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions


label_colors = np.array([(0, 0, 0),  # 0=background
                         # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                         (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                         # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                         (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                         # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                         (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                         # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                         (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])


def seg2rgb(preds):
    colors = label_colors
    colors = label_colors.astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    rgb = Image.fromarray(preds.byte().cpu().numpy())  # .resize(preds.shape)
    rgb.putpalette(colors)
    return rgb


def semantic_segmentation(request):
    if request.method == 'POST' and request.FILES['myfile']:
        my_file = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(my_file.name, my_file)
        img_file = fs.url(filename)

        # Img is a PIL image of size 224*224
        img_file_dir = settings.BASE_DIR + '/' + img_file
        img = Image.open(img_file_dir)

        # Applying Model (Pre_trained Model)
        model = load_segmentation_model()

        # Making Predictions on passed image (img_as_a_numpy_array)
        predictions = get_segmentation(img_file_dir, model)
        rgb = seg2rgb(predictions)

        # Saving Segmented Image to be show later
        seg_file = settings.MEDIA_ROOT + '/seg_img.png'
        rgb.save(seg_file)

        # Returning Final output to the frond end
        return render(request, 'App_1_CV_Images/semantic_segmentation.html', {'original_img': img_file,
                                                                              'segmented_img': '/media/seg_img.png'})

    return render(request, 'App_1_CV_Images/semantic_segmentation.html')

def object_detection(request):
    print('Nthing')
    return render(request, 'App_1_CV_Images/object_detection.html')
