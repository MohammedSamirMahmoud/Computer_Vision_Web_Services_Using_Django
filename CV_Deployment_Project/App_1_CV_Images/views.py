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
#from darknet import Darknet


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



# Object Detection Part

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic', 'light', 'fire', 'hydrant', 'N/A', 'stop',
    'sign', 'parking', 'meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports', 'ball',
    'kite', 'baseball', 'bat', 'baseball', 'glove', 'skateboard', 'surfboard', 'tennis',
    'racket', 'bottle', 'N/A', 'wine', 'glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot', 'dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted', 'plant', 'bed', 'N/A', 'dining', 'table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell',
    'phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy', 'bear', 'hair', 'drier', 'toothbrush',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]


def get_prediction(img_path, threshold):
    img = Image.open(img_path) # Load the image
    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
    pred = model([img]) # Pass the image to the model
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


def object_detection(request):
    if request.method == 'POST' and request.FILES['myfile']:
        
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        img_file = fs.url(filename)
        
        rect_th = 1 
        text_size = 0.2
        text_th = 1
        img_file_ = settings.BASE_DIR + '/' + img_file
        boxes, pred_cls = get_prediction(img_file_, threshold=0.8) # Get predictions
        img = cv2.imread(img_file_) # Read image with cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
        #plt.imshow(img)
        #plt.show()
        for box, cls in zip(boxes, pred_cls):#range(len(boxes)):
            cv2.rectangle(img, box[0], box[1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
            cv2.putText(img,cls, box[0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
        
        
        obb_file = settings.MEDIA_ROOT + '/obb_img.png' 
        cv2.imwrite(obb_file, img)
        return render(request, 'App_1_CV_Images/object_detection.html', {'original_img': img_file,
                                                                             'obb_img': '/media/obb_img.png'})

    return render(request, 'App_1_CV_Images/object_detection.html')
