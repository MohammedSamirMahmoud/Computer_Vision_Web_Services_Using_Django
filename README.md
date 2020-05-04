# Computer_Vision_Web_Services_Using_Django_Hosted_On_AWS_Or_GCP

### Try it now & Leave your Feedback: http://35.222.210.60/App_1_CV_Images/

![Alt text](CV_Deployment_Project/media/CV_Demo.jpg?raw=true "Computer Vision Demo") 

Using Django to provide simple Computer vision tasks as a web service on different cloud providers either (AWS or GCP)

The main purpose of this repo is to provide all steps required for hosting your ML Model on cloud as a web service.

* This ML Model can be either a custom trained one or a pre-trained one and can be related to any type (Computer vision or NLP or classical ML).

* Here we will demonstrate the usage of different pre-trained models for famous Computer vision tasks (classification - Object Detection and Segmentation).

* ML Models are deployed using two ML frameworks: Tensorflow/Keras and Pytorch. 

* We can deploy our Web Service either using Django or Flask, here we will use Django.

* Hosting on Cloud will be shown for both AWS & GCP.


## Project Layout 
*  Local/Development Side
    * Setting up your ML Virtual environment
    * ML Models Deployment
    * Django Framework 
    * Optional : Push your code to git
    
    
* Server Side
    * Website Settings
    * Cloud Provider Choice
    * Remote server setup and access
    
    
## Local/Development Side   
   1    - Setting up your ML Virtual environment
   
   Assuming OS : Ubuntu 18 --> Let's setup our Python3.7 env, in your CLI:
           
           * $ python ––version # If the revision level is lower than 3.7.x, or if Python is not installed, continue to the next step.
           * $ sudo apt update
           * $ sudo apt install software-properties-common
           * $ sudo add-apt-repository ppa:deadsnakes/ppa
           * $ sudo apt update
           * $ sudo apt install python3.7
           * $ python ––version # Version should match python 3.7.x
           * $ python3 -m pip install --upgrade pip # Make sure pip is installed and up to date
         
   Now installing virtualenv package & Creating ML Custom env:
   
           * $ pip3 install virtualenv # Or sudo apt install python3.7-venv
           * $ which virtualenv
           * $ cd # Navigating to home dir
           * $ python3.7 -m venv ml_venv # Creating ml_venv
           * $ source ml_venv/bin/activate # Activating ml_venv
           
   Now let's set-up required packages for our demo:
   
           * $ git clone https://github.com/MohammedSamirMahmoud/Computer_Vision_Web_Services_Using_Django.git
           * $ cd Computer_Vision_Web_Services_Using_Django
           * $ pip3 install -r requirements.txt # Note: you may need to use --no cache dir if any of packages failed to install on instances with weak resources
           
   Having done that you are ready to run the web service on your machine
           
           * $ cd  Computer_Vision_Web_Services_Using_Django/CV_Deployment_Project
           * $ python3 manage.py runserver
   Then in your web browser go to http://localhost:8000/App_1_CV_Images
         
         
   2    - ML Models Deployment
    
        As Mentioned before we will demo 3 CV Tasks (CLassification - Segmentation - Object Detection).
        user will be required to upload an image then the output will be produced.
        for fast deployment we will use-pretrained models. however it's still applicable to use your custom trained ml model with same steps.
   Colab Notebooks for CV Tasks:
   
   -    [Classification Task Colab!](https://colab.research.google.com/drive/1PFfQB11RKzpbpKhQ57yIN8OnLLNFoIXe)
   
   -    [Segmentation Task Colab!](https://colab.research.google.com/drive/1qPDMWYzqdqEOAut7eDp1jFLREfEm1Kx9)
  
   -    [Object Detection Task Colab!](https://colab.research.google.com/drive/1qPDMWYzqdqEOAut7eDp1jFLREfEm1Kx9)
   
        CV Tasks : 
        
            A. Classification: We use pre-trained VGG model with Keras. Details can be found in colab. 
                    # The user loads and image, and gets a class representing the most dominant object class in the image, according to ImageNet dataset classes.
                    
            B. Semantic segmentaion: We use pre-trained FCN model from torchvision. Details can be found in colab.
                    # The user loads and image, and gets an image that represent the semantic pixel wise mask of all the classes in the image according to COCOO dataset classes.
  
            C. Object detection: We use pre-trained Faster R-CNN model from torchvision. . Details can be found in colab.
                    # The user loads and image, and gets an image with single/multiple objects detected with an Object Bounding Box (OBB), and its class name, with certainity more than X% (threshold).
   ![Alt text](CV_Deployment_Project/media/Object_Detection_Media.png?raw=true "Computer Vision Object Detection Demo")                 
   
   ![Alt text](CV_Deployment_Project/media/Segmentation_Media.png?raw=true "Computer Vision Segmentation Demo")
   
   
   3    - Django Framework:
        IMP Note: If you just want to run the web service, this part is not required. This part of the demo shows a brief step by step impelementation for the whole project.
        
   Note : For more code details please visit the repo codes it self.
   
   Starting a new Django Project:
        
            * $ cd
            * $ mkdir Computer_Vision_Web_Services_Using_Django
            * $ cd Computer_Vision_Web_Services_Using_Django
            * $ django-admin startproject CV_Deployment_Project # This will create the basic website files, like the entry point of the landing page, urls routing,..etc. But no applications yet.
            * $ python manage.py startapp App_1_CV_Images # Here we create our computer vision 3 tasks application ,This will create the application files, like the urls routing, backends (views),...etc
            * $ python manage.py startapp App_2_CV_Videos # Another App but for Videos.
        Now we have several things to configure (URLs routing , Front-end & templates , Backend)
            
   A - URLS Routing:
        
   -    In CV_Deployment_Project/CV_Deployment_Project/urls.py:
                
                       
                        from django.conf.urls.static import static
                        from django.contrib import admin
                        from django.urls import path, include
                        from django.conf import settings
                        
                        urlpatterns = [
                            path('admin/', admin.site.urls),
                            path('App_1_CV_Images/', include('App_1_CV_Images.urls')),
                            #path('App_2_CV_Videos/', include('App_2_CV_Videos.urls')),
                        ]
                        
                        urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
                 
                    # The last line is particularly important, since the MEDIA_URL is the path to the /media/ directory, which has all the image files (input and output), that the code will use. So it's impartant to have that URL configured so that the templates can find those images when passed from the backend. In other words, the /media/ directory is the link or shared space the links the backend and the front end.
                 
                 
   -    In CV_Deployment_Project/App_1_CV_Images/urls.py
                 
                 
                    # Add 3 sub-apps routes representing our 3 tasks:
                        
                        from django.urls import path
                        from . import views                         

                        urlpatterns = [
                            path('', views.base, name='base'),
                            path('Classification', views.classification, name='Classification'),
                            path('Semantic_Segmentation', views.semantic_segmentation, name='Semantic_Segmentation'),
                            path('Object_Detection', views.object_detection, name='Object_Detection'),
                            path('feedback_form', views.feedback_form, name="feedback_form"), # Extra Feedback Module
                            ]
                 
   ##### The configured urlpatterns will set the default landing page to the base.html, which will be rendered using the backed function views.base. This is simply rendering the base.html template.
                    
   B - Front-End:
                
   -    in CV_Deployment_Project/App_1_CV_Images/templates/App_1_CV_Images:
                    
           ###### The base.html has the main navigation bar. When the user clicks any of the tasks, the website is routed to the required backend: Each href above will route to the configured url. In cv/urls.py we already configured which backend handler will take care of those.                  
                    
           base.html file:
                    
                            <nav class="navbar navbar-expand-sm bg-dark navbar-dark">
                              <!--a class="navbar-brand" href="#">Upload your model</a>
                              <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#collapsibleNavbar">
                                <span class="navbar-toggler-icon"></span>
                              </button-->
                              <div class="collapse navbar-collapse" id="collapsibleNavbar">
                                <ul class="navbar-nav">
                                  <li class="nav-item">
                                    <a class="nav-link" href="Classification">Classification</a>
                                  </li>
                                  <li class="nav-item">
                                    <a class="nav-link" href="Semantic_Segmentation">Semantic Segmentation</a>
                                  </li>
                                  <li class="nav-item">
                                    <a class="nav-link" href="Object_Detection">Object Detection</a>
                                          </li>
                                    <li class="nav-item">
                                    <a class="nav-link" href="feedback_form">Feedback</a>
                                  </li>
                                </ul>
                              </div>
                            </nav>
                            
                 
                    
   classification.html --> This one will have an upload file form, and output processing part that renders the uploaded image + the returned prediction:        
                            
                            {% extends 'App_1_CV_Images/base.html' %}
                            
                            {% block content %}
                            <head>
                              <title>{% block title %}Computer Vision Web Service Demo - Classification{% endblock %}</title>
                              <meta charset="utf-8">
                              <meta name="viewport" content="width=device-width, initial-scale=1">
                                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
                            
                                <!-- jQuery library -->
                                <script src="https://ajax.googleapis.com/ajax/libs/jquery/4.3.1/jquery.min.js"></script>
                            
                                <!-- Latest compiled JavaScript -->
                                <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"></script>
                              <style>
                              .fakeimg {
                                height: 200px;
                                background: #aaa;
                            
                            
                              }
                              body {
                              {% load static %}
                              background-image: url({% static "App_1_CV_Images/computer_vision_background_11.jpg" %});
                              background-repeat: no-repeat;
                              background-attachment: fixed;
                              background-size: cover;
                              }
                              </style>
                            </head>
                            
                            
                            
                            <form method="post" enctype="multipart/form-data">
                                {% csrf_token %}
                                <input type="file" name="myfile">
                                <button type="submit">Upload</button>
                              </form>
                            
                            
                            
                              {% if original_img %}
                                <h3>{{prediction}}</h3>
                                <img src="{{ original_img }}" alt="Prediction" width="500" height="333">
                            
                              {% endif %}
                            
                            
                            {% endblock %}
                            
                 
   semantic_segmentation.html & object_detection.html -->  This one will have an upload file form, and output processing part that renders the uploaded image, and the segmented image that is passed from the backed:
                            
                            {% block content %}
                            <form method="post" enctype="multipart/form-data">
                                {% csrf_token %}
                                <input type="file" name="myfile">
                                <button type="submit">Upload</button>
                            </form>
                             
                             
                             
                            {% if original_img %}
                            <img src="{{ original_img }}" alt="Original" width="500" height="333">
                             
                            {% endif %}
                             
                            {% if segmented_img %}
                            <img src="{{ segmented_img }}" alt="Segmented" width="500" height="333">
                             
                            {% endif %}
                             
                             
                            {% endblock %}
            
            
   C -  Backend: 
   
   -    in CV_Deployment_Project/App_1_CV_Images/views.py:
   
   Here is where the real work is done , we will define functions for each CV task eahc one will:
                    
                    * handles the POST request of the upload button. If not file uploaded, the basic template (without predictions) is rendered.
                    * passes the uploaded image file to the prediction model.
                    * renders back the html template with the returned prediction from the model. In cases of object detection and segmentation, those are the paths of the saved images.
   Functions details will be found in the code itself.
                 
   Sample Classification Function:
                            
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
        
        
   Now that you have done all the configuration required (Frontend , backend , Urls routing)
   
   You can start running your web service LOCALLY:
   
            $ cd Computer_Vision_Web_Services_Using_Django/CV_Deployment_Project
            $ python3 manage.py migrate
            $ python3 manage.py runserver
   Then in your web browser go to http://localhost:8000/App_1_CV_Images
        
   4    - Optional : Push your code to git
   
   ###### Note that we do this step just for version control also to pull it back on your remote server.
            $ cd Computer_Vision_Web_Services_Using_Django
            $ git init
            $ git remote add <remote repo url>
            $ git add *
            $ git commit -am "committing all local side code"
            $ git push origin                    

##### Note : This work is done for educational purpose. I've used/redeployed alot of github repos/stackoverflow posts ,etc . One of the main repos that inspired this work is this one : https://ahmadelsallab.github.io/CV/  Created by Dr.AhmadElSallab - Senior Expert of AI at Valeo. 
