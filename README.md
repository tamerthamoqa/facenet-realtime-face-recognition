# facenet-realtime-face-recognition

A small-scale flask server facial recognition implementation, using a pre-trained facenet model with real-time web camera face recognition functionality, and a pre-trained Multi-Task Cascading Convolutional Neural Network (MTCNN) for face detection and cropping.

* The main inspiration is vinyakkailas's [repository](https://github.com/vinayakkailas/Face_Recognition) which uses David Sandberg's [facenet](https://github.com/davidsandberg/facenet) repository, the required dependencies from David Sandberg's 'facenet' repository were imported in the 'lib' folder and slightly cleaned.

* The pre-trained facenet and MTCNN models are provided by David Sandberg's repository, the pre-trained facenet model I used can be downloaded (version 20170512-110547) [here](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit) and the MTCNN model is located in the 'lib' directory in the 'mtcnn' folder. A full list of available facenet models in David Sandberg's repository can be seen [here](https://github.com/davidsandberg/facenet/wiki/Training-using-the-VGGFace2-dataset#difference-to-previous-models) and [here](https://github.com/davidsandberg/facenet#pre-trained-models). Though please note the different specifications in each pre-trained model.

**Note**: This is intended as only a **small-scale** facial recognition system, that uses comparison by Euclidean Distance according to an arbitrary Euclidean Distance threshold (**1.1** in this implementation) with one stored image embedding per person. The image files would be needed to be manually uploaded via the web interface or by a mobile app that uploads image files to the address of your server ('localhost:5000/upload' in this implementation) in order to create the embedding files that use the image file's name as the identity.

If you want a scalable solution for hundreds of people or more that would need a classification algorithm instead of Euclidean Distance comparison to each stored embedding file (e.g: K-Nearest Neigbours or Support Vector Machine) on the embedding data with 5-10 examples per person, please refer to the David Sandberg repository [here](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw#4-align-the-lfw-dataset) on how to align the dataset, and [here](https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images) on how to train the classifier (a support vector machine classifier in that implementation).

## Warning
This implementation does not have "liveliness detection" functionality. If you present an image of a person to the web camera it would not know the difference between a real person and a picture.

## References
* FaceNet: [paper](https://arxiv.org/abs/1503.03832) - [repository](https://github.com/davidsandberg/facenet) 


* Multi-Task Cascading Convolutional Neural Network (MTCNN) for face detection: [paper](https://arxiv.org/abs/1604.02878) - [repository](https://github.com/kpzhang93/MTCNN_face_detection_alignment)


## Requirements
* Python 3.6

* The list of required libraries are listed in the *requirements.txt* files, a virtualenv python environment for each running mode is highly recommended.

#### Running on CPU:
```pip3 install -r requirements_cpu.txt```

#### Running using a CUDA GPU:
* The pre-trained facial detection and recognition models I used from David Sandberg's [repository](https://github.com/davidsandberg/facenet) require the following to use CUDA-accelerated computing:
    * [CUDA](https://developer.nvidia.com/cuda-90-download-archive) Toolkit 9.0
    * [cuDNN](https://developer.nvidia.com/cudnn) 7.1.4 for the facial recognition model, the MTCNN face detection model was compiled with cuDNN version 7.0.5, if it is not possible to have both versions installed the version for the facial recognition model must be installed for good performance.
    * ```pip3 install -r requirements_gpu.txt```
    * __Note__: This stackoverflow [answer](https://stackoverflow.com/questions/48428415/importerror-libcublas-so-9-0-cannot-open-shared-object-file#48429585) might help you if you are running on Ubuntu 18.04
    
## Steps
1. Download the pre-trained model [here](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit).

2. Move the model file to the 'model/' folder, the path of the model should be as follows:

     ```'model/20170512-110547/20170512-110547.pb'```

3. Run the server by using the ```python server.py``` command.

__Note:__ Running the server (server.py) using the ```flask run``` method would cause issues, because the code that defines the CNN models exists in the 'if __ name __ == ' __ main __ ':' block, the 'flask run' method would make __ name __ not equal to ' __ main __ ' in this case and would not execute the code inside that block. If you wish to use this method, then you should move the code inside that block to the top of the code below the 'allowed_set' variable with deleting the 'serve(app=app, host='0.0.0.0', port=5000)' statement.

4. Navigate to the url of the server (default: localhost:5000).

5. Upload image files of the people via the web GUI interface (**.jpg image files are recommended**). An image should contain one human face, make sure to name the image file as the name of the person inside the image.

    * **Note**: When the image file is uploaded successfully, the cropped face images will appear in the 'uploads/' folder, and the embedding files will appear in the 'embeddings/' folder.

6. With an available web camera, click the 'Click here for live facial recognition with Web Camera!' button in the index web page, press the 'q' keyboard key to shut down the web camera when you are done.
