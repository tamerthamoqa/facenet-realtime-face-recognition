# facenet-realtime-face-recognition

A small-scale flask server facial recognition implementation, using a pre-trained facenet model with real-time web camera face recognition functionality, and a pre-trained Multi-Task Cascading Convolutional Neural Network (MTCNN) for face detection and cropping.

* The main inspiration is from vinyakkailas's [repository](https://github.com/vinayakkailas/Face_Recognition) which is imported in the 'lib/' folder and uses David Sandberg's [facenet](https://github.com/davidsandberg/facenet) repository.

* The pre-trained facenet and MTCNN models are provided by David Sandberg's repository, the pre-trained facenet model I used can be downloaded [here](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit). A full list of available facenet models in that repository can be seen [here](https://github.com/davidsandberg/facenet/wiki/Training-using-the-VGGFace2-dataset#difference-to-previous-models) and [here](https://github.com/davidsandberg/facenet#pre-trained-models). Though please note the different specifications in each pre-trained model.

&nbsp;

**Note**: This is intended as only a **small-scale** facial recognition system, that uses comparison by Euclidean Distance according to an arbitrary threshold (1.1 in this implementation) with one stored image embedding per person. The image files would be needed to be manually uploaded via the web interface or by a mobile app that uploads image files to the address of your server ('localhost:5000/upload' in this implementation) in order to create the embedding files that use the image file's name as the identity.

If you want a scalable solution for hundreds of people or more that would need a classification algorithm instead of Euclidean Distance comparison to each stored embedding file (e.g: K-Nearest Neigbours or Support Vector Machine) on the embedding data with 5-10 examples per person, please refer to the David Sandberg repository [here](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw#4-align-the-lfw-dataset) on how to align the dataset, and [here](https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images) on how to train the classifier (a support vector machine classifier in that implementation).

## Warning
This implementation does not have "liveliness detection" functionality. If you present an image of a person to the web camera it would not know the difference between a real person and a picture.

## References
* Facenet: [paper](https://arxiv.org/abs/1503.03832) - [repository](https://github.com/davidsandberg/facenet)


* Multi-Task Cascading Convolutional Neural Network (MTCNN) for face detection: [paper](https://arxiv.org/abs/1604.02878) - [repository](https://github.com/foreverYoungGitHub/MTCNN)


## Requirements
* Python 3.6

* The pre-trained model I used requires the following:
    * Tensorflow version 1.5
    * [CUDA](https://developer.nvidia.com/cuda-90-download-archive) Toolkit 9.0
    * [cuDNN](https://developer.nvidia.com/cudnn) 7.0


* The rest of the required libraries are listed in the *requirements.txt* file, a virtualenv python environment for this implementation is recommended.    

## Steps
1. Download the pre-trained model [here](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit).

2. Move the model file to the 'model/' folder, the path of the model should be as follows:

     ```'model/20170512-110547/20170512-110547.pb'```


3. Run the ```server.py``` python file.

4. Navigate to the url of the server (default: localhost:5000).

5. Upload image files of the people via the web GUI interface (**.jpg image files are recommended**). An image should contain one human face, make sure to name the image file as the name of the person inside the image.

    * **Note**: When the image file is uploaded successfully, the cropped face images will appear in the 'uploads/' folder, and the embedding files will appear in the 'embeddings/' folder.


6. With an available web camera, click the 'Click here for live facial recognition with Web Camera!' button in the index web page, press the 'q' keyboard key to shut down the web camera when you are done.
