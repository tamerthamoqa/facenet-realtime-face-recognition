#!/usr/bin/env python3

import cv2  # for web camera
import tensorflow as tf
import os
import numpy as np
from scipy.misc import imread, imsave
from lib.src.align import detect_face  # face detection
from utils import load_model, get_face, get_faces_live, embed_image, save_embedding, load_embeddings, who_is_it
from flask import Flask, request, render_template, flash
from werkzeug.utils import secure_filename
from waitress import serve


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])  # allowed image formats for upload


def allowed_file(filename):
    """Checks if filename extension is one of the allowed filename extensions for upload.
    Args:
        filename: filename of the uploaded file to be checked.

    Returns:
        check: boolean value representing if the file extension is in the allowed extension list.
                True = file is allowed
                False = file not allowed.
    """
    check = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return check


def remove_file_extension(filename):
    """Returns image filename without the file extension for file storage purposes.
    Args:
        filename: filename of the image file.

    Returns:
          filename: filename of the image file without the file extension.
    """
    filename = os.path.splitext(filename)[0]
    return filename


def save_image(img, filename):
    """Saves an image file to the 'uploads' folder.

    Args:
        img: image file (numpy array).
        filename: filename of the image file.
    """
    try:
        imsave(os.path.join(app.config['UPLOAD_FOLDER'], filename), arr=np.squeeze(img))
        flash("Image saved!")
    except Exception as e:
        print(str(e))
        return str(e)


@app.route('/upload', methods=['POST', 'GET'])
def get_image():
    """Gets an image file via POST request, feeds the image to the FaceNet model then saves both the original image
     and its resulting embedding from the FaceNet model in their designated folders.

        'uploads' folder: for image files
        'embeddings' folder: for embedding numpy files.
    """

    if request.method == 'POST':
        # Check if the POST request has the 'file' field  or not
        if 'file' not in request.files:
            flash('No file part')
            return "No file part"

        file = request.files['file']
        filename = file.filename
        # Check if user did not select any file for upload
        if filename == "":
            flash("No selected file")
            return "No selected file"

        if file and allowed_file(filename):
            filename = secure_filename(filename)
            # Read image as numpy array
            img = imread(file)
            # Detect and crop a 160 x 160 image containing the face in the image file
            img = get_face(img=img, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)

            # If human face is detected from get_face()
            if img is not None:
                # Feed image to FaceNet model and return embedding
                embedding = embed_image(img=img, session=facenet_persistent_session,
                                        images_placeholder=images_placeholder, embeddings=embeddings,
                                        phase_train_placeholder=phase_train_placeholder,
                                        image_size=image_size)
                # Save image
                save_image(img=img, filename=filename)
                # Remove file extension from image filename for numpy file storage based on image filename
                filename = remove_file_extension(filename=filename)
                # Save embedding to 'embeddings/' folder
                save_embedding(embedding=embedding, filename=filename, path=APP_ROOT)

                return "Image uploaded and embedded successfully!"

            else:
                return "Image upload unsuccessful (no human face detected)."

    else:
        return "POST HTTP method required!"


@app.route('/predictImage', methods=['POST', 'GET'])
def predict_image():
    """Gets an image file via POST request, feeds the image to the FaceNet model, the resulting embedding is then
    sent to be compared with the embeddings database.

    No file is stored.
    """
    if request.method == 'POST':
        # Check if the POST request has the 'file' field  or not
        if 'file' not in request.files:
            flash('No file part')
            return "No file part"

        file = request.files['file']
        filename = file.filename
        # Check if user did not select any file for upload
        if filename == "":
            flash("No selected file")
            return "No selected file"

        if file and allowed_file(filename):
            # Read image as numpy array
            img = imread(file)
            # Detect and crop a 160 x 160 image containing the face in the image file
            img = get_face(img=img, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)

            # If human face is detected from get_face()
            if img is not None:
                # Feed image to FaceNet model and return embedding
                embedding = embed_image(img=img, session=facenet_persistent_session,
                                        images_placeholder=images_placeholder, embeddings=embeddings,
                                        phase_train_placeholder=phase_train_placeholder,
                                        image_size=image_size)

                embedding_dict = load_embeddings()
                # Compare euclidean distance between this embedding and the embeddings stored in 'embeddings/
                result = who_is_it(embedding=embedding, embedding_dict=embedding_dict)

                return result

            else:
                return "Operation is unsuccessful (no human face detected)."
    else:
        return "POST HTTP method required!"


@app.route("/live", methods=['GET', 'POST'])
def face_detect_live():
    """Detects faces in real-time via Web Camera."""

    embedding_dict = load_embeddings()
    try:
        cap = cv2.VideoCapture(0)

        while True:
            return_code, frame = cap.read()  # RGB frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame.size > 0:
                faces, rects = get_faces_live(img=frame, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
                # If there are human faces detected
                if faces:
                    for i in range(len(faces)):
                        face_img = faces[i]
                        rect = rects[i]

                        face_embedding = embed_image(img=face_img, session=facenet_persistent_session,
                                                     images_placeholder=images_placeholder, embeddings=embeddings,
                                                     phase_train_placeholder=phase_train_placeholder,
                                                     image_size=image_size)

                        identity = who_is_it(embedding=face_embedding, embedding_dict=embedding_dict)

                        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), 2)

                        W = int(rect[2] - rect[0]) // 2
                        H = int(rect[3] - rect[1]) // 2

                        cv2.putText(frame, identity, (rect[0]+W-(W//2), rect[1]-7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 215, 0), 1, cv2.LINE_AA)

                    cv2.imshow('Video', frame)
            else:
                continue

        cap.release()
        cv2.destroyAllWindows()
        return "Web Camera turned off!"
    except Exception as e:
        print(e)


@app.route("/")
def index_page():
    """Renders the 'index.html' page for manual image file uploads."""
    return render_template("index.html")


@app.route("/predict")
def predict_page():
    """Renders the 'predict.html' page for manual image file uploads for prediction."""
    return render_template("predict.html")


if __name__ == '__main__':
    """Server and FaceNet Tensorflow configuration."""

    # Load FaceNet model and configure placeholders for forward pass into the FaceNet model to calculate embeddings
    model_path = 'model/20170512-110547/20170512-110547.pb'
    facenet_model = load_model(model_path)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    image_size = 160
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # Initiate persistent FacNet model in memory
    facenet_persistent_session = tf.Session(graph=facenet_model, config=config)

    # Create Multi-Task Cascading Convolutional neural networks for Face Detection
    pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path=None)

    # Start flask application on waitress WSGI server
    serve(app=app, host='0.0.0.0', port=5000)
