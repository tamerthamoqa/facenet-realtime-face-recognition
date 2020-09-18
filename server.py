#!/usr/bin/env python3

import cv2
import tensorflow as tf
import os
from imutils.video import WebcamVideoStream  # For more performant non-blocking multi-threaded OpenCV Web Camera Stream
from scipy.misc import imread
from lib.mtcnn import detect_face  # for MTCNN face detection
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from waitress import serve
from utils import (
    load_model,
    get_face,
    get_faces_live,
    forward_pass,
    save_embedding,
    load_embeddings,
    identify_face,
    allowed_file,
    remove_file_extension,
    save_image
)

app = Flask(__name__)
app.secret_key = os.urandom(24)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
uploads_path = os.path.join(APP_ROOT, 'uploads')
embeddings_path = os.path.join(APP_ROOT, 'embeddings')
allowed_set = set(['png', 'jpg', 'jpeg'])  # allowed image formats for upload


@app.route('/upload', methods=['POST', 'GET'])
def get_image():
    """Gets an image file via POST request, feeds the image to the FaceNet model then saves both the original image
     and its resulting embedding from the FaceNet model in their designated folders.

        'uploads' folder: for image files
        'embeddings' folder: for embedding numpy files.
    """

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template(
                template_name_or_list="warning.html",
                status="No 'file' field in POST request!"
            )

        file = request.files['file']
        filename = file.filename

        if filename == "":
            return render_template(
                template_name_or_list="warning.html",
                status="No selected file!"
            )

        if file and allowed_file(filename=filename, allowed_set=allowed_set):
            filename = secure_filename(filename=filename)
            # Read image file as numpy array of RGB dimension
            img = imread(name=file, mode='RGB')

            # Detect and crop a 160 x 160 image containing a human face in the image file
            img = get_face(
                img=img,
                pnet=pnet,
                rnet=rnet,
                onet=onet,
                image_size=image_size
            )

            # If a human face is detected
            if img is not None:

                embedding = forward_pass(
                    img=img,
                    session=facenet_persistent_session,
                    images_placeholder=images_placeholder,
                    embeddings=embeddings,
                    phase_train_placeholder=phase_train_placeholder,
                    image_size=image_size
                )
                # Save cropped face image to 'uploads/' folder
                save_image(img=img, filename=filename, uploads_path=uploads_path)

                # Remove file extension from image filename for numpy file storage being based on image filename
                filename = remove_file_extension(filename=filename)

                # Save embedding to 'embeddings/' folder
                save_embedding(
                    embedding=embedding,
                    filename=filename,
                    embeddings_path=embeddings_path
                )

                return render_template(
                    template_name_or_list="upload_result.html",
                    status="Image uploaded and embedded successfully!"
                )

            else:
                return render_template(
                    template_name_or_list="upload_result.html",
                    status="Image upload was unsuccessful! No human face was detected!"
                )

    else:
        return render_template(
            template_name_or_list="warning.html",
            status="POST HTTP method required!"
        )


@app.route('/predictImage', methods=['POST', 'GET'])
def predict_image():
    """Gets an image file via POST request, feeds the image to the FaceNet model, the resulting embedding is then
    sent to be compared with the embeddings database. The image file is not stored.

    An html page is then rendered showing the prediction result.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template(
                template_name_or_list="warning.html",
                status="No 'file' field in POST request!"
            )

        file = request.files['file']
        filename = file.filename

        if filename == "":
            return render_template(
                template_name_or_list="warning.html",
                status="No selected file!"
            )

        if file and allowed_file(filename=filename, allowed_set=allowed_set):
            # Read image file as numpy array of RGB dimension
            img = imread(name=file, mode='RGB')

            # Detect and crop a 160 x 160 image containing a human face in the image file
            img = get_face(
                img=img,
                pnet=pnet,
                rnet=rnet,
                onet=onet,
                image_size=image_size
            )

            # If a human face is detected
            if img is not None:

                embedding = forward_pass(
                    img=img,
                    session=facenet_persistent_session,
                    images_placeholder=images_placeholder,
                    embeddings=embeddings,
                    phase_train_placeholder=phase_train_placeholder,
                    image_size=image_size
                )

                embedding_dict = load_embeddings()
                if embedding_dict:
                    # Compare euclidean distance between this embedding and the embeddings in 'embeddings/'
                    identity = identify_face(
                        embedding=embedding,
                        embedding_dict=embedding_dict
                    )

                    return render_template(
                        template_name_or_list='predict_result.html',
                        identity=identity
                    )

                else:
                    return render_template(
                        template_name_or_list='predict_result.html',
                        identity="No embedding files detected! Please upload image files for embedding!"
                    )

            else:
                return render_template(
                    template_name_or_list='predict_result.html',
                    identity="Operation was unsuccessful! No human face was detected!"
                )
    else:
        return render_template(
            template_name_or_list="warning.html",
            status="POST HTTP method required!"
        )


@app.route("/live", methods=['GET'])
def face_detect_live():
    """Detects faces in real-time via Web Camera."""

    embedding_dict = load_embeddings()
    if embedding_dict:
        try:
            # Start non-blocking multi-threaded OpenCV video stream
            cap = WebcamVideoStream(src=0).start()

            while True:
                frame_orig = cap.read()  # Read frame

                # Resize frame to half its size for faster computation
                frame = cv2.resize(src=frame_orig, dsize=(0, 0), fx=0.5, fy=0.5)

                # Convert the image from BGR color (which OpenCV uses) to RGB color
                frame = frame[:, :, ::-1]

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if frame.size > 0:
                    faces, rects = get_faces_live(
                        img=frame,
                        pnet=pnet,
                        rnet=rnet,
                        onet=onet,
                        image_size=image_size
                    )

                    # If there are human faces detected
                    if faces:
                        for i in range(len(faces)):
                            face_img = faces[i]
                            rect = rects[i]

                            # Scale coordinates of face locations by the resize ratio
                            rect = [coordinate * 2 for coordinate in rect]

                            face_embedding = forward_pass(
                                img=face_img,
                                session=facenet_persistent_session,
                                images_placeholder=images_placeholder,
                                embeddings=embeddings,
                                phase_train_placeholder=phase_train_placeholder,
                                image_size=image_size
                            )

                            # Compare euclidean distance between this embedding and the embeddings in 'embeddings/'
                            identity = identify_face(
                                embedding=face_embedding,
                                embedding_dict=embedding_dict
                            )

                            cv2.rectangle(
                                img=frame_orig,
                                pt1=(rect[0], rect[1]),
                                pt2=(rect[2], rect[3]),
                                color=(255, 215, 0),
                                thickness=2
                            )

                            W = int(rect[2] - rect[0]) // 2

                            cv2.putText(
                                img=frame_orig,
                                text=identity,
                                org=(rect[0] + W - (W // 2), rect[1]-7),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                color=(255, 215, 0),
                                thickness=1,
                                lineType=cv2.LINE_AA
                            )

                        cv2.imshow(winname='Video', mat=frame_orig)
                    # Keep showing camera stream even if no human faces are detected
                    cv2.imshow(winname='Video', mat=frame_orig)
                else:
                    continue
            
            cap.stop()  # Stop multi-threaded Video Stream
            cv2.destroyAllWindows()

            return render_template(template_name_or_list='index.html')

        except Exception as e:
            print(e)

    else:
        return render_template(
            template_name_or_list="warning.html",
            status="No embedding files detected! Please upload image files for embedding!"
        )


@app.route("/")
def index_page():
    """Renders the 'index.html' page for manual image file uploads."""
    return render_template(template_name_or_list="index.html")


@app.route("/predict")
def predict_page():
    """Renders the 'predict.html' page for manual image file uploads for prediction."""
    return render_template(template_name_or_list="predict.html")


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

    # Initiate persistent FaceNet model in memory
    facenet_persistent_session = tf.Session(graph=facenet_model, config=config)

    # Create Multi-Task Cascading Convolutional (MTCNN) neural networks for Face Detection
    pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path=None)

    # Start flask application on waitress WSGI server
    serve(app=app, host='0.0.0.0', port=5000)
