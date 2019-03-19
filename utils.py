#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import glob
import os
from tensorflow.python.platform import gfile
from lib.src.facenet import get_model_filenames
from lib.src.align.detect_face import detect_face  # face detection
from lib.src.facenet import load_img
from scipy.misc import imresize, imsave
from collections import defaultdict
from flask import flash


def allowed_file(filename, allowed_set):
    """Checks if filename extension is one of the allowed filename extensions for upload.
    
    Args:
        filename: filename of the uploaded file to be checked.
        allowed_set: set containing the valid image file extensions.

    Returns:
        check: boolean value representing if the file extension is in the allowed extension list.
                True = file is allowed
                False = file not allowed.
    """
    check = '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set
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


def save_image(img, filename, uploads_path):
    """Saves an image file to the 'uploads' folder.

    Args:
        img: image file (numpy array).
        filename: filename of the image file.
        uploads_path: absolute path of the 'uploads/' folder.
    """
    try:
        imsave(os.path.join(uploads_path, filename), arr=np.squeeze(img))
        flash("Image saved!")
    except Exception as e:
        print(str(e))
        return str(e)


def load_model(model):
    """Loads the FaceNet model from its directory path.

    Checks if the model is a model directory (containing a metagraph and a checkpoint file)
    or if it is a protocol buffer file with a frozen graph.

    Note: This is a modified function from the facenet.py load_model() function in the lib directory to return
    the graph object.

    Args:
        model: model path

    Returns:
        graph: Tensorflow graph object of the model
    """

    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.import_graph_def(graph_def, name='')
            return graph
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        graph = saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
        return graph


def get_face(img, pnet, rnet, onet, image_size):
    """Crops an image containing a single human face from the input image if it exists; using a Multi-Task Cascading
    Convolutional neural network, then resizes the image to the required image size: default = (160 x 160 x 3).
    If no face is detected, it returns a null value.

    Args:
          img: (numpy array) image file
          pnet: proposal net, first stage of the MTCNN face detection
          rnet: refinement net, second stage of the MTCNN face detection
          onet: output net,  third stage of the MTCNN face detection
          image_size: (int) required square image size

    Returns:
          face_img: an image containing a face of image_size: default = (160 x 160 x 3)
                    if no human face is detected a None value is returned instead.
    """
    # Default constants from the FaceNet repository implementation of the MTCNN
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    margin = 44
    input_image_size = image_size

    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face(
        img=img, minsize=minsize, pnet=pnet, rnet=rnet,
        onet=onet, threshold=threshold, factor=factor
    )

    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            det = np.squeeze(face[0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]: bb[3], bb[0]:bb[2], :]
            face_img = imresize(arr=cropped, size=(input_image_size, input_image_size), mode='RGB')
            return face_img
    else:
        return None


def get_faces_live(img, pnet, rnet, onet, image_size):
    """Detects multiple human faces live from web camera frame.

    Args:
        img: web camera frame.
        pnet: proposal net, first stage of the MTCNN face detection.
        rnet: refinement net, second stage of the MTCNN face detection.
        onet: output net,  third stage of the MTCNN face detection.
        image_size: (int) required square image size.

     Returns:
           faces: List containing the cropped human faces.
           rects: List containing the rectangle coordinates to be drawn around each human face.
    """
    # Default constants from the FaceNet repository implementation of the MTCNN
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    margin = 44
    input_image_size = image_size

    faces = []
    rects = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face(
        img=img, minsize=minsize, pnet=pnet, rnet=rnet,
        onet=onet, threshold=threshold, factor=factor
    )
    # If human face(s) is/are detected:
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = imresize(arr=cropped, size=(input_image_size, input_image_size), mode='RGB')
                faces.append(resized)
                rects.append([bb[0], bb[1], bb[2], bb[3]])

    return faces, rects


def forward_pass(img, session, images_placeholder, phase_train_placeholder, embeddings, image_size):
    """Feeds an image to the FaceNet model and returns a 128-dimension embedding for facial recognition.

    Args:
        img: image file (numpy array).
        session: The active Tensorflow session.
        images_placeholder: placeholder of the 'input:0' tensor of the pre-trained FaceNet model graph.
        phase_train_placeholder: placeholder of the 'phase_train:0' tensor of the pre-trained FaceNet model graph.
        embeddings: placeholder of the 'embeddings:0' tensor from the pre-trained FaceNet model graph.
        image_size: (int) required square image size.

    Returns:
          embedding: (numpy array) of 128 values after the image is fed to the FaceNet model.
    """
    # If there is a human face
    if img is not None:
        # Normalize the pixel values of the image for noise reduction for better accuracy and resize to desired size
        image = load_img(
            img=img, do_random_crop=False, do_random_flip=False,
            do_prewhiten=True, image_size=image_size
        )
        # Run forward pass on FaceNet model to calculate embedding
        feed_dict = {images_placeholder: image, phase_train_placeholder: False}
        embedding = session.run(embeddings, feed_dict=feed_dict)
        return embedding

    else:
        return None


def save_embedding(embedding, filename, embeddings_path):
    """Saves the embedding numpy file to the 'embeddings' folder.

    Args:
        embedding: numpy array of 128 values after the image is fed to the FaceNet model.
        filename: filename of the image file.
        embeddings_path: absolute path of the 'embeddings/' folder.
    """
    # Save embedding of image using filename
    path = os.path.join(embeddings_path, str(filename))
    try:
        np.save(path, embedding)

    except Exception as e:
        print(str(e))


def load_embeddings():
    """Loads embedding numpy files in the 'embedding' folder into a defaultdict object and returns it.

    Returns:
        embedding_dict: defaultdict containing the embedding file name and its numpy array contents.
    """
    embedding_dict = defaultdict()

    for embedding in glob.iglob(pathname='embeddings/*.npy'):
        name = remove_file_extension(embedding)
        dict_embedding = np.load(embedding)
        embedding_dict[name] = dict_embedding

    return embedding_dict


def identify_face(embedding, embedding_dict):
    """Compares its received embedding with the embeddings stored in the 'embeddings' folder  by
    minimum euclidean distance (norm), the embedding with the least euclidean distance is the predicted class.

    If all embeddings have a distance above the distance threshold (1.1), then the class of the image does not exist
    in the embeddings folder.

    Args:
        embedding: (numpy array) containing the embedding that will be compared to the stored embeddings by euclidean
                   distance.
        embedding_dict: (defaultdict) containing the embedding file name and its numpy array contents.

    Returns:
          result: (string) describes the most likely person that the image looks like, or that the face
                  does not exist in the database if the resulting euclidean distance is above the threshold.
    """
    min_distance = 100
    try:
        for (name, dict_embedding) in embedding_dict.items():
            # Compute euclidean distance between current embedding and the embeddings from the 'embeddings' folder
            distance = np.linalg.norm(embedding - dict_embedding)

            if distance < min_distance:
                min_distance = distance
                identity = name

        if min_distance <= 1.1:
            # remove 'embeddings/' from identity
            identity = identity[11:]
            result = "It's " + str(identity) + ", the distance is " + str(min_distance)
            return result

        else:
            result = "Not in the database, the distance is " + str(min_distance)
            return result

    except Exception as e:
        print(str(e))
        return str(e)
