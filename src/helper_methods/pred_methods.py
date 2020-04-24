#Importing required header files
import os
import cv2
import numpy as np
import random
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(0) #set_random_seed(0)
seed = 0
seed_everything(seed)

# helper methods
def classify(x):
    if x < 0.5:
        return 0
    elif x < 1.5:
        return 1
    elif x < 2.5:
        return 2
    elif x < 3.5:
        return 3
    return 4

def crop_image(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)

        return img

def circle_crop(img):
    img = crop_image(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = width // 2
    y = height // 2
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image(img)

    return img

#preprocess the image
def preprocess_image(base_path, save_path, image_id, HEIGHT, WIDTH, sigmaX=10):
    image = cv2.imread(base_path + image_id)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = circle_crop(image)
    image = cv2.resize(image, (HEIGHT, WIDTH))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4 , 128)
    cv2.imwrite(save_path + image_id, image)

import tensorflow as tf
#graph = tf.compat.v1.get_default_graph()
batch_size = 32
HEIGHT = 224
WIDTH = 224

# Prediction
def prediction(model, preprocessed_image_target, filename):
    #tim = image.reshape((-1, image.shape[0], image.shape[1], image.shape[2]))

    test_dest_path = preprocessed_image_target

    datagen = ImageDataGenerator(rescale=1. / 255,
                                 rotation_range=360,
                                 horizontal_flip=True,
                                 vertical_flip=True)
    test = pd.DataFrame([[filename]], columns=['id_code'])
    test_generator = datagen.flow_from_dataframe(
        dataframe=test,
        directory=test_dest_path,
        x_col="id_code",
        batch_size=1,
        class_mode=None,
        shuffle=False,
        target_size=(HEIGHT, WIDTH),
        seed=seed)

    #tim = np.expand_dims(image, axis=0)
    #print(tim.shape)
    test_generator.reset()
    tim = next(test_generator)
    print(tim.shape)
    #global model
    #with graph.as_default():
    pred = model.predict(tim)
    #pred_normalized_with_batch = pred[0][0]/batch_size
    #print(pred)
    stage = classify(pred[0][0])
    return pred, stage
    #return -1,-1