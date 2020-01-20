from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model/model.h5')

'''
save tensorflow trained model
'''
def save_model(model):
    model.save(MODEL_DIR)

'''
load tensorflow trained model from disk
'''
def load_model():
    return tf.keras.models.load_model(MODEL_DIR)

'''
get image from disk, and return as numpy array
'''
def get_image_normal(src, new_size=(50, 50)):
    pil_image = Image.open(src)
    
    # resize image to new_size
    pil_final_image = pil_image.resize(new_size, Image.ANTIALIAS)

    
    img_array = np.array(pil_final_image, np.uint8)
    return img_array

'''
count all image file from IMAGE_DIR
'''
def count_files():
    types = ('*/*.*.jpg', '*/*.*.jpeg')
    files_grabbed = []
    for files in types:
        file_dir = os.path.join(IMAGE_DIR, files)
        files_grabbed.extend(glob.glob(file_dir))
    return files_grabbed

CLASS_NAME = np.array([item.name for item in pathlib.Path(IMAGE_DIR).glob('*')])

IMAGES_COUNT = len(count_files())
BATCH_SIZE = 5
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150
STEP_PER_EPOCH = np.ceil(IMAGES_COUNT / BATCH_SIZE)
EPOCHS = 50

'''
show batch from train gen data
'''
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(4):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAME[label_batch[n] == 1][0].title())
        plt.axis('off')
    plt.tight_layout()
    plt.show()

'''
create tensorflow model
'''
def create_model(input_shape):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(16, 3, input_shape=input_shape, padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(32, 3, padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Conv2D(64, 3, padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

'''
train and save tensorflow model to disk
'''
def train_model():
    # convert from uint8 to float32 in range [0,1].
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                    rotation_range=45,
                                                    width_shift_range=.15,
                                                    height_shift_range=.15,
                                                    horizontal_flip=True,
                                                    zoom_range=0.5
                                                    )

    train_data_gen = image_generator.flow_from_directory(directory=str(pathlib.Path(IMAGE_DIR)),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                    classes=list(CLASS_NAME),
                                                    class_mode='sparse'
                                                )

    # image_batch,image_label = next(train_data_gen)
    # show_batch(image_batch, image_label)

    model = create_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    model.fit(train_data_gen, steps_per_epoch=STEP_PER_EPOCH, epochs=EPOCHS)
    save_model(model)

def test_model(model, img_path):

    class_names = ['Drum', 'Guitar']

    image_to_search = get_image_normal(img_path, new_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    image_to_search = np.expand_dims(image_to_search, 0)
    #predictions = model.predict(test_images)
    predictions = model.predict(image_to_search)
    print(predictions)
    print(np.argmax(predictions))

    plt.figure(figsize=(5,5))
    plt.grid(False)
    plt.imshow(get_image_normal(img_path, new_size=(100, 100)), cmap=plt.cm.binary)
    plt.title(class_names[np.argmax(predictions)])
    plt.show()

def main():
    
    # print(model.summary())

    model = load_model()
    test_model(model, 'img1.jpg')

    # train_model()

if __name__ == '__main__':
    main()