from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ImagePreporcessor(object):

    def __init__(self, *args, **kwargs):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        if 'image_dir' not in kwargs:
            raise TypeError('image_dir must be provide')
        if 'model_dir' not in kwargs:
            raise TypeError('image_dir must be provide')
        if 'image_height' not in kwargs:
            raise TypeError('image_height must be provide')
        if 'image_width' not in kwargs:
            raise TypeError('image_width must be provide')
        if 'batch_size' not in kwargs:
            raise TypeError('batch_size must be provide')
        if 'epochs' not in kwargs:
            raise TypeError('epochs must be provide')

        self.image_height = kwargs['image_height']
        self.image_width = kwargs['image_width']
        self.batch_size = kwargs['batch_size']
        self.epochs = kwargs['epochs']

        self.image_dir = os.path.join(BASE_DIR, kwargs['image_dir'])
        self.model_dir = os.path.join(BASE_DIR, kwargs['model_dir'])

    '''
    count all image file from IMAGE_DIR
    '''
    def __count_files(self):
        types = ('*/*.*.jpg', '*/*.*.jpeg')
        files_grabbed = []
        for files in types:
            file_dir = os.path.join(self.image_dir, files)
            files_grabbed.extend(glob.glob(file_dir))
        return files_grabbed
    
    '''
    get class names, return all class name model
    '''
    def __get_class_names(self):
        '''
        assuming folder structure is like this
        /data:
            /cat:
                - cat.0.jpg
                - cat.1.jpg
            /dog:
                - dog.0.jpg
                - dog.1.jpg
        '''
        return np.array([item.name for item in pathlib.Path(self.image_dir).glob('*')])

    '''
    create tensorflow model
    '''
    def __create_model(self):

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(16, 3, input_shape=(self.image_height, self.image_width, 3), padding='same'))
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
    save tensorflow trained model
    '''
    def __save_model(self, model):
        model.save(self.model_dir)

    '''
    train and save tensorflow model to disk
    '''
    def train_model(self):
        # convert from uint8 to float32 in range [0,1].
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                        rotation_range=45,
                                                        width_shift_range=.15,
                                                        height_shift_range=.15,
                                                        horizontal_flip=True,
                                                        zoom_range=0.5
                                                        )

        train_data_gen = image_generator.flow_from_directory(directory=str(pathlib.Path(self.image_dir)),
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        target_size=(self.image_height, self.image_width),
                                                        classes=list(self.__get_class_names()),
                                                        class_mode='sparse'
                                                    )

        # image_batch,image_label = next(train_data_gen)
        # show_batch(image_batch, image_label)

        model = self.__create_model()

        STEP_PER_EPOCH = np.ceil(len(self.__count_files()) / self.batch_size)

        model.fit(train_data_gen, steps_per_epoch=STEP_PER_EPOCH, epochs=self.epochs)
        self.__save_model(model)

    '''
    load tensorflow trained model from disk
    '''
    def load_model(self):
        print(self.__get_class_names())
        return tf.keras.models.load_model(self.model_dir)

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

def test_model(img_proc, img_path):

    class_names = ['Drum', 'Guitar']
    model = img_proc.load_model()

    image_to_search = get_image_normal(img_path, new_size=(img_proc.image_height, img_proc.image_width))
    image_to_search = np.expand_dims(image_to_search, 0)

    predictions = model.predict(image_to_search)
    print(predictions)
    print(np.argmax(predictions))

    plt.figure(figsize=(5,5))
    plt.grid(False)
    plt.imshow(get_image_normal(img_path, new_size=(100, 100)), cmap=plt.cm.binary)
    plt.title(class_names[np.argmax(predictions)])
    plt.show()

def main():
    BATCH_SIZE = 5
    IMAGE_HEIGHT = 150
    IMAGE_WIDTH = 150
    EPOCHS = 50

    img_proc = ImagePreporcessor(image_dir='data', 
                model_dir='model/model.h5', 
                image_height=IMAGE_HEIGHT, 
                image_width=IMAGE_WIDTH,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS
                )

    # img_proc.train_model()

    test_model(img_proc, 'img1.jpg')

if __name__ == '__main__':
    main()