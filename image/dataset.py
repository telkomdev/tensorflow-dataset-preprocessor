import os
import glob
import pathlib
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'data')

'''
get class names, return all class name model
'''
def get_class_names(image_dir):
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
    return np.array([item.name for item in pathlib.Path(image_dir).glob('*')])

def get_label_from_keyword(key):
    class_names = get_class_names(IMAGE_DIR)
    labels = {l:i for i, l in enumerate(class_names)}
    return labels.get(key)

'''
get image from disk, and return as numpy array
'''
def get_image(src, new_size=(28, 28)):
    pil_image = Image.open(src)
    
    # resize image to new_size
    pil_final_image = pil_image.resize(new_size, Image.ANTIALIAS)

    
    img_array = np.array(pil_final_image, np.uint8)
    return img_array

'''
image to pixel part, assuming every image file have 3 color channels
params: numpy array
returns: flat list from numpy array with size 2352 = (784 x 3)
'''
def image_to_pixel_part(img):
    return img.reshape(2352,)

'''
convert image to pandas dataframe
'''
def convert_image_to_dataframe():
    c_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    df = pd.DataFrame()

    for root, dirs, files in os.walk(IMAGE_DIR):
        for file in files:
            if file.endswith('jpg') or file.endswith('jpeg') or file.endswith('png') or file.endswith('webp'):
                file_path = os.path.join(root, file)
                label = os.path.basename(root).replace(' ', '-').lower()

                if not label in label_ids:
                    label_ids[label] = c_id
                    c_id += 1
                id_ = label_ids[label]

                img_array = get_image(file_path)
                image_pixel_part = image_to_pixel_part(img_array)

                y_labels.append(id_)
                x_train.append(','.join(str(x) for x in image_pixel_part))

    df['label'] = y_labels
    df['data'] = x_train
    return df

'''
convert pandas dataframe to csv
'''
def convert_dataframe_to_csv(df: pd.DataFrame, file_path='out.csv'):
    df.to_csv(file_path, header=False, sep=',', index=False, doublequote=False)

'''
convert csv to pandas dataframe
'''
def convert_csv_to_dataframe(file_path='out.csv'):
    df = pd.read_csv(file_path, header=None)
    label_frame = df.iloc[:, 0]
    pixel_data_frame_str = df.iloc[:, 1]
    

    new_df = pd.DataFrame(columns=['label', 'data'])
    new_df['label'] = label_frame
    pixel_data_frame = []
    for p in pixel_data_frame_str:
        pixel_splited = p.split(',')
        pixel_data_frame.append(pixel_splited)
    new_df['data'] = pixel_data_frame
    return new_df

'''
display image from numpy array to matplotlib
params: numpy array
'''
def display_image(img):
    plt.figure(figsize=(5,5))
    plt.grid(False)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title('Image')
    plt.show()
    

if __name__ == '__main__':
    # df = convert_image_to_dataframe()
    # convert_dataframe_to_csv(df)

    df = convert_csv_to_dataframe('out.csv')
    label_series = df.iloc[:, 0]
    pixel_series = df.iloc[:, 1]

    # convert data to numpy array
    pixel_list = []
    for p in pixel_series:
        pixel_list.append(np.array(p, np.uint8))

    label_array = label_series.values
    pixel_array = np.array(pixel_list)

    # and reshaping the array to 4-dims so that it can work with the Keras API
    pixel_array = pixel_array.reshape(22, 28, 28, 3)

    '''
    If the values of the input data are in too wide a range it can negatively impact how the network performs. 
    In this case, the input values are the pixels in the image, which have a value between 0 to 255.
    So in order to normalize the data we can simply divide the image values by 255. To do this 
    we first need to make the data a float type, since they are currently integers. We can do this by using the
    '''
    pixel_array = pixel_array.astype('float32')

    pixel_array = pixel_array / 255.0

    img = pixel_array[12]
    display_image(img)
    