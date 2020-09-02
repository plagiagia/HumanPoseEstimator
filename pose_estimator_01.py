import json

import matplotlib.pyplot as plt
import numpy as np

annotations_dir = './annotations/person_keypoints_train2017.json'
images_dir = './train2017/'

with open(annotations_dir, 'r') as f:
    json_data = json.load(f)

image_keypoints = json_data['categories'][0]['keypoints']
images_list = json_data['images']


def visualize_random_image():
    random_image_name = np.random.choice(images_list, 1)[0]['file_name']
    img = plt.imread(images_dir + random_image_name)
    print('Image shape: {}'.format(img.shape))
    plt.imshow(img)
    plt.show()


visualize_random_image()
