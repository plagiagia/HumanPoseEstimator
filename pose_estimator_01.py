import json

import matplotlib.patches as patches
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


def pick_random_image():
    random_image_name = np.random.choice(images_list, 1)[0]['file_name']
    img = plt.imread(images_dir + random_image_name)
    return img, random_image_name


def img_keypoints(name):
    img_keypoints = {}
    for each in json_data['annotations']:
        img_id = each['image_id']
        if img_id == int(name[:-4]):
            img_keypoints = each
    return img_keypoints


def draw_box():
    img, name = pick_random_image()
    keypoints = img_keypoints(name)
    if len(keypoints.items()) > 0:
        box = keypoints.get('bbox')
        img_keypoints_x = []
        img_keypoints_y = []
        img_keypoints_v = []
        i_old = 0
        for i in range(3, 54, 3):
            x, y, v = keypoints['keypoints'][i_old:i]
            img_keypoints_x.append(x)
            img_keypoints_y.append(y)
            img_keypoints_v.append(v)
            i_old = i
    return box, img


def plot_box(box, img):
    fig, ax = plt.subplots(ncols=2)
    x, y, width, height = box
    ax[0].imshow(img)
    rect = patches.Rectangle((x, y), width, height,
                             linewidth=1, edgecolor='r', facecolor='b', alpha=0.3, hatch='x')
    ax[0].add_patch(rect)
    ax[1].imshow(img[int(y):int(y + height), int(x):int(x + width), :])
    plt.tight_layout()
    plt.show()


box, img = draw_box()
plot_box(box, img)