# =============================================================================
# IMPORTS
# =============================================================================
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

# =============================================================================
# DIRECTORIES
# =============================================================================
# Train
annotations_dir_train = './annotations/person_keypoints_train2017.json'
images_dir_train = './train2017/'

# Validation
annotations_dir_val = './annotations/person_keypoints_val2017.json'
images_dir_val = './val2017/'


class AnnotationsDataset(Dataset):

    def __init__(self, path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = path
        self.data = self._read_json(self.path)
        self.keypoints_names = self.data['categories'][0]['keypoints']
        self.dict_images = self.data['images']
        self.image_ids = [img['id'] for img in self.data['annotations'] if
                          img['iscrowd'] == 0 and
                          img['num_keypoints'] > 0 and
                          img['bbox'][2] > 100 and
                          img['bbox'][3] > 100]

        self.box = []
        for each in self.image_ids:
            self.box.append(self._take_box(each, self.data))

        self.keypoints = []
        for each in self.image_ids:
            self.keypoints.append(self._take_keypoints(each, self.data))

        self.images = []
        for each in self.image_ids:
            self.images.append(self._load_images(each, self.path))

        self.transformed_images = []
        for each in self.images:
            self.transformed_images.append(self._transforms(each))

        self.heatmaps = []
        for each in zip(self.images, self.keypoints):
            img, kp = each[0], each[1]
            self.heatmaps.append(self._heatmaps(img, kp))

    def _read_json(self, path):
        """
        Reads a json file and returns the data

        Parameters:
            path (string) : path to json file

        Returns:
            data (dict) : Dictionary from json file
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def _load_images(self, id, path):
        for each in self.dict_images:
            if id == each['id']:
                image_name = each['file_name']
        img = plt.imread(path + image_name)
        return img

    def _img_annotations(self, id, data):
        """
        Given the name of an image return the image's annotations

        Parameters:
            id (int): image id
            data (dict): dataset from which we will read the annotations for each image

         Returns:
               annotations (dict): A dictionary with all the annotations for each image id
        """
        annotations = {}
        for each in data['annotations']:
            if id == each['id']:
                annotations = each
        return annotations

    def _take_box(self, id, data):
        """
        Given the id of an image return the image's box coordinates

        Parameters:
            id (int): image id
            data (dict): dataset from which we will read the annotations for each image
        Returns:
            box (List): A list with the coordinates of the box [X, y, width, height]
        """
        annotations = self._img_annotations(id, data)
        box = annotations.get('bbox')
        return box

    def _take_keypoints(self, id, data):
        """
        Given the id of an image return the image's keypoints

        Parameters:
            id (int): image id
            data (dict): dataset from which we will read the annotations for each image

        Returns:
              keypoints (List): A list of lists, of totally 17 keypoints each. [X_keypoints, Y_keypoints, alpha_keypoints]
        """
        annotations = self._img_annotations(id, data)
        img_keypoints_x = []
        img_keypoints_y = []
        img_keypoints_v = []
        i_old = 0
        for i in range(3, 54, 3):
            x, y, v = annotations['keypoints'][i_old:i]
            img_keypoints_x.append(x)
            img_keypoints_y.append(y)
            img_keypoints_v.append(v)
            i_old = i
        return [img_keypoints_x, img_keypoints_y, img_keypoints_v]

    def _transforms(self, img):
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((256, 192)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        img_array = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
        return transform(img_array)

    def _heatmaps(self, img, keypoints):
        keypoints = np.array(keypoints)
        keypoints_downsampled = (keypoints[0] * 256 / img.shape[0] // 4,
                                 keypoints[1] * 192 / img.shape[1] // 4)
        validity = [1 if each > 0 else 0 for each in keypoints[2]]
        heatmap = np.zeros((17, 64, 48))
        for i in range(17):
            heatmap[i, int(keypoints_downsampled[0][i]), int(keypoints_downsampled[1][i])] = 1
        heatmap = gaussian_filter(heatmap, sigma=2)
        for i in range(17):
            temp = heatmap[i, :, :]
            heatmap[i, :, :] = ((temp - temp.min()) / (temp.max() - temp.min()))
        return heatmap


test = AnnotationsDataset(annotations_dir_train)
