import json

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset


# =============================================================================
# DATASET CLASS
# =============================================================================
class AnnotationsDataset(Dataset):

    def __init__(self, annotations_path, img_dir):
        self.annotations_path = annotations_path
        self.img_dir = img_dir
        self.data = self.read_json(self.annotations_path)
        self.annotations = self.data['annotations']
        # Clean annotations
        self.clean_annotations = self.clean_data()

    @staticmethod
    def read_json(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def clean_data(self):
        clean_annotations = []
        for each_annot in self.annotations:
            # Checks if the picture shows more than one person
            if each_annot['iscrowd'] != 0:
                continue
            # Checks if the picture has annotations
            if each_annot['num_keypoints'] < 1:
                continue

            # Check if the picture has a big enough box
            if each_annot['bbox'][2] < 48 and each_annot['bbox'][3] < 64:
                continue

            clean_annotations.append(each_annot)

        return clean_annotations

    def load_image(self, annotation):
        img_name = '000000000000'
        img_name = img_name[0:len(img_name) - len(str(annotation['image_id']))] + str(annotation['image_id']) + '.jpg'
        pil_image = Image.open(self.img_dir + '/' + img_name)

        x, y, w, h = annotation['bbox']
        resized_img = pil_image.resize((192, 256), box=(x, y, x + w, y + h))
        array = np.array(resized_img)

        # Add on more dimension in case image is Black & White
        if len(array.shape) != 3:
            array = np.stack((array,) * 3, axis=-1)
        #     array = np.atleast_3d(array)

        transformations = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
        rescaled_img = array / 255.
        tensor_image = transformations(rescaled_img).float()

        return tensor_image

    @staticmethod
    def heatmaps(annotation):
        keypoints = np.array(annotation['keypoints']).reshape(17, 3)
        x, y, w, h = annotation['bbox']
        heatmap_w = 48
        heatmap_h = 64
        rescaled_keypoints = np.ceil(
            (keypoints - [x, y, 0]) * (np.array([heatmap_w, heatmap_h, 1]) / np.array([w, h, 1]))).astype(np.int)

        heatmaps = np.zeros((17, 64, 48))
        for i in range(17):
            if rescaled_keypoints[i][2] > 0:

                temp_x = rescaled_keypoints[i][1]
                temp_y = rescaled_keypoints[i][0]

                if temp_x >= 64 or temp_y >= 48 or temp_x < 0 or temp_y < 0:
                    rescaled_keypoints[i][2] = 0
                    continue
                else:
                    heatmaps[i, temp_x, temp_y] = 1.0
                    heatmaps[i, :, :] = gaussian_filter(heatmaps[i, :, :], sigma=2, mode='constant', cval=0.0)
                    heatmaps[i, :, :] = (heatmaps[i, :, :] - np.min(heatmaps[i, :, :])) / (
                            np.max(heatmaps[i, :, :]) - np.min(heatmaps[i, :, :]))

        validity = [1 if v > 0 else 0 for v in rescaled_keypoints[:, 2]]

        validity_tensor = torch.tensor(validity).float()
        heatmaps_tensor = torch.tensor(heatmaps).float()

        return heatmaps_tensor, validity_tensor

    def __len__(self):
        return len(self.clean_annotations)

    def __getitem__(self, id):
        annotation = self.annotations[id]
        image = self.load_image(annotation)
        heatmap, validity = self.heatmaps(annotation)

        return {'image': image, 'heatmap': heatmap, 'validity': validity}