# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Load the data from the SVHN dataset
train = loadmat('train_32x32.mat')
test = loadmat('test_32x32.mat')
# Change the labels for number 0
train['y'][train['y'] == 10] = 0
test['y'][test['y'] == 10] = 0


def visualise_sample_SVHN(sample):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    for img_id, ax in zip(img_samples_ids, axes.flat):
        img = train['X'][:, :, :, img_id]
        label = train['y'][img_id].squeeze()
        ax.imshow(img)
        ax.set_title("Centered number: {}".format(label))
    plt.tight_layout()
    plt.show()


# Take a random sample and visualize it
img_samples_ids = np.random.choice(train['X'].shape[3], 4)
visualise_sample_SVHN(img_samples_ids)
