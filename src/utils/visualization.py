from typing import List

import numpy as np
from skimage.color import label2rgb
import matplotlib.pyplot as plt


def get_colored_label_image_for_3d(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    depth = image.shape[0]
    colored_label_image = []
    for i in range(depth):
        colored_label_image.append(
            label2rgb(labels[i, :, :], image=image[i, :, :], bg_label=0)
        )
    return np.array(colored_label_image)


def plot_3d_images_as_map(images:List[np.ndarray], n_images_per_slide:int=10):
    depth = np.squeeze(images[0]).shape[0]
    figures = []
    fig = None
    for i in range(len(images)):
        idx = i % n_images_per_slide
        if idx == 0:
            fig, ax = plt.subplots(nrows = n_images_per_slide, ncols=depth, sharex=True, sharey=True)
        for j in range(depth):
            ax[idx, j].imshow(np.squeeze(images[i])[j,:,:], cmap='seismic')
            ax[idx,j].axis('off')
        if idx == n_images_per_slide-1:
            figures.append(fig)
            fig.tight_layout()
            fig.subplots_adjust(hspace=0, wspace=0)
            plt.savefig('object_slides{}.png'.format(i))
    return figures