from typing import List

import cv2
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


def plot_3d_images_as_map(
    images: List[np.ndarray], save_path: str, n_images_per_slide: int = 20, max_depth:int=23
):
    depth = max_depth
    figures = []
    fig = None
    for i in range(len(images)):
        idx = i % n_images_per_slide
        if idx == 0:
            fig, ax = plt.subplots(
                nrows=n_images_per_slide, ncols=depth, sharex=True, sharey=True, figsize=[12.8,9.6]
            )
        for j in range(depth):
            if j < np.squeeze(images[i]).shape[0]:
                img = cv2.resize(np.squeeze(images[i])[j, :, :], dsize=(64, 64))
                img = img/img.max()
            else:
                img = np.zeros([64, 64])
            ax[idx, j].imshow(img, interpolation='nearest',
                cmap="magma",
            )
            ax[idx, j].axis("off")
        if idx == n_images_per_slide - 1:
            figures.append(fig)
            #fig.tight_layout(pad=0, h_pad=0.0, w_pad=0.0)
            fig.subplots_adjust(hspace=0.0, wspace=0.0)
            plt.savefig(save_path + "/summary_slide_{}.png".format(i + 1))
            plt.close()
    return figures
