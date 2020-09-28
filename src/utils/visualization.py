import numpy as np
from skimage.color import label2rgb


def get_colored_label_image_for_3d(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    depth = image.shape[0]
    colored_label_image = []
    for i in range(depth):
        colored_label_image.append(
            label2rgb(labels[i, :, :], image=image[i, :, :], bg_label=0)
        )
    return np.array(colored_label_image)
