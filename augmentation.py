import h5py
#from deoxys.data.preprocessor import ImageNormalizerPreprocessor
import matplotlib.pyplot as plt
from medvis import apply_cmap_with_blend


def get_images_and_targets(filepath, indice):
    """
    Returns normalized images and targets from the given indices from the hdf5 file.
    """
    with h5py.File(filepath, 'r') as f:
        images = f['val/352']['input'][indice]
        targets = f['val/352']['target_an'][indice]
        #images, targets = ImageNormalizerPreprocessor().transform(images, targets)
        return images, targets

def plot_single_image(ax, img, contour):
    """
    Plots a single image and contour.
    """
    ax.imshow(img, 'gray') #, vmin=0, vmax=1
    ax.axis('off')
    ax.contour(contour[..., 0], 1, levels=[0.5], colors='yellow')

def plot_4_single(images, targets):
    """
    Plots 4 single images with contours.
    """
    for i in range(4):
        ax = plt.subplot(2, 2, i+1)
        plot_single_image(ax, images[i][..., 0], targets[i])

    plt.show()

path = '/Volumes/HARDDISK/MasterThesis/HDF5_data/traditionalSplit_Oxy.h5'
indice = [6, 10, 14, 16]
images, targets = get_images_and_targets(path, indice)
plot_4_single(images, targets)
