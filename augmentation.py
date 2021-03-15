import h5py
from deoxys.data.preprocessor import ImageNormalizerPreprocessor, HounsfieldWindowingPreprocessor
import matplotlib.pyplot as plt
from medvis import apply_cmap_with_blend
from deoxys_image import affine_transform
from deoxys_image import point_operation
from deoxys_image import filters


def get_images_and_targets(filepath, indice):
    """
    Returns normalized images and targets from the given indices from the hdf5 file.
    """
    with h5py.File(filepath, 'r') as f:
        images = f['train/352']['input'][indice]
        targets = f['train/352']['target_an'][indice]
        images, targets = ImageNormalizerPreprocessor(0,1000).transform(images, targets)
        #images, targets = HounsfieldWindowingPreprocessor(4500,10000,0).transform(images, targets)
        return images, targets

def plot_single_image(img, contour):
    """
    Plots a single image and contour.
    """
    plt.figure()
    ax = plt.subplot(111)
    ax.imshow(img[0][..., 0], 'gray', vmin=0, vmax=1)
    ax.axis('off')
    ax.contour(contour[0][..., 0], 1, levels=[0.5], colors='yellow')
    plt.show()

def plot_4_single(images, targets):
    """
    Plots 4 single images with contours.
    """
    for i in range(4):
        ax = plt.subplot(2, 2, i+1)
        plot_single_image(ax, images[i][..., 0], targets[i])

    plt.show()

def flip(img, contour, axis):
    transformed_images = img.copy()
    transformed_targets = contour.copy()

    transformed_images[0] = affine_transform.apply_flip(img[0], axis=axis)
    transformed_targets[0] = affine_transform.apply_flip(contour[0], axis=axis)

    plot_single_image(transformed_images, transformed_targets)

def rotate(img, contour, degrees):
    transformed_images = img.copy()
    transformed_targets = contour.copy()

    transformed_images[0] = affine_transform.apply_affine_transform(img[0], theta=degrees, axis=2)
    transformed_targets[0] = affine_transform.apply_affine_transform(contour[0], theta=degrees, axis=2)

    plot_single_image(transformed_images, transformed_targets)

def zoom(img, contour, zoom):
    transformed_images = img.copy()
    transformed_targets = contour.copy()

    transformed_images[0] = affine_transform.apply_affine_transform(img[0], zoom_factor=zoom)
    transformed_targets[0] = affine_transform.apply_affine_transform(contour[0], zoom_factor=zoom)

    plot_single_image(transformed_images, transformed_targets)

def shift(img, contour, shift_range):
    transformed_images = img.copy()
    transformed_targets = contour.copy()

    transformed_images[0] = affine_transform.apply_affine_transform(img[0], shift=shift_range)
    transformed_targets[0] = affine_transform.apply_affine_transform(contour[0], shift=shift_range)

    plot_single_image(transformed_images, transformed_targets)


def brightness(img, contour, brightness_factor, channel):
    transformed_images = img.copy()
    transformed_targets = contour.copy()

    transformed_images[0] = point_operation.change_brightness(img[0], factor=brightness_factor, channel=channel)

    plot_single_image(transformed_images, transformed_targets)

def contrast(img, contour, contrast_factor, channel):
    transformed_images = img.copy()
    transformed_targets = contour.copy()

    transformed_images[0] = point_operation.change_brightness(img[0], factor=contrast_factor, channel=channel)

    plot_single_image(transformed_images, transformed_targets)

def noise(img, contour, noise_var, channel):
    transformed_images = img.copy()
    transformed_targets = contour.copy()

    transformed_images[0] = point_operation.gaussian_noise(img[0], noise_var=noise_var, channel=channel)

    plot_single_image(transformed_images, transformed_targets)

def blur(img, contour, blur_value, channel):
    transformed_images = img.copy()
    transformed_targets = contour.copy()

    transformed_images[0] = filters.gaussian_blur(img[0], sigma=blur_value, channel=channel)

    plot_single_image(transformed_images, transformed_targets)



path = '/Volumes/LaCie/MasterThesis_Ingvild/HDF5_data/traditionalSplit_LARC.h5'
#path = '/Volumes/LaCie/MasterThesis_Ingvild/HDF5_data/traditionalSplit_Oxy.h5'
indice = [603]#, 10, 14, 16]
images, targets = get_images_and_targets(path, indice)
plot_single_image(images, targets)
#flip(images, targets, axis=0)
#rotate(images, targets, degrees=90)
#blur(images, targets, blur_value=0.5, channel=None)


file = h5py.File(path,'r')
data = file['train/352/input'][603]
patient = file['train/352/patient_ids'][603]

#data = file['train/256/input'][35]
#patient = file['train/256/patient_ids'][35]
print(data)
print(data.shape)
print(data[0][...,0].shape)
print(data.max())
print(patient)
