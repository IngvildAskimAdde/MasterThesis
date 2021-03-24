
from skimage.exposure import match_histograms
import SimpleITK as sitk
import matplotlib.pyplot as plt
import h5py

im1_Oxy = sitk.ReadImage('/Volumes/LaCie/MasterThesis_Ingvild/Oxy_cropped/Oxytarget_90_PRE/T2.nii')
im1_LARC = sitk.ReadImage('/Volumes/LaCie/MasterThesis_Ingvild/LARC_cropped/LARC-RRP-035/image.nii')

im1_Oxy_array = sitk.GetArrayFromImage(im1_Oxy)
im1_LARC_array = sitk.GetArrayFromImage(im1_LARC)

matched_LARC = match_histograms(im1_LARC_array, im1_Oxy_array, multichannel=False)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

for aa in (ax1, ax2, ax3):
    aa.set_axis_off()
"""
ax1.imshow(im1_LARC_array[20], cmap='gray')
ax1.set_title('Source')
ax2.imshow(im1_Oxy_array[20], cmap='gray')
ax2.set_title('Reference')
ax3.imshow(matched_LARC[20], cmap='gray')
ax3.set_title('Matched')
"""
ax1.hist(im1_LARC_array[20].flatten())
ax1.set_title('Source')
ax2.hist(im1_Oxy_array[20].flatten())
ax2.set_title('Reference')
ax3.hist(matched_LARC[20].flatten())
ax3.set_title('Matched')

plt.tight_layout()
plt.show()

#path = '/Volumes/LaCie/MasterThesis_Ingvild/HDF5_data/traditionalSplit_LARC.h5'
#file = h5py.File(path,'r')
#data = file['train']['352']['input'][603]
#mask = file['val/352/target_an'][603]
#patient = file['train']['352']['patient_ids'][603]
#print(data)
#print(data.shape)
#print(data[0][...,0].shape)
#print(data.max())
#print(mask.max())
#print(patient)

