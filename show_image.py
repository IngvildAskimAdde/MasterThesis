import ImageViewer as iv
import SimpleITK as sitk
import h5py

image = sitk.ReadImage('/Volumes/LaCie/MasterThesis_Ingvild/Oxy_cropped_new/Oxytarget_103_PRE/T2.nii')
mask = sitk.ReadImage('/Volumes/LaCie/MasterThesis_Ingvild/Oxy_cropped_new/Oxytarget_103_PRE/Manual_an.nii')

#v = iv.Viewer(view_mode='2', mask_to_show=['a'])
#v.set_image(image, label='image')
#v.set_mask(mask, label='mask')
#v.show()

#print(sitk.GetArrayFromImage(mask).max())

f = h5py.File('/Volumes/LaCie/MasterThesis_Ingvild/HDF5_data/traditionalSplit_Oxy_new.h5', 'r')
print(f['train/352']['target_an'].max())
