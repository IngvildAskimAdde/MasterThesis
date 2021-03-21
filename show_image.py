import ImageViewer as iv
import SimpleITK as sitk

image = sitk.ReadImage('/Volumes/LaCie/MasterThesis_Ingvild/Oxy_cropped_ZScoreNorm/Oxytarget_91_PRE/T2.nii')
mask = sitk.ReadImage('/Volumes/LaCie/MasterThesis_Ingvild/Oxy_cropped_ZScoreNorm/Oxytarget_91_PRE/Manual_an.nii')

v = iv.Viewer(view_mode='2', mask_to_show=['a'])
v.set_image(image, label='image')
v.set_mask(mask, label='mask')
v.show()

print(sitk.GetArrayFromImage(mask).max())
