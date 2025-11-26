import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
import os


def read_data(path_to_nifti, return_numpy=True):
        """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
        # if return_numpy:
        #     return nib.load(str(path_to_nifti)).get_fdata()
        # return nib.load(str(path_to_nifti))
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_to_nifti)))

def standadize_nonzeros(image):
    img_nonzeros = image[image!=0]
    norm_img = (image - img_nonzeros.mean()) / img_nonzeros.std()
    return norm_img


def find_largest_roi_slice(mask):
    # Compute ROI size per slice (sum of ROI voxels in each slice)
    mask = (mask > 0).astype(np.uint8)  # Binary mask
    roi_per_slice = mask.sum(axis=(1, 2))   # shape: (D,)

    # Find slice with max ROI
    slice_idx = np.argmax(roi_per_slice)

    return slice_idx 

def overlay_on_image(img_4ch, mask_3d, slice_idx, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base = img_4ch[0, slice_idx]  # flair
    vmin, vmax = base.min(), base.max()
    base_uint8 = ((base - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
    base_bgr = cv2.cvtColor(base_uint8, cv2.COLOR_GRAY2BGR)

    mask_slice = mask_3d[slice_idx]
    color = np.zeros_like(base_bgr)
    color[mask_slice == 1] = (0, 255, 255)   # yellow
    color[mask_slice == 2] = (0, 255, 0)     # green
    color[mask_slice == 3] = (0, 0, 255)     # red

    overlay = cv2.addWeighted(base_bgr, 0.7, color, 0.3, 0)
    cv2.imwrite(save_path, overlay)


################
if __name__=="__main__":
    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

    path_image = "./data"

    model = torch.load("./checkpoint/model_brats.pt", map_location="cuda")

    # Val phrase
    model.to(device)
    model.eval() 

    ID = "BraTS20_Training_001"
    ## Load data
    t1 = read_data(os.path.join(path_image, ID, ID+'_t1.nii.gz')).astype(np.float32)
    t1 = standadize_nonzeros(t1)
    t1ce = read_data(os.path.join(path_image, ID, ID+'_t1ce.nii.gz')).astype(np.float32)
    t1ce = standadize_nonzeros(t1ce)
    flair = read_data(os.path.join(path_image, ID, ID+'_flair.nii.gz')).astype(np.float32)
    flair = standadize_nonzeros(flair)
    t2 = read_data(os.path.join(path_image, ID, ID+'_t2.nii.gz')).astype(np.float32)
    t2 = standadize_nonzeros(t2)
    img_ori = np.stack([flair, t1, t1ce, t2], axis=0)

    img = img_ori[:, 13:-14, 56:-56, 56:-56]   #Crop(img_ori, target_size=128, stride=128)
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img).to(device)
    # print(img.shape)
    ## Prediction
    with torch.no_grad():
        output = model(img).argmax(dim=1).detach().cpu().numpy()[0]
        out_padded = np.pad(output, ((13, 14), (56, 56), (56, 56)), mode='constant')
        # print(out_padded.shape)
        
        # Visualization
        sl_max_tumor = find_largest_roi_slice(out_padded)
        overlay_on_image(img_ori, out_padded, sl_max_tumor, "./overlay_slice.png")
