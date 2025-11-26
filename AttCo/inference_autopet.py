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


def find_largest_roi_slice(mask, view):
    # Compute ROI size per slice (sum of ROI voxels in each slice)
    mask = (mask > 0).astype(np.uint8)  # Binary mask
    if view ==0:
        roi_per_slice = mask.sum(axis=(1, 2))   # shape: (D,)
    elif view ==1:
        roi_per_slice = mask.sum(axis=(0, 2))   
    else:
        roi_per_slice = mask.sum(axis=(0, 1))

    # Find slice with max ROI
    slice_idx = np.argmax(roi_per_slice)

    return slice_idx 


def overlay_single_channel(img, mask_3d, slice_idx, view, save_path, channel_idx=0, alpha=0.3):
    """
    Overlay a segmentation mask on a single-channel slice (or one channel of a multi-channel volume).
    - img: numpy array shaped (C, D, H, W) or (D, H, W)
    - mask_3d: numpy array shaped (D, H, W) with integer labels
    - slice_idx: depth index to visualize
    - save_path: output path for the saved PNG
    - channel_idx: which channel to visualize when img has multiple channels
    - alpha: mask opacity (0 transparent, 1 fully opaque)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if img.ndim == 4:
        base = img[channel_idx]
    else:
        base = img

    if view == 0:
        base = base[slice_idx]
        mask_slice = mask_3d[slice_idx]
    elif view == 1:
        base = base[:, slice_idx, :]
        mask_slice = mask_3d[:, slice_idx, :]      
    else:
        base = base[:, :, slice_idx]
        mask_slice = mask_3d[:, :, slice_idx]

    vmin, vmax = base.min(), base.max()
    base_uint8 = ((base - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
    base_bgr = cv2.cvtColor(base_uint8, cv2.COLOR_GRAY2BGR)

    color = np.zeros_like(base_bgr)
    color[mask_slice == 1] = (0, 255, 0)     # green

    overlay = cv2.addWeighted(base_bgr, 1 - alpha, color, alpha, 0)
    cv2.imwrite(save_path, overlay)


def load_autopet_image(path_image, ID):
    ct = np.load(os.path.join(path_image, ID, 'CTres.npy'))
    ct = np.clip(ct, -1024, 1024) / 1024

    pet = np.load(os.path.join(path_image, ID, 'SUV.npy'))
    pet =  (pet - np.mean(pet)) / (np.std(pet) + 1e-3)

    img = np.stack([ct, pet], axis=0)
    
    return img  

################
if __name__=="__main__":
    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

    path_image = "./data"

    model = torch.load("./checkpoint/model_autopet.pt", map_location="cuda")

    # Val phrase
    model.to(device)
    model.eval() 

    ID = "PETCT_6a3477cd9a"
    ## Load data
    img_ori = load_autopet_image(path_image, ID)
    img = torch.tensor(np.expand_dims(img_ori, 0)).to(device)

    ## Prediction
    with torch.no_grad():
        output = model(img).detach().cpu().numpy()[0, 0]
        output[output>=0.5] = 1
        output[output<0.5] = 0

        # # Visualization
        for idx in range (3):
            sl_max_tumor = find_largest_roi_slice(output, idx)
            # print("Slice with largest tumor:", sl_max_tumor)
            overlay_single_channel(img_ori, output, sl_max_tumor, idx, "./overlay_slice_view_{}.png".format(idx))
