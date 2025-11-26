"""
Utility functions for NIfTI file operations and preprocessing
"""

import os
import tempfile
import requests
import numpy as np
import SimpleITK as sitk


def read_nifti_from_url(url: str) -> np.ndarray:
    """Download and read NIfTI file from URL"""
    response = requests.get(str(url), timeout=30, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as temp_file:
        temp_path = temp_file.name
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)

    # Read with SimpleITK
    data = sitk.GetArrayFromImage(sitk.ReadImage(temp_path))

    # Cleanup
    os.unlink(temp_path)

    return data


def standardize_nonzeros(image: np.ndarray) -> np.ndarray:
    """Standardize non-zero voxels"""
    img_nonzeros = image[image != 0]
    if len(img_nonzeros) == 0:
        return image
    norm_img = (image - img_nonzeros.mean()) / img_nonzeros.std()
    return norm_img
