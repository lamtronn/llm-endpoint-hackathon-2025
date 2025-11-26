"""
Pydantic schemas for AutoPET tumor segmentation endpoints
"""

from pydantic import BaseModel, HttpUrl
from typing import Optional, List


class AutoPETUrlRequest(BaseModel):
    ct_url: HttpUrl  # URL to CT scan (CTres.npy or CT.nii.gz)
    pet_url: HttpUrl  # URL to PET scan (SUV.npy or PET.nii.gz)


class AutoPETMultiViewRequest(BaseModel):
    ct_url: HttpUrl  # URL to CT scan (CTres.npy or CT.nii.gz)
    pet_url: HttpUrl  # URL to PET scan (SUV.npy or PET.nii.gz)
