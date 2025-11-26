"""
Pydantic schemas for AttCo brain tumor segmentation endpoints
"""

from pydantic import BaseModel, HttpUrl
from typing import Optional, List


class NiftiUrlRequestAttCo(BaseModel):
    flair_url: HttpUrl  # URL to FLAIR NIfTI file
    t1_url: HttpUrl  # URL to T1 NIfTI file
    t1ce_url: HttpUrl  # URL to T1CE NIfTI file
    t2_url: HttpUrl  # URL to T2 NIfTI file


class NiftiUrlRequestAttCoMultiView(BaseModel):
    flair_url: HttpUrl  # URL to FLAIR NIfTI file
    t1_url: HttpUrl  # URL to T1 NIfTI file
    t1ce_url: HttpUrl  # URL to T1CE NIfTI file
    t2_url: HttpUrl  # URL to T2 NIfTI file


class NiftiUrlRequestAttCoComposite(BaseModel):
    flair_url: HttpUrl  # URL to FLAIR NIfTI file
    t1_url: HttpUrl  # URL to T1 NIfTI file
    t1ce_url: HttpUrl  # URL to T1CE NIfTI file
    t2_url: HttpUrl  # URL to T2 NIfTI file


class NiftiToPngRequest(BaseModel):
    flair_url: HttpUrl  # URL to FLAIR NIfTI file
    t1_url: HttpUrl  # URL to T1 NIfTI file
    t1ce_url: HttpUrl  # URL to T1CE NIfTI file
    t2_url: HttpUrl  # URL to T2 NIfTI file
    slice_index: Optional[int] = None  # Optional slice index, if None uses middle slice
