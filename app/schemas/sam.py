"""
Pydantic schemas for SAM segmentation endpoints
"""

from pydantic import BaseModel, HttpUrl
from typing import List, Optional


class DicomUrlRequestSAM(BaseModel):
    dicom_url: HttpUrl
    bounding_boxes: List[List[float]]  # Format: [[x1, y1, x2, y2], ...]


class NpyUrlRequest(BaseModel):
    npy_url: HttpUrl
    bounding_boxes: List[List[float]]


class NiftiUrlRequestSAM(BaseModel):
    nifti_url: HttpUrl
    bounding_boxes: List[List[float]]
    slice_index: Optional[int] = None
