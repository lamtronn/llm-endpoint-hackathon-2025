"""
Pydantic schemas for BLIP2 image-to-text endpoints
"""

from pydantic import BaseModel, HttpUrl


class DicomUrlRequestBLIP2(BaseModel):
    dicom_url: HttpUrl
