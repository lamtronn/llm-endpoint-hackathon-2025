"""
Pydantic request/response schemas
"""

from .blip2 import DicomUrlRequestBLIP2
from .sam import DicomUrlRequestSAM, NpyUrlRequest, NiftiUrlRequestSAM
from .attco import NiftiUrlRequestAttCo, NiftiUrlRequestAttCoMultiView, NiftiUrlRequestAttCoComposite, NiftiToPngRequest
from .autopet import AutoPETUrlRequest, AutoPETMultiViewRequest

__all__ = [
    'DicomUrlRequestBLIP2',
    'DicomUrlRequestSAM',
    'NpyUrlRequest',
    'NiftiUrlRequestSAM',
    'NiftiUrlRequestAttCo',
    'NiftiUrlRequestAttCoMultiView',
    'NiftiUrlRequestAttCoComposite',
    'NiftiToPngRequest',
    'AutoPETUrlRequest',
    'AutoPETMultiViewRequest'
]
