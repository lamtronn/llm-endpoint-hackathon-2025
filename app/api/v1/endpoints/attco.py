"""
AttCo brain tumor segmentation endpoints
"""

from fastapi import APIRouter, HTTPException
from app.schemas.attco import NiftiUrlRequestAttCo, NiftiUrlRequestAttCoMultiView, NiftiUrlRequestAttCoComposite, NiftiToPngRequest
from app.services import attco_service
import requests

router = APIRouter()


@router.post("/process-nifti-urls")
async def process_nifti_urls(payload: NiftiUrlRequestAttCo):
    """
    Process 4 NIfTI modalities from URLs using AttCo brain tumor segmentation model

    Requires 4 modalities:
    - FLAIR (Fluid Attenuated Inversion Recovery)
    - T1 (T1-weighted)
    - T1CE (T1-weighted with contrast enhancement)
    - T2 (T2-weighted)

    Returns:
    - 3D segmentation mask
    - Overlay visualization of slice with largest tumor (base64 and Supabase URL)
    - Tumor volume statistics

    The segmentation overlay image is automatically uploaded to Supabase bucket
    in the "segmentation result" folder.

    Usage:
    curl -X POST http://localhost:8000/api/v1/attco/process-nifti-urls \\
      -H "Content-Type: application/json" \\
      -d '{
        "flair_url": "https://example.com/patient_flair.nii.gz",
        "t1_url": "https://example.com/patient_t1.nii.gz",
        "t1ce_url": "https://example.com/patient_t1ce.nii.gz",
        "t2_url": "https://example.com/patient_t2.nii.gz"
      }'
    """
    try:
        result = attco_service.process_nifti_urls(
            flair_url=payload.flair_url,
            t1_url=payload.t1_url,
            t1ce_url=payload.t1ce_url,
            t2_url=payload.t2_url,
            bucket_name="hackathon-bucket",
            folder_name=None
        )
        return result
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download NIfTI file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing NIfTI: {str(e)}")


@router.post("/convert-nifti-to-png")
async def convert_nifti_to_png(payload: NiftiToPngRequest):
    """
    Convert 4 NIfTI modalities to PNG and upload to Supabase

    1. Downloads 4 NIfTI files from provided URLs
    2. Extracts a 2D slice from each volume
    3. Converts each slice to PNG format
    4. Uploads PNG files to Supabase storage bucket in a folder
    5. Returns Supabase URLs for the uploaded PNG files

    Each set of 4 PNG files is saved in its own folder for organization.

    Usage:
    curl -X POST http://localhost:8000/api/v1/attco/convert-nifti-to-png \\
      -H "Content-Type: application/json" \\
      -d '{
        "flair_url": "https://example.com/patient_flair.nii.gz",
        "t1_url": "https://example.com/patient_t1.nii.gz",
        "t1ce_url": "https://example.com/patient_t1ce.nii.gz",
        "t2_url": "https://example.com/patient_t2.nii.gz",
        "slice_index": 78
      }'
    """
    try:
        result = attco_service.convert_nifti_to_png(
            flair_url=payload.flair_url,
            t1_url=payload.t1_url,
            t1ce_url=payload.t1ce_url,
            t2_url=payload.t2_url,
            bucket_name="hackathon-bucket",
            slice_index=payload.slice_index,
            folder_name=None
        )
        return result
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download NIfTI file: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting NIfTI to PNG: {str(e)}")


@router.post("/process-nifti-urls-multiview")
async def process_nifti_urls_multiview(payload: NiftiUrlRequestAttCoMultiView):
    """
    Process 4 NIfTI modalities with multi-view visualization

    Generates segmentation overlays for multiple views:
    - 0: Axial (top-down)
    - 1: Coronal (front-back)
    - 2: Sagittal (left-right)

    Each view shows the slice with the largest tumor area for that orientation.
    Multi-class segmentation with 3 tumor types displayed in different colors.

    Usage:
    curl -X POST http://localhost:8000/api/v1/attco/process-nifti-urls-multiview \\
      -H "Content-Type: application/json" \\
      -d '{
        "flair_url": "https://example.com/patient_flair.nii.gz",
        "t1_url": "https://example.com/patient_t1.nii.gz",
        "t1ce_url": "https://example.com/patient_t1ce.nii.gz",
        "t2_url": "https://example.com/patient_t2.nii.gz"
      }'

    Configuration (hardcoded):
    - views: [0, 1, 2] (axial, coronal, sagittal)
    - channel_idx: 0 (FLAIR)
    - alpha: 0.3 (overlay opacity)
    - bucket_name: "hackathon-bucket"
    """
    try:
        result = attco_service.process_nifti_urls_multiview(
            flair_url=payload.flair_url,
            t1_url=payload.t1_url,
            t1ce_url=payload.t1ce_url,
            t2_url=payload.t2_url,
            bucket_name="hackathon-bucket",
            folder_name=None,
            views=[0, 1, 2],
            channel_idx=0,
            alpha=0.3
        )
        return result
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download NIfTI file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing NIfTI multi-view: {str(e)}")
