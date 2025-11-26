"""
AutoPET tumor segmentation endpoints
"""

from fastapi import APIRouter, HTTPException
from app.schemas.autopet import AutoPETUrlRequest, AutoPETMultiViewRequest
from app.services import autopet_service
import requests

router = APIRouter()


@router.post("/process-autopet")
async def process_autopet(payload: AutoPETUrlRequest):
    """
    Process AutoPET CT and PET scans from URLs for tumor segmentation

    Requires 2 modalities (supports both NPY and NIfTI formats):
    - CT scan (CTres.npy or CT.nii.gz)
    - PET scan (SUV.npy or PET.nii.gz)

    Returns:
    - Binary segmentation mask (tumor/no-tumor)
    - Overlay visualization of axial slice with largest tumor (base64 and Supabase URL)
    - Tumor volume statistics

    The segmentation overlay image is automatically uploaded to Supabase bucket
    in the "autopet segmentation" folder.

    Usage:
    curl -X POST http://localhost:8000/api/v1/autopet/process-autopet \\
      -H "Content-Type: application/json" \\
      -d '{
        "ct_url": "https://example.com/patient_CTres.npy",
        "pet_url": "https://example.com/patient_SUV.npy"
      }'

    Configuration (hardcoded):
    - alpha: 0.3 (overlay opacity)
    - bucket_name: "hackathon-bucket"
    """
    try:
        result = autopet_service.process_autopet_urls(
            ct_url=payload.ct_url,
            pet_url=payload.pet_url,
            bucket_name="hackathon-bucket",
            folder_name=None,
            alpha=0.3
        )
        return result
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download NPY file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing AutoPET: {str(e)}")


@router.post("/process-autopet-multiview")
async def process_autopet_multiview(payload: AutoPETMultiViewRequest):
    """
    Process AutoPET CT and PET scans with multi-view visualization

    Supports both NPY and NIfTI formats for CT and PET scans.

    Generates segmentation overlays for multiple views:
    - 0: Axial (top-down)
    - 1: Coronal (front-back)
    - 2: Sagittal (left-right)

    Each view shows the slice with the largest tumor area for that orientation.

    Usage:
    curl -X POST http://localhost:8000/api/v1/autopet/process-autopet-multiview \\
      -H "Content-Type: application/json" \\
      -d '{
        "ct_url": "https://example.com/patient_CTres.npy",
        "pet_url": "https://example.com/patient_SUV.npy"
      }'

    Configuration (hardcoded):
    - views: [0, 1, 2] (axial, coronal, sagittal)
    - alpha: 0.3 (overlay opacity)
    - bucket_name: "hackathon-bucket"
    """
    try:
        result = autopet_service.process_autopet_multiview(
            ct_url=payload.ct_url,
            pet_url=payload.pet_url,
            bucket_name="hackathon-bucket",
            folder_name=None,
            alpha=0.3,
            views=[0, 1, 2]
        )
        return result
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download NPY file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing AutoPET multi-view: {str(e)}")
