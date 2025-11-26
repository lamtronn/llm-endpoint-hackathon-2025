"""
SAM brain tumor segmentation endpoints
"""

from fastapi import APIRouter, HTTPException
from app.schemas.sam import DicomUrlRequestSAM, NpyUrlRequest, NiftiUrlRequestSAM
from app.services import sam_service
import requests

router = APIRouter()


@router.post("/process-dicom-url")
async def process_dicom_url(payload: DicomUrlRequestSAM):
    """
    Process DICOM file from URL using SAM for brain tumor segmentation

    Usage:
    curl -X POST http://localhost:8000/api/v1/sam/process-dicom-url \\
      -H "Content-Type: application/json" \\
      -d '{
        "dicom_url": "https://example.com/scan.dcm",
        "bounding_boxes": [[100, 100, 200, 200]]
      }'
    """
    try:
        result = sam_service.process_dicom_url(payload.dicom_url, payload.bounding_boxes)
        return result
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download DICOM file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing DICOM: {str(e)}")


@router.post("/process-npy-url")
async def process_npy_url(payload: NpyUrlRequest):
    """
    Process NPY file from URL using SAM for brain tumor segmentation

    Usage:
    curl -X POST http://localhost:8000/api/v1/sam/process-npy-url \\
      -H "Content-Type: application/json" \\
      -d '{
        "npy_url": "https://example.com/scan.npy",
        "bounding_boxes": [[100, 100, 200, 200]]
      }'
    """
    try:
        result = sam_service.process_npy_url(payload.npy_url, payload.bounding_boxes)
        return result
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download NPY file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing NPY: {str(e)}")


@router.post("/process-nifti-url")
async def process_nifti_url(payload: NiftiUrlRequestSAM):
    """
    Process NIfTI file from URL using SAM for brain tumor segmentation

    Usage:
    curl -X POST http://localhost:8000/api/v1/sam/process-nifti-url \\
      -H "Content-Type: application/json" \\
      -d '{
        "nifti_url": "https://example.com/scan.nii.gz",
        "bounding_boxes": [[100, 100, 200, 200]],
        "slice_index": 78
      }'
    """
    try:
        result = sam_service.process_nifti_url(
            payload.nifti_url,
            payload.bounding_boxes,
            payload.slice_index
        )
        return result
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download NIfTI file: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing NIfTI: {str(e)}")
