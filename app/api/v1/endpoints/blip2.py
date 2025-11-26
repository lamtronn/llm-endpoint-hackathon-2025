"""
BLIP2 image-to-text endpoints
"""

from fastapi import APIRouter, HTTPException
from app.schemas.blip2 import DicomUrlRequestBLIP2
from app.services import blip2_service
import requests

router = APIRouter()


@router.post("/process-dicom-url")
async def process_dicom_url(payload: DicomUrlRequestBLIP2):
    """
    Process DICOM file from URL using BLIP2 for image-to-text generation

    Usage:
    curl -X POST http://localhost:8000/api/v1/blip2/process-dicom-url \\
      -H "Content-Type: application/json" \\
      -d '{"dicom_url": "https://example.com/scan.dcm"}'
    """
    try:
        result = blip2_service.process_dicom_url(payload.dicom_url)
        return result
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download DICOM file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing DICOM: {str(e)}")
