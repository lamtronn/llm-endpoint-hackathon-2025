"""
Unified Medical Imaging AI API

This FastAPI application provides endpoints for tumor analysis using:
- BLIP2: Image-to-text generation from medical images
- SAM: Brain tumor segmentation with bounding box prompts
- AttCo: 3D brain tumor segmentation with 4 MRI modalities
- AutoPET: PET/CT tumor segmentation with multi-view support

All endpoints support processing medical images from URLs (DICOM, NIfTI, NPY formats)
with automatic Supabase storage integration.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1 import api_router


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=__doc__
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": settings.API_TITLE,
        "version": settings.API_VERSION,
        "endpoints": {
            "BLIP2 Image-to-Text": {
                "POST /api/v1/blip2/process-dicom-url": "Generate text from DICOM images"
            },
            "SAM Brain Tumor Segmentation": {
                "POST /api/v1/sam/process-dicom-url": "Segment brain tumors in DICOM files",
                "POST /api/v1/sam/process-npy-url": "Segment brain tumors in NPY files",
                "POST /api/v1/sam/process-nifti-url": "Segment brain tumors in NIfTI files"
            },
            "AttCo 3D Brain Tumor Segmentation": {
                "POST /api/v1/attco/process-nifti-urls": "Segment brain tumors using 4 MRI modalities",
                "POST /api/v1/attco/process-nifti-urls-multiview": "Segment brain tumors with multi-view visualization",
                "POST /api/v1/attco/convert-nifti-to-png": "Convert NIfTI to PNG and upload to Supabase"
            },
            "AutoPET Tumor Segmentation": {
                "POST /api/v1/autopet/process-autopet": "Segment tumors in PET/CT scans",
                "POST /api/v1/autopet/process-autopet-multiview": "Segment tumors with multi-view visualization"
            }
        },
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
