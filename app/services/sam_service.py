"""
SAM brain tumor segmentation service
"""

import torch
import pydicom
import requests
import tempfile
import os
import base64
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor
from app.core.config import settings
from app.utils import read_nifti_from_url


class SAMService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load SAM model and processor"""
        print("Loading SAM model and processor...")
        self.processor = SamProcessor.from_pretrained(settings.SAM_MODEL_NAME)
        self.model = SamModel.from_pretrained(settings.SAM_MODEL_NAME).to(self.device)
        self.model.eval()
        print(f"SAM model loaded on device: {self.device}")

    def _process_image_with_sam(self, image: Image.Image, bounding_boxes: list) -> tuple:
        """Process image with SAM model"""
        # SAM expects [[[x1, y1, x2, y2]]] format (3-level nesting)
        input_boxes_formatted = [bounding_boxes]

        # Process inputs
        inputs = self.processor(
            image,
            input_boxes=input_boxes_formatted,
            return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process masks
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )

        # Get binary mask
        mask = masks[0].squeeze().numpy()
        mask_binary = (mask > 0).astype(np.uint8) * 255

        # Convert mask to base64
        mask_image = Image.fromarray(mask_binary)
        import io
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return mask_base64, mask_binary.shape

    def process_dicom_url(self, dicom_url: str, bounding_boxes: list) -> dict:
        """Process DICOM from URL with SAM segmentation"""
        # Download DICOM file
        response = requests.get(str(dicom_url), timeout=30, stream=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
            temp_path = temp_file.name
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)

        try:
            # Read DICOM
            ds = pydicom.dcmread(temp_path)
            pixel_array = ds.pixel_array

            # Normalize to 0-255
            pixel_min = pixel_array.min()
            pixel_max = pixel_array.max()
            if pixel_max > pixel_min:
                pixel_normalized = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype('uint8')
            else:
                pixel_normalized = pixel_array.astype('uint8')

            # Convert to RGB
            image = Image.fromarray(pixel_normalized)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Process with SAM
            mask_base64, mask_shape = self._process_image_with_sam(image, bounding_boxes)

            return {
                "status": "success",
                "dicom_url": str(dicom_url),
                "bounding_boxes": bounding_boxes,
                "mask_base64": mask_base64,
                "mask_shape": str(mask_shape),
                "image_shape": str(pixel_array.shape)
            }

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def process_npy_url(self, npy_url: str, bounding_boxes: list) -> dict:
        """Process NPY from URL with SAM segmentation"""
        # Download NPY file
        response = requests.get(str(npy_url), timeout=30, stream=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as temp_file:
            temp_path = temp_file.name
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)

        try:
            # Load numpy array
            data = np.load(temp_path)

            # Normalize to 0-255
            data_min = data.min()
            data_max = data.max()
            if data_max > data_min:
                data_normalized = ((data - data_min) / (data_max - data_min) * 255).astype('uint8')
            else:
                data_normalized = data.astype('uint8')

            # Convert to RGB
            image = Image.fromarray(data_normalized)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Process with SAM
            mask_base64, mask_shape = self._process_image_with_sam(image, bounding_boxes)

            return {
                "status": "success",
                "npy_url": str(npy_url),
                "bounding_boxes": bounding_boxes,
                "mask_base64": mask_base64,
                "mask_shape": str(mask_shape),
                "image_shape": str(data.shape)
            }

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def process_nifti_url(self, nifti_url: str, bounding_boxes: list, slice_index: int = None) -> dict:
        """Process NIfTI from URL with SAM segmentation"""
        # Download and read NIfTI
        nifti_data = read_nifti_from_url(nifti_url)

        # Select slice
        if slice_index is not None:
            slice_idx = slice_index
        else:
            slice_idx = nifti_data.shape[0] // 2

        # Validate slice index
        if slice_idx < 0 or slice_idx >= nifti_data.shape[0]:
            raise ValueError(f"Slice index {slice_idx} out of range [0, {nifti_data.shape[0]-1}]")

        # Extract slice
        slice_data = nifti_data[slice_idx]

        # Normalize to 0-255
        slice_min = slice_data.min()
        slice_max = slice_data.max()
        if slice_max > slice_min:
            slice_normalized = ((slice_data - slice_min) / (slice_max - slice_min) * 255).astype('uint8')
        else:
            slice_normalized = slice_data.astype('uint8')

        # Convert to RGB
        image = Image.fromarray(slice_normalized)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process with SAM
        mask_base64, mask_shape = self._process_image_with_sam(image, bounding_boxes)

        return {
            "status": "success",
            "nifti_url": str(nifti_url),
            "slice_index": slice_idx,
            "bounding_boxes": bounding_boxes,
            "mask_base64": mask_base64,
            "mask_shape": str(mask_shape),
            "volume_shape": str(nifti_data.shape)
        }


# Create singleton instance
sam_service = SAMService()
