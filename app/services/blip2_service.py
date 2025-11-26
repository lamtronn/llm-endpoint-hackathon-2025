"""
BLIP2 image-to-text service
"""

import torch
import pydicom
import requests
import tempfile
import os
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from app.core.config import settings


class BLIP2Service:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load BLIP2 model and processor"""
        print("Loading BLIP2 model and processor...")
        self.processor = Blip2Processor.from_pretrained(settings.BLIP2_MODEL_NAME)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            settings.BLIP2_MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model = self.model.to(self.device)
        print(f"BLIP2 model loaded on device: {self.device}")

    def process_dicom_url(self, dicom_url: str) -> dict:
        """Process DICOM from URL and generate description"""
        # Download DICOM file
        response = requests.get(str(dicom_url), timeout=30, stream=True)
        response.raise_for_status()

        # Save to temporary file
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

            # Process with BLIP2
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=50)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            return {
                "status": "success",
                "dicom_url": str(dicom_url),
                "generated_text": generated_text,
                "image_shape": str(pixel_array.shape)
            }

        finally:
            # Cleanup temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# Create singleton instance
blip2_service = BLIP2Service()
