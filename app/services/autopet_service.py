"""
AutoPET tumor segmentation service
"""

import sys
import os
import tempfile
import base64
import torch
import numpy as np
import cv2
import requests
from datetime import datetime
from PIL import Image
from supabase import create_client, Client
from app.core.config import settings
from app.utils import read_nifti_from_url

# Add AttCo directory to path
sys.path.insert(0, './AttCo')
import models.JointFusionNet3D_v11_autopet


class AutoPETService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.supabase_client = None
        self._load_model()
        self._init_supabase()

    def _load_model(self):
        """Load AutoPET model"""
        print("Loading AutoPET tumor segmentation model...")
        model_path = "./AttCo/checkpoint/model_autopet.pt"
        self.model = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"AutoPET model loaded on device: {self.device}")

    def _init_supabase(self):
        """Initialize Supabase client"""
        if settings.SUPABASE_URL and settings.SUPABASE_KEY:
            self.supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            print("Supabase client initialized")
        else:
            print("Warning: Supabase credentials not found")

    def _download_scan(self, url: str) -> np.ndarray:
        """Download scan file from URL (supports both NPY and NIfTI formats)"""
        url_str = str(url)

        # Check if it's a NIfTI file
        if url_str.endswith('.nii') or url_str.endswith('.nii.gz'):
            print(f"Detected NIfTI format, using NIfTI loader")
            return read_nifti_from_url(url_str)

        # Otherwise, assume it's NPY format
        print(f"Detected NPY format, using NumPy loader")
        response = requests.get(url_str, timeout=30, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as temp_file:
            temp_path = temp_file.name
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)

        try:
            data = np.load(temp_path, allow_pickle=True)
            return data
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _preprocess_ct(self, ct: np.ndarray) -> np.ndarray:
        """Preprocess CT scan: clip to [-1024, 1024] and normalize"""
        ct_clipped = np.clip(ct, -1024, 1024) / 1024
        return ct_clipped

    def _preprocess_pet(self, pet: np.ndarray) -> np.ndarray:
        """Preprocess PET scan: Z-score normalization"""
        pet_normalized = (pet - np.mean(pet)) / (np.std(pet) + 1e-3)
        return pet_normalized

    def _find_largest_roi_slice(self, mask: np.ndarray, view: int) -> int:
        """Find slice with largest tumor ROI for a given view"""
        mask_binary = (mask > 0).astype(np.uint8)

        if view == 0:  # Axial
            roi_per_slice = mask_binary.sum(axis=(1, 2))
        elif view == 1:  # Coronal
            roi_per_slice = mask_binary.sum(axis=(0, 2))
        else:  # Sagittal (view == 2)
            roi_per_slice = mask_binary.sum(axis=(0, 1))

        slice_idx = np.argmax(roi_per_slice)
        return int(slice_idx)

    def _create_overlay_image(
        self,
        img: np.ndarray,
        mask_3d: np.ndarray,
        slice_idx: int,
        view: int,
        channel_idx: int = 0,
        alpha: float = 0.3
    ) -> tuple:
        """
        Create overlay visualization for a specific view and return as base64 and PNG bytes

        Args:
            img: numpy array shaped (C, D, H, W) or (D, H, W)
            mask_3d: numpy array shaped (D, H, W) with binary labels
            slice_idx: slice index to visualize
            view: 0=axial, 1=coronal, 2=sagittal
            channel_idx: which channel to visualize when img has multiple channels
            alpha: mask opacity (0 transparent, 1 fully opaque)
        """
        # Extract base image
        if img.ndim == 4:
            base = img[channel_idx]
        else:
            base = img

        # Extract slice based on view
        if view == 0:  # Axial
            base_slice = base[slice_idx]
            mask_slice = mask_3d[slice_idx]
        elif view == 1:  # Coronal
            base_slice = base[:, slice_idx, :]
            mask_slice = mask_3d[:, slice_idx, :]
        else:  # Sagittal
            base_slice = base[:, :, slice_idx]
            mask_slice = mask_3d[:, :, slice_idx]

        # Normalize to 0-255
        vmin, vmax = base_slice.min(), base_slice.max()
        base_uint8 = ((base_slice - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
        base_bgr = cv2.cvtColor(base_uint8, cv2.COLOR_GRAY2BGR)

        # Create color overlay (green for tumor)
        color = np.zeros_like(base_bgr)
        color[mask_slice == 1] = (0, 255, 0)  # Green for tumor

        # Blend base image and overlay
        overlay = cv2.addWeighted(base_bgr, 1 - alpha, color, alpha, 0)

        # Convert to PNG bytes and base64
        _, buffer = cv2.imencode('.png', overlay)
        png_bytes = buffer.tobytes()
        encoded = base64.b64encode(png_bytes).decode('utf-8')

        return encoded, png_bytes

    def process_autopet_urls(
        self,
        ct_url: str,
        pet_url: str,
        bucket_name: str,
        folder_name: str = None,
        alpha: float = 0.3
    ) -> dict:
        """Process AutoPET CT and PET scans from URLs"""
        # Download CT and PET data
        print(f"Downloading CT from: {ct_url}")
        ct = self._download_scan(ct_url)
        ct_preprocessed = self._preprocess_ct(ct).astype(np.float32)

        print(f"Downloading PET from: {pet_url}")
        pet = self._download_scan(pet_url)
        pet_preprocessed = self._preprocess_pet(pet).astype(np.float32)

        # Stack CT and PET
        img_ori = np.stack([ct_preprocessed, pet_preprocessed], axis=0)
        print(f"Stacked image shape: {img_ori.shape}")

        # Prepare for model
        img_tensor = torch.tensor(np.expand_dims(img_ori, 0)).to(self.device)

        # Run inference
        print("Running AutoPET inference...")
        with torch.no_grad():
            output = self.model(img_tensor).detach().cpu().numpy()[0, 0]
            # Apply threshold
            output_binary = (output >= 0.5).astype(np.uint8)

        print(f"Segmentation output shape: {output_binary.shape}")

        # Find slice with largest tumor (axial view)
        slice_idx = self._find_largest_roi_slice(output_binary, view=0)
        print(f"Largest tumor slice (axial): {slice_idx}")
        # Create overlay visualization  
        overlay_base64, overlay_png_bytes = self._create_overlay_image(
            img_ori, output_binary, slice_idx, view=0, channel_idx=0, alpha=alpha
        )

        # Upload overlay to Supabase
        segmentation_url = None
        if self.supabase_client:
            try:
                # Generate folder name if not provided
                if folder_name:
                    folder = folder_name
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    folder = f"autopet_{timestamp}"

                # Upload to "autopet segmentation" folder
                file_path = f"autopet segmentation/{folder}/overlay_axial_slice_{slice_idx}.png"

                self.supabase_client.storage.from_(bucket_name).upload(
                    path=file_path,
                    file=overlay_png_bytes,
                    file_options={"content-type": "image/png", "upsert": "true"}
                )

                # Get public URL
                segmentation_url = self.supabase_client.storage.from_(bucket_name).get_public_url(file_path)
                print(f"Uploaded segmentation overlay to Supabase: {segmentation_url}")
            except Exception as e:
                print(f"Warning: Failed to upload to Supabase: {str(e)}")

        # Calculate tumor statistics
        total_voxels = output_binary.size
        tumor_voxels = (output_binary == 1).sum()

        return {
            "status": "success",
            "input_info": {
                "ct_url": str(ct_url),
                "pet_url": str(pet_url),
                "image_shape": str(img_ori.shape)
            },
            "segmentation": {
                "output_shape": str(output_binary.shape),
                "largest_tumor_slice": int(slice_idx),
                "view": "axial",
                "overlay_image_base64": overlay_base64,
                "segmentation_url": segmentation_url
            },
            "tumor_statistics": {
                "total_voxels": int(total_voxels),
                "tumor_voxels": int(tumor_voxels),
                "tumor_percentage": float(tumor_voxels / total_voxels * 100)
            },
            "model_used": "AutoPET-JointFusionNet3D"
        }

    def process_autopet_multiview(
        self,
        ct_url: str,
        pet_url: str,
        bucket_name: str,
        folder_name: str = None,
        alpha: float = 0.3,
        views: list = [0, 1, 2]
    ) -> dict:
        """Process AutoPET with multiple views (axial, coronal, sagittal)"""
        # Download CT and PET data
        print(f"Downloading CT from: {ct_url}")
        ct = self._download_scan(ct_url)
        ct_preprocessed = self._preprocess_ct(ct).astype(np.float32)

        print(f"Downloading PET from: {pet_url}")
        pet = self._download_scan(pet_url)
        pet_preprocessed = self._preprocess_pet(pet).astype(np.float32)

        # Stack CT and PET
        img_ori = np.stack([ct_preprocessed, pet_preprocessed], axis=0)
        print(f"Stacked image shape: {img_ori.shape}")

        # Prepare for model
        img_tensor = torch.tensor(np.expand_dims(img_ori, 0)).to(self.device)

        # Run inference
        print("Running AutoPET inference...")
        with torch.no_grad():
            output = self.model(img_tensor).detach().cpu().numpy()[0, 0]
            # Apply threshold
            output_binary = (output >= 0.5).astype(np.uint8)

        print(f"Segmentation output shape: {output_binary.shape}")

        # Generate folder name if not provided
        if folder_name:
            folder = folder_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = f"autopet_{timestamp}"

        # Process each view
        view_names = {0: "axial", 1: "coronal", 2: "sagittal"}
        view_results = {}

        for view in views:
            view_name = view_names[view]

            # Find slice with largest tumor
            slice_idx = self._find_largest_roi_slice(output_binary, view)
            print(f"Largest tumor slice ({view_name}): {slice_idx}")

            # Create overlay visualization
            overlay_base64, overlay_png_bytes = self._create_overlay_image(
                img_ori, output_binary, slice_idx, view, channel_idx=0, alpha=alpha
            )

            # Upload overlay to Supabase
            segmentation_url = None
            if self.supabase_client:
                try:
                    file_path = f"autopet segmentation/{folder}/overlay_{view_name}_slice_{slice_idx}.png"

                    self.supabase_client.storage.from_(bucket_name).upload(
                        path=file_path,
                        file=overlay_png_bytes,
                        file_options={"content-type": "image/png", "upsert": "true"}
                    )

                    segmentation_url = self.supabase_client.storage.from_(bucket_name).get_public_url(file_path)
                    print(f"Uploaded {view_name} overlay to Supabase: {segmentation_url}")
                except Exception as e:
                    print(f"Warning: Failed to upload {view_name} to Supabase: {str(e)}")

            view_results[view_name] = {
                "slice_index": int(slice_idx),
                "segmentation_url": segmentation_url
            }

        # Calculate tumor statistics
        total_voxels = output_binary.size
        tumor_voxels = (output_binary == 1).sum()

        return {
            "status": "success",
            "segmentation": {
                "output_shape": str(output_binary.shape),
                "views": view_results
            },
            "tumor_statistics": {
                "total_voxels": int(total_voxels),
                "tumor_voxels": int(tumor_voxels),
                "tumor_percentage": float(tumor_voxels / total_voxels * 100)
            },
            "model_used": "AutoPET-JointFusionNet3D"
        }


# Create singleton instance
autopet_service = AutoPETService()
