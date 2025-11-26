"""
AttCo brain tumor segmentation service
"""

import sys
import os
import tempfile
import torch
import numpy as np
from datetime import datetime
from PIL import Image
from supabase import create_client, Client
from app.core.config import settings
from app.utils import read_nifti_from_url, standardize_nonzeros, find_largest_roi_slice, create_overlay_image, create_multiview_composite

# Add AttCo directory to path
sys.path.insert(0, './AttCo')
import models.JointFusionNet3D_v11


class AttCoService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.supabase_client = None
        self._load_model()
        self._init_supabase()

    def _load_model(self):
        """Load AttCo model"""
        print("Loading AttCo brain tumor segmentation model...")
        self.model = torch.load(
            settings.ATTCO_MODEL_PATH,
            map_location=self.device,
            weights_only=False
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"AttCo model loaded on device: {self.device}")

    def _init_supabase(self):
        """Initialize Supabase client"""
        if settings.SUPABASE_URL and settings.SUPABASE_KEY:
            self.supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            print("Supabase client initialized")
        else:
            print("Warning: Supabase credentials not found")

    def process_nifti_urls(
        self,
        flair_url: str,
        t1_url: str,
        t1ce_url: str,
        t2_url: str,
        bucket_name: str,
        folder_name: str = None
    ) -> dict:
        """Process 4 NIfTI modalities with AttCo model"""
        # Download all 4 modalities
        print(f"Downloading FLAIR from: {flair_url}")
        flair = read_nifti_from_url(flair_url).astype(np.float32)
        flair = standardize_nonzeros(flair)

        print(f"Downloading T1 from: {t1_url}")
        t1 = read_nifti_from_url(t1_url).astype(np.float32)
        t1 = standardize_nonzeros(t1)

        print(f"Downloading T1CE from: {t1ce_url}")
        t1ce = read_nifti_from_url(t1ce_url).astype(np.float32)
        t1ce = standardize_nonzeros(t1ce)

        print(f"Downloading T2 from: {t2_url}")
        t2 = read_nifti_from_url(t2_url).astype(np.float32)
        t2 = standardize_nonzeros(t2)

        # Stack modalities
        img_ori = np.stack([flair, t1, t1ce, t2], axis=0)
        print(f"Stacked image shape: {img_ori.shape}")

        # Crop and prepare for model
        img = img_ori[:, 13:-14, 56:-56, 56:-56]
        img = np.expand_dims(img, axis=0)
        img_tensor = torch.tensor(img).to(self.device)

        # Run inference
        print("Running AttCo inference...")
        with torch.no_grad():
            output = self.model(img_tensor).argmax(dim=1).detach().cpu().numpy()[0]
            out_padded = np.pad(output, ((13, 14), (56, 56), (56, 56)), mode='constant')

        print(f"Segmentation output shape: {out_padded.shape}")

        # Find slice with largest tumor
        slice_idx = find_largest_roi_slice(out_padded)
        print(f"Largest tumor slice: {slice_idx}")

        # Create overlay visualization
        overlay_base64, overlay_png_bytes = create_overlay_image(img_ori, out_padded, slice_idx)

        # Upload overlay to Supabase
        segmentation_url = None
        if self.supabase_client:
            try:
                # Generate folder name if not provided
                if folder_name:
                    folder = folder_name
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    folder = f"segmentation_{timestamp}"

                # Upload to "segmentation result" folder
                file_path = f"segmentation result/{folder}/overlay_slice_{slice_idx}.png"

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
        total_voxels = out_padded.size
        tumor_class_1 = (out_padded == 1).sum()
        tumor_class_2 = (out_padded == 2).sum()
        tumor_class_3 = (out_padded == 3).sum()
        total_tumor = tumor_class_1 + tumor_class_2 + tumor_class_3

        return {
            "status": "success",
            "input_info": {
                "flair_url": str(flair_url),
                "t1_url": str(t1_url),
                "t1ce_url": str(t1ce_url),
                "t2_url": str(t2_url),
                "image_shape": str(img_ori.shape)
            },
            "segmentation": {
                "output_shape": str(out_padded.shape),
                "largest_tumor_slice": int(slice_idx),
                "overlay_image_base64": overlay_base64,
                "segmentation_url": segmentation_url
            },
            "tumor_statistics": {
                "total_voxels": int(total_voxels),
                "necrotic_core_voxels": int(tumor_class_1),
                "edema_voxels": int(tumor_class_2),
                "enhancing_tumor_voxels": int(tumor_class_3),
                "total_tumor_voxels": int(total_tumor),
                "tumor_percentage": float(total_tumor / total_voxels * 100)
            },
            "tumor_classes": {
                "0": "Background",
                "1": "Necrotic/Non-enhancing tumor core (yellow)",
                "2": "Peritumoral edema (green)",
                "3": "GD-enhancing tumor (red)"
            },
            "model_used": "AttCo-JointFusionNet3D_v11"
        }

    def convert_nifti_to_png(
        self,
        flair_url: str,
        t1_url: str,
        t1ce_url: str,
        t2_url: str,
        bucket_name: str,
        slice_index: int = None,
        folder_name: str = None
    ) -> dict:
        """Convert 4 NIfTI modalities to PNG and upload to Supabase"""
        if not self.supabase_client:
            raise RuntimeError("Supabase client not initialized")

        # Generate folder name if not provided
        if folder_name:
            folder = folder_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = f"patient_{timestamp}"

        modalities = {
            "flair": flair_url,
            "t1": t1_url,
            "t1ce": t1ce_url,
            "t2": t2_url
        }

        png_urls = {}
        temp_files = []

        try:
            for modality_name, modality_url in modalities.items():
                # Download NIfTI file
                print(f"Downloading {modality_name} from: {modality_url}")
                nifti_data = read_nifti_from_url(modality_url)
                print(f"{modality_name} volume shape: {nifti_data.shape}")

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

                # Normalize and convert to PNG
                slice_data = slice_data - slice_data.min()
                if slice_data.max() > 0:
                    slice_data = (slice_data / slice_data.max() * 255).astype(np.uint8)
                else:
                    slice_data = slice_data.astype(np.uint8)

                # Create PIL Image
                image = Image.fromarray(slice_data)
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Save to temporary PNG file
                temp_png = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_png_path = temp_png.name
                temp_files.append(temp_png_path)
                image.save(temp_png_path, format='PNG')
                temp_png.close()

                print(f"Saved {modality_name} slice to temporary PNG: {temp_png_path}")

                # Upload to Supabase with folder structure
                png_filename = f"{modality_name}_slice_{slice_idx}.png"
                file_path = f"{folder}/{png_filename}"

                with open(temp_png_path, 'rb') as f:
                    png_bytes = f.read()

                self.supabase_client.storage.from_(bucket_name).upload(
                    path=file_path,
                    file=png_bytes,
                    file_options={"content-type": "image/png", "upsert": "true"}
                )

                # Get public URL
                public_url = self.supabase_client.storage.from_(bucket_name).get_public_url(file_path)
                png_urls[modality_name] = public_url

                print(f"Uploaded {modality_name} to Supabase: {public_url}")

            return {
                "status": "success",
                "slice_index": slice_idx,
                "bucket_name": bucket_name,
                "folder_name": folder,
                "png_urls": {
                    "flair_png_url": png_urls["flair"],
                    "t1_png_url": png_urls["t1"],
                    "t1ce_png_url": png_urls["t1ce"],
                    "t2_png_url": png_urls["t2"]
                },
                "message": f"Successfully converted NIfTI files to PNG and uploaded to Supabase in folder '{folder}'"
            }

        finally:
            # Cleanup temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    print(f"Cleaned up temporary file: {temp_file}")

    def process_nifti_urls_multiview(
        self,
        flair_url: str,
        t1_url: str,
        t1ce_url: str,
        t2_url: str,
        bucket_name: str,
        folder_name: str = None,
        views: list = [0, 1, 2],
        channel_idx: int = 0,
        alpha: float = 0.3
    ) -> dict:
        """Process 4 NIfTI modalities with AttCo model and generate multi-view overlays"""
        # Download all 4 modalities
        print(f"Downloading FLAIR from: {flair_url}")
        flair = read_nifti_from_url(flair_url).astype(np.float32)
        flair = standardize_nonzeros(flair)

        print(f"Downloading T1 from: {t1_url}")
        t1 = read_nifti_from_url(t1_url).astype(np.float32)
        t1 = standardize_nonzeros(t1)

        print(f"Downloading T1CE from: {t1ce_url}")
        t1ce = read_nifti_from_url(t1ce_url).astype(np.float32)
        t1ce = standardize_nonzeros(t1ce)

        print(f"Downloading T2 from: {t2_url}")
        t2 = read_nifti_from_url(t2_url).astype(np.float32)
        t2 = standardize_nonzeros(t2)

        # Stack modalities
        img_ori = np.stack([flair, t1, t1ce, t2], axis=0)
        print(f"Stacked image shape: {img_ori.shape}")

        # Crop and prepare for model
        img = img_ori[:, 13:-14, 56:-56, 56:-56]
        img = np.expand_dims(img, axis=0)
        img_tensor = torch.tensor(img).to(self.device)

        # Run inference
        print("Running AttCo inference...")
        with torch.no_grad():
            output = self.model(img_tensor).argmax(dim=1).detach().cpu().numpy()[0]
            out_padded = np.pad(output, ((13, 14), (56, 56), (56, 56)), mode='constant')

        print(f"Segmentation output shape: {out_padded.shape}")

        # Generate folder name if not provided
        if folder_name:
            folder = folder_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = f"attco_multiview_{timestamp}"

        # Process each view
        view_names = {0: "axial", 1: "coronal", 2: "sagittal"}
        modality_names = {0: "FLAIR", 1: "T1", 2: "T1CE", 3: "T2"}
        view_results = {}

        for view in views:
            view_name = view_names[view]

            # Find slice with largest tumor
            slice_idx = find_largest_roi_slice(out_padded, view)
            print(f"Largest tumor slice ({view_name}): {slice_idx}")

            # Create overlay visualization
            overlay_base64, overlay_png_bytes = create_overlay_image(
                img_ori, out_padded, slice_idx, view, channel_idx, alpha
            )

            # Upload overlay to Supabase
            segmentation_url = None
            if self.supabase_client:
                try:
                    modality_name = modality_names[channel_idx]
                    file_path = f"segmentation result/{folder}/overlay_{view_name}_{modality_name}_slice_{slice_idx}.png"

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
        total_voxels = out_padded.size
        tumor_class_1 = (out_padded == 1).sum()
        tumor_class_2 = (out_padded == 2).sum()
        tumor_class_3 = (out_padded == 3).sum()
        total_tumor = tumor_class_1 + tumor_class_2 + tumor_class_3

        return {
            "status": "success",
            "segmentation": {
                "output_shape": str(out_padded.shape),
                "views": view_results
            },
            "tumor_statistics": {
                "total_voxels": int(total_voxels),
                "necrotic_core_voxels": int(tumor_class_1),
                "edema_voxels": int(tumor_class_2),
                "enhancing_tumor_voxels": int(tumor_class_3),
                "total_tumor_voxels": int(total_tumor),
                "tumor_percentage": float(total_tumor / total_voxels * 100)
            },
            "tumor_classes": {
                "0": "Background",
                "1": "Necrotic/Non-enhancing tumor core (yellow)",
                "2": "Peritumoral edema (green)",
                "3": "GD-enhancing tumor (red)"
            },
            "model_used": "AttCo-JointFusionNet3D_v11"
        }


# Create singleton instance
attco_service = AttCoService()
