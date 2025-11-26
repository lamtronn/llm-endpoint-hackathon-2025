"""
Utility functions for image processing and visualization
"""

import base64
import numpy as np
import cv2


def find_largest_roi_slice(mask: np.ndarray, view: int = 0) -> int:
    """
    Find slice with largest tumor ROI for a given view

    Args:
        mask: 3D segmentation mask
        view: 0=axial (default), 1=coronal, 2=sagittal
    """
    mask_binary = (mask > 0).astype(np.uint8)

    if view == 0:  # Axial
        roi_per_slice = mask_binary.sum(axis=(1, 2))
    elif view == 1:  # Coronal
        roi_per_slice = mask_binary.sum(axis=(0, 2))
    else:  # Sagittal (view == 2)
        roi_per_slice = mask_binary.sum(axis=(0, 1))

    slice_idx = np.argmax(roi_per_slice)
    return int(slice_idx)


def create_overlay_image(img_4ch: np.ndarray, mask_3d: np.ndarray, slice_idx: int, view: int = 0, channel_idx: int = 0, alpha: float = 0.3) -> tuple:
    """
    Create overlay visualization and return as base64 and PNG bytes

    Args:
        img_4ch: 4-channel image (C, D, H, W)
        mask_3d: 3D segmentation mask (D, H, W)
        slice_idx: slice index to visualize
        view: 0=axial (default), 1=coronal, 2=sagittal
        channel_idx: which channel to use as base (0=FLAIR, 1=T1, 2=T1CE, 3=T2)
        alpha: mask opacity (default 0.3)
    """
    # Extract base image and mask slice based on view
    if view == 0:  # Axial
        base = img_4ch[channel_idx, slice_idx]
        mask_slice = mask_3d[slice_idx]
    elif view == 1:  # Coronal
        base = img_4ch[channel_idx, :, slice_idx, :]
        mask_slice = mask_3d[:, slice_idx, :]
    else:  # Sagittal (view == 2)
        base = img_4ch[channel_idx, :, :, slice_idx]
        mask_slice = mask_3d[:, :, slice_idx]

    # Normalize to 0-255
    vmin, vmax = base.min(), base.max()
    base_uint8 = ((base - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
    base_bgr = cv2.cvtColor(base_uint8, cv2.COLOR_GRAY2BGR)

    # Create color overlay
    color = np.zeros_like(base_bgr)
    color[mask_slice == 1] = (0, 255, 255)   # yellow - necrotic/non-enhancing tumor core
    color[mask_slice == 2] = (0, 255, 0)     # green - peritumoral edema
    color[mask_slice == 3] = (0, 0, 255)     # red - GD-enhancing tumor

    # Blend base image and overlay
    overlay = cv2.addWeighted(base_bgr, 1 - alpha, color, alpha, 0)

    # Convert to PNG bytes and base64
    _, buffer = cv2.imencode('.png', overlay)
    png_bytes = buffer.tobytes()
    encoded = base64.b64encode(png_bytes).decode('utf-8')

    return encoded, png_bytes


def create_multiview_composite(img_4ch: np.ndarray, mask_3d: np.ndarray, channel_idx: int = 0, alpha: float = 0.3) -> tuple:
    """
    Create a composite image showing all 3 views (axial, coronal, sagittal) side-by-side

    Args:
        img_4ch: 4-channel image (C, D, H, W)
        mask_3d: 3D segmentation mask (D, H, W)
        channel_idx: which channel to use as base (0=FLAIR, 1=T1, 2=T1CE, 3=T2)
        alpha: mask opacity (default 0.3)

    Returns:
        tuple: (base64_encoded_string, png_bytes)
    """
    # Find largest tumor slice for each view
    axial_slice = find_largest_roi_slice(mask_3d, view=0)
    coronal_slice = find_largest_roi_slice(mask_3d, view=1)
    sagittal_slice = find_largest_roi_slice(mask_3d, view=2)

    # Create overlay for each view
    views = []
    view_names = ["Axial", "Coronal", "Sagittal"]
    slices = [axial_slice, coronal_slice, sagittal_slice]

    for view_idx, (view_name, slice_idx) in enumerate(zip(view_names, slices)):
        # Extract base image and mask slice based on view
        if view_idx == 0:  # Axial
            base = img_4ch[channel_idx, slice_idx]
            mask_slice = mask_3d[slice_idx]
        elif view_idx == 1:  # Coronal
            base = img_4ch[channel_idx, :, slice_idx, :]
            mask_slice = mask_3d[:, slice_idx, :]
        else:  # Sagittal
            base = img_4ch[channel_idx, :, :, slice_idx]
            mask_slice = mask_3d[:, :, slice_idx]

        # Normalize to 0-255
        vmin, vmax = base.min(), base.max()
        base_uint8 = ((base - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
        base_bgr = cv2.cvtColor(base_uint8, cv2.COLOR_GRAY2BGR)

        # Create color overlay
        color = np.zeros_like(base_bgr)
        color[mask_slice == 1] = (0, 255, 255)   # yellow
        color[mask_slice == 2] = (0, 255, 0)     # green
        color[mask_slice == 3] = (0, 0, 255)     # red

        # Blend base image and overlay
        overlay = cv2.addWeighted(base_bgr, 1 - alpha, color, alpha, 0)

        # Add text label
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{view_name} (Slice {slice_idx})"
        cv2.putText(overlay, text, (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        views.append(overlay)

    # Resize all views to same height (use the maximum height)
    max_height = max(view.shape[0] for view in views)
    resized_views = []
    for view in views:
        if view.shape[0] != max_height:
            aspect_ratio = view.shape[1] / view.shape[0]
            new_width = int(max_height * aspect_ratio)
            resized = cv2.resize(view, (new_width, max_height), interpolation=cv2.INTER_LINEAR)
            resized_views.append(resized)
        else:
            resized_views.append(view)

    # Concatenate horizontally
    composite = np.concatenate(resized_views, axis=1)

    # Convert to PNG bytes and base64
    _, buffer = cv2.imencode('.png', composite)
    png_bytes = buffer.tobytes()
    encoded = base64.b64encode(png_bytes).decode('utf-8')

    return encoded, png_bytes, {
        "axial_slice": int(axial_slice),
        "coronal_slice": int(coronal_slice),
        "sagittal_slice": int(sagittal_slice)
    }
