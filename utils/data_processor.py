import numpy as np
import nibabel as nib
import os
from typing import Tuple, Optional, Union
import io
import pydicom
import tempfile

class DataProcessor:
    """Handles NIfTI file processing and preprocessing for lung nodule classification"""

    @staticmethod
    def validate_file(file) -> Tuple[bool, str]:
        """
        Validate uploaded file

        Args:
            file: UploadFile object (FastAPI) or UploadedFile object (Streamlit)

        Returns:
            Tuple of (is_valid, message)
        """
        # Get filename - handle both FastAPI (filename) and Streamlit (name) attributes
        filename = getattr(file, 'filename', getattr(file, 'name', None))
        if not filename:
            return False, "No file provided"

        # Check file extension
        valid_extensions = ['.nii', '.nii.gz', '.dcm', '.dicom', '.tcia']
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in valid_extensions:
            if filename.lower().endswith('.gz'):
                # Check for .nii.gz
                base_name = os.path.splitext(filename)[0]
                if not base_name.lower().endswith('.nii'):
                    return False, "Invalid file format. Please upload .nii, .nii.gz, .dcm, or .dicom files"
            else:
                return False, "Invalid file format. Please upload .nii, .nii.gz, .dcm, or .dicom files"

        # Check file size (limit to 100MB)
        file_size = getattr(file, 'size', 0)
        if file_size > 100 * 1024 * 1024:
            return False, "File too large. Maximum size is 100MB"

        return True, "File is valid"

    @staticmethod
    def dicom_to_nifti(file_input: Union[str, object]) -> Tuple[Optional[nib.Nifti1Image], Optional[str]]:
        """
        Convert DICOM file or series to NIfTI format

        Args:
            file_input: Path to DICOM file (str) or file object

        Returns:
            Tuple of (nifti_image, error_message)
        """
        try:
            if isinstance(file_input, str):
                # Single DICOM file path
                dcm = pydicom.dcmread(file_input)
                pixel_array = dcm.pixel_array.astype(np.float32)

                # Get spacing information if available
                try:
                    spacing = [float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1]), float(dcm.SliceThickness)]
                except:
                    spacing = [1.0, 1.0, 1.0]  # Default spacing

                # Create affine matrix
                affine = np.eye(4)
                affine[0, 0] = spacing[0]
                affine[1, 1] = spacing[1]
                affine[2, 2] = spacing[2]

                # Convert to NIfTI
                nifti_img = nib.Nifti1Image(pixel_array, affine)

            else:
                # File object (Streamlit UploadedFile, etc.)
                file_bytes = file_input.getvalue()
                file_like = io.BytesIO(file_bytes)

                # Read DICOM from bytes
                dcm = pydicom.dcmread(file_like)
                pixel_array = dcm.pixel_array.astype(np.float32)

                # Get spacing information if available
                try:
                    spacing = [float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1]), float(dcm.SliceThickness)]
                except:
                    spacing = [1.0, 1.0, 1.0]  # Default spacing

                # Create affine matrix
                affine = np.eye(4)
                affine[0, 0] = spacing[0]
                affine[1, 1] = spacing[1]
                affine[2, 2] = spacing[2]

                # Convert to NIfTI
                nifti_img = nib.Nifti1Image(pixel_array, affine)

            return nifti_img, None

        except Exception as e:
            error_msg = str(e)
            if "File is missing DICOM File Meta Information header" in error_msg or "DICM" in error_msg:
                return None, f"File appears to be invalid DICOM format. .tcia files from TCIA may be manifest files or archives, not individual DICOM images. Please ensure you upload actual DICOM (.dcm) or NIfTI (.nii) image files."
            return None, f"Error converting DICOM to NIfTI: {error_msg}"

    @staticmethod
    def preprocess_image(file_input: Union[str, object]) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Preprocess NIfTI image for model prediction

        Args:
            file_input: Path to the NIfTI file (str) or file object with getvalue() method

        Returns:
            Tuple of (processed_patch, error_message)
        """
        try:
            # Determine file type and load accordingly
            if isinstance(file_input, str):
                # File path - extract filename from path
                filename = os.path.basename(file_input)
            else:
                # File object - get name from attributes
                filename = getattr(file_input, 'filename', getattr(file_input, 'name', ''))
            
            is_dicom = filename.lower().endswith(('.dcm', '.dicom', '.tcia'))

            if is_dicom:
                # Convert DICOM to NIfTI first
                nii_img, error = DataProcessor.dicom_to_nifti(file_input)
                if error:
                    return None, f"DICOM processing error: {error}"
            else:
                # Handle NIfTI files directly
                if isinstance(file_input, str):
                    # File path
                    nii_img = nib.load(file_input)
                else:
                    # File object (Streamlit UploadedFile, etc.)
                    file_bytes = file_input.getvalue()
                    file_like = io.BytesIO(file_bytes)
                    nii_img = nib.load(file_like)

            img_data = nii_img.get_fdata()

            # Basic validation
            if img_data.ndim not in [2, 3]:
                return None, "Image file must contain 2D or 3D volume data"

            # Handle 2D DICOM images by adding a singleton dimension
            if img_data.ndim == 2:
                img_data = np.expand_dims(img_data, axis=0)

            # Normalize HU values (typical lung window)
            min_bound = -1000.0
            max_bound = 400.0
            img_data = np.clip(img_data, min_bound, max_bound)

            # Normalize to [0, 1]
            img_data = (img_data - min_bound) / (max_bound - min_bound)

            # Ensure we have a 3D volume
            if img_data.shape[0] == 0 or img_data.shape[1] == 0 or img_data.shape[2] == 0:
                return None, "Invalid volume dimensions"

            # For simplicity, we'll take a center crop
            # In a real application, you'd want to segment the nodule first
            target_size = 64
            processed_patch = DataProcessor._extract_center_patch(img_data, target_size)

            # Add channel dimension for model input
            processed_patch = np.expand_dims(processed_patch, axis=[0, -1])

            return processed_patch, None

        except Exception as e:
            return None, f"Error processing image file: {str(e)}"

    @staticmethod
    def _extract_center_patch(volume: np.ndarray, target_size: int) -> np.ndarray:
        """
        Extract a center patch from the 3D volume

        Args:
            volume: 3D numpy array
            target_size: Size of the cubic patch to extract

        Returns:
            Extracted patch
        """
        depth, height, width = volume.shape

        # Calculate center coordinates
        center_d = depth // 2
        center_h = height // 2
        center_w = width // 2

        # Calculate patch boundaries
        half_size = target_size // 2
        d_start = max(0, center_d - half_size)
        d_end = min(depth, center_d + half_size)
        h_start = max(0, center_h - half_size)
        h_end = min(height, center_h + half_size)
        w_start = max(0, center_w - half_size)
        w_end = min(width, center_w + half_size)

        # Extract patch
        patch = volume[d_start:d_end, h_start:h_end, w_start:w_end]

        # Pad if necessary to reach target size
        pad_d = target_size - patch.shape[0]
        pad_h = target_size - patch.shape[1]
        pad_w = target_size - patch.shape[2]

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            pad_width = (
                (max(0, pad_d // 2), max(0, (pad_d + 1) // 2)),
                (max(0, pad_h // 2), max(0, (pad_h + 1) // 2)),
                (max(0, pad_w // 2), max(0, (pad_w + 1) // 2))
            )
            patch = np.pad(patch, pad_width, mode='constant', constant_values=0)

        return patch