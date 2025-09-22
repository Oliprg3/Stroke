import os
import pydicom
import SimpleITK as sitk
from typing import List, Tuple
import json
from pathlib import Path

def load_series(dicom_dir: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise ValueError(f"No DICOM series in {dicom_dir}")
    
    series_files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(series_files)
    image = reader.Execute()
    return image

def anonymize_image(img: sitk.Image) -> sitk.Image:
    return img

def resample_image(img: sitk.Image, out_spacing=(1.0, 1.0, 5.0)) -> sitk.Image:
    in_spacing = img.GetSpacing()
    in_size = img.GetSize()
    out_size = [
        int(round(in_size[i] * (in_spacing[i] / out_spacing[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    return resampler.Execute(img)

def save_nifti(img: sitk.Image, out_path: str):
    sitk.WriteImage(img, out_path)

def process_series(dicom_dir: str, out_path: str, out_spacing=(1.0,1.0,5.0)):
    img = load_series(dicom_dir)
    img = anonymize_image(img)
    img = resample_image(img, out_spacing=out_spacing)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_nifti(img, out_path)
