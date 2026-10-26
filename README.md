# AutoSlicer: Head CT to Inertial Parameters

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23653334.svg)](https://doi.org/10.5281/zenodo.23653334)

This project reads a DICOM CT series, runs TotalSegmentator, creates head and neck masks, and calculates mass, center of mass, inertia tensors, principal moments of inertia, and principal axes. Results are reported in both the global RAS coordinate system and a local Frankfort coordinate system.

## Input data

The input is a directory containing all DICOM slices from one CT series. The files do not need to use the `.dcm` extension.

Use a complete head CT that meets the following requirements:

- The superior end starts at the top of the head and fully includes the skull and orbits.
- The inferior end extends far enough to include the complete C7 vertebra.
- The left and right auditory canals, left orbit, and C1-C7 vertebrae are present.
- A continuous thin-slice axial reconstruction is preferred.
- Localizer images and series reconstructed with different kernels should not be mixed into the selected series.

If the input directory contains multiple DICOM series, the program selects the series with the largest number of slices. Set `SERIES_UID` in `run.py` when a specific series must be used.

## Installation

Python 3.10 or 3.11 in a dedicated virtual environment is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins TotalSegmentator 2.18.0 because this project relies on label indices from the `total` and `head_glands_cavities` multilabel tasks in that version.

TotalSegmentator downloads model weights on the first run. An internet connection and sufficient disk space are therefore required.

For NVIDIA GPU processing, install the PyTorch build appropriate for the local CUDA version before installing `requirements.txt`. The program automatically falls back to CPU processing when CUDA is unavailable, although segmentation will take considerably longer.

## Configuration and execution

Open `run.py` and set the input and output directories near the top of the file:

```python
DICOM_FOLDER = Path(r"D:\CT\case001")
OUTPUT_FOLDER = Path(r"D:\results\case001")
```

Run the script without command-line arguments:

```powershell
python run.py
```

The remaining settings are also defined at the top of `run.py`:

```python
DEVICE = "auto"             # "auto", "cpu", or "gpu"
SERIES_UID = None            # None selects the series with the most slices
LOWER_HU = None              # Optional soft-tissue HU lower threshold
UPPER_HU = None              # Optional soft-tissue HU upper threshold
KEEP_INTERMEDIATE = False    # Keep converted CT and whole-body segmentation
```

When the HU thresholds are left as `None`, the program estimates them from the 5th and 95th percentiles of CT voxels between -300 and 400 HU.

## Outputs

The output directory contains the following key files:

| File | Description |
| --- | --- |
| `CT_head_mask.nii.gz` | Label mask used for head inertia calculations |
| `CT_neck_mask.nii.gz` | Label mask used for neck inertia calculations |
| `head_inertia_global.json` | Head parameters in the global RAS coordinate system |
| `neck_inertia_global.json` | Neck parameters in the global RAS coordinate system |
| `head_inertia_frankfort.json` | Head parameters in the Frankfort coordinate system |
| `neck_inertia_frankfort.json` | Neck parameters using the Frankfort orientation and a local neck origin |

When `KEEP_INTERMEDIATE` is `True`, the following files are also retained:

```text
intermediate/CT_Source_Volume.nii.gz
intermediate/CT_TotalSegmentation.nii.gz
```

The JSON output contains these primary quantities:

- `T1`: volume in cm³;
- `T2`: estimated mass in kg;
- `T3`: inertia tensor in kg·mm²;
- `T4`: inertia tensor in kg·cm²;
- `T7`: center of mass in mm;
- `T8`: principal moments of inertia in kg·mm²;
- `T9`: principal-axis unit vectors stored as columns.

## Calculation conventions and limitations

- NIfTI world coordinates are interpreted as RAS coordinates.
- The combined inertia tensor is evaluated about the total center of mass using the parallel-axis theorem.
- Each voxel is treated as a uniform cuboid, including its intrinsic inertia contribution.
- Mass is estimated by multiplying label volume by the tissue densities defined in the source code. It is not a direct patient measurement.
- The Frankfort plane is determined from the superior points of the left and right auditory canals and the inferior point of the left orbit.
- The results depend on automatic segmentation quality. Inspect both NIfTI masks and the anatomical landmarks before using the results in research.
- TotalSegmentator and this pipeline are not intended to be used independently for clinical diagnosis or treatment decisions.

## Patient data

Do not upload identifiable DICOM data to a public repository. DICOM metadata and burned-in annotations may contain protected health information. Public examples should use synthetic data or data that have been appropriately de-identified and approved for sharing.

## Citation

If you use AutoSlicer in research, cite the archived release:

> Yang, B. (2026). *AutoSlicer: Head CT to Inertial Parameters* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23653334

Machine-readable citation metadata are provided in `CITATION.cff`.

## License

This project is released under the MIT License. See `LICENSE`.
