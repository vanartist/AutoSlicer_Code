from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
import torch
from scipy.ndimage import label as connected_components
from totalsegmentator.python_api import totalsegmentator


LOGGER = logging.getLogger(__name__)

DENSITY_G_CM3 = {
    44: 1.90,
    45: 1.90,
    46: 1.90,
    47: 1.90,
    48: 1.90,
    49: 1.90,
    50: 1.90,
    57: 1.10,
    58: 1.10,
    79: 1.10,
    86: 1.10,
    87: 1.10,
    90: 1.04,
    91: 1.90,
    118: 1.10,
}

HEAD_LABELS = (90, 91, 118)
NECK_LABELS = (44, 45, 46, 47, 48, 49, 50, 57, 58, 79, 86, 87, 118)


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path

    @property
    def intermediate_dir(self) -> Path:
        return self.output_dir / "intermediate"

    @property
    def source_volume(self) -> Path:
        return self.intermediate_dir / "CT_Source_Volume.nii.gz"

    @property
    def total_segmentation(self) -> Path:
        return self.intermediate_dir / "CT_TotalSegmentation.nii.gz"

    @property
    def head_mask(self) -> Path:
        return self.output_dir / "CT_head_mask.nii.gz"

    @property
    def neck_mask(self) -> Path:
        return self.output_dir / "CT_neck_mask.nii.gz"

    @property
    def head_global(self) -> Path:
        return self.output_dir / "head_inertia_global.json"

    @property
    def neck_global(self) -> Path:
        return self.output_dir / "neck_inertia_global.json"

    @property
    def head_frankfort(self) -> Path:
        return self.output_dir / "head_inertia_frankfort.json"

    @property
    def neck_frankfort(self) -> Path:
        return self.output_dir / "neck_inertia_frankfort.json"


class AutoSlicer:
    """Convert one head CT DICOM series into head/neck inertial parameters."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        device: str = "auto",
        lower_hu: float | None = None,
        upper_hu: float | None = None,
        keep_intermediate: bool = False,
    ) -> None:
        self.paths = OutputPaths(Path(output_dir).expanduser().resolve())
        self.device = self._resolve_device(device)
        self.lower_hu = lower_hu
        self.upper_hu = upper_hu
        self.keep_intermediate = keep_intermediate
        self.selected_dicom_files: list[str] = []
        self.frankfort_points: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "gpu" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'auto', 'cpu', or 'gpu'")
        if device == "gpu" and not torch.cuda.is_available():
            raise RuntimeError("GPU was requested, but CUDA is not available to PyTorch.")
        return device

    def run(self, dicom_dir: str | Path, *, series_uid: str | None = None) -> dict[str, Path]:
        dicom_dir = Path(dicom_dir).expanduser().resolve()
        if not dicom_dir.is_dir():
            raise NotADirectoryError(f"DICOM directory not found: {dicom_dir}")

        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.intermediate_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Converting DICOM series to NIfTI")
        self._dicom_to_nifti(dicom_dir, series_uid)

        if self.lower_hu is None or self.upper_hu is None:
            estimated_lower, estimated_upper = self._estimate_soft_tissue_hu()
            self.lower_hu = estimated_lower if self.lower_hu is None else self.lower_hu
            self.upper_hu = estimated_upper if self.upper_hu is None else self.upper_hu
        if self.lower_hu >= self.upper_hu:
            raise ValueError("lower_hu must be smaller than upper_hu")
        LOGGER.info("Soft-tissue HU range: %.2f to %.2f", self.lower_hu, self.upper_hu)

        source_nifti = nib.load(self.paths.source_volume)
        LOGGER.info("Running TotalSegmentator task: total (%s)", self.device)
        total_seg = totalsegmentator(
            input=source_nifti,
            task="total",
            ml=True,
            device=self.device,
        )
        nib.save(total_seg, self.paths.total_segmentation)

        LOGGER.info("Running TotalSegmentator task: head_glands_cavities (%s)", self.device)
        head_landmarks = totalsegmentator(
            input=source_nifti,
            task="head_glands_cavities",
            ml=True,
            device=self.device,
        )
        self.frankfort_points = self._find_frankfort_points(head_landmarks)

        labeled = self._add_unlabeled_soft_tissue(total_seg)
        for target in (118, 50, 90, 91, 44):
            labeled = self._keep_largest_component(labeled, target)

        head, neck = self._split_head_and_neck(labeled, offset_mm=10.0)
        head, neck = self._crop_head_and_neck(head, neck)
        head = self._keep_largest_component(head, 118)
        head = self._keep_labels(head, HEAD_LABELS)
        neck = self._keep_labels(neck, NECK_LABELS)
        nib.save(head, self.paths.head_mask)
        nib.save(neck, self.paths.neck_mask)

        metadata = self._patient_metadata()
        head_global = self._calculate_inertia(head)
        neck_global = self._calculate_inertia(neck)
        self._write_result(self.paths.head_global, metadata, "Inertia Analysis Result", head_global)
        self._write_result(self.paths.neck_global, metadata, "Inertia Analysis Result", neck_global)

        axes, head_origin = self._frankfort_frame()
        neck_origin = self._neck_origin(neck, target_label=44)
        head_local = self._transform_inertia(head_global, axes, head_origin)
        neck_local = self._transform_inertia(neck_global, axes, neck_origin)
        local_key = "Inertia Analysis Result (Frankfort Space)"
        self._write_result(self.paths.head_frankfort, metadata, local_key, head_local)
        self._write_result(self.paths.neck_frankfort, metadata, local_key, neck_local)

        if not self.keep_intermediate:
            shutil.rmtree(self.paths.intermediate_dir, ignore_errors=True)

        return {
            "head_mask": self.paths.head_mask,
            "neck_mask": self.paths.neck_mask,
            "head_global": self.paths.head_global,
            "neck_global": self.paths.neck_global,
            "head_frankfort": self.paths.head_frankfort,
            "neck_frankfort": self.paths.neck_frankfort,
        }

    def _dicom_to_nifti(self, dicom_dir: Path, series_uid: str | None) -> None:
        reader = sitk.ImageSeriesReader()
        series_ids = list(reader.GetGDCMSeriesIDs(str(dicom_dir)) or [])
        if not series_ids:
            raise ValueError(f"No readable DICOM series found in {dicom_dir}")

        series_files = {
            uid: list(reader.GetGDCMSeriesFileNames(str(dicom_dir), uid))
            for uid in series_ids
        }
        if series_uid is not None:
            if series_uid not in series_files:
                available = ", ".join(series_ids)
                raise ValueError(f"Series UID {series_uid!r} was not found. Available: {available}")
            selected_uid = series_uid
        else:
            selected_uid = max(series_files, key=lambda uid: len(series_files[uid]))

        self.selected_dicom_files = series_files[selected_uid]
        LOGGER.info(
            "Selected series %s (%d slices) from %d available series",
            selected_uid,
            len(self.selected_dicom_files),
            len(series_ids),
        )
        reader.SetFileNames(self.selected_dicom_files)
        image = reader.Execute()
        sitk.WriteImage(image, str(self.paths.source_volume))

    def _estimate_soft_tissue_hu(self) -> tuple[float, float]:
        image = sitk.ReadImage(str(self.paths.source_volume))
        hu = sitk.GetArrayViewFromImage(image)
        valid = hu[(hu > -300) & (hu < 400)]
        if valid.size == 0:
            raise ValueError("No voxels in the expected -300 to 400 HU range.")
        lower, upper = np.percentile(valid, (5, 95))
        return float(lower), float(upper)

    def _add_unlabeled_soft_tissue(self, segmentation: nib.Nifti1Image) -> nib.Nifti1Image:
        source = np.asanyarray(nib.load(self.paths.source_volume).dataobj)
        labels = np.asanyarray(segmentation.dataobj).astype(np.uint16, copy=True)
        if source.shape != labels.shape:
            raise ValueError("CT and segmentation shapes differ.")
        mask = (source >= self.lower_hu) & (source <= self.upper_hu) & (labels == 0)
        labels[mask] = 118
        return nib.Nifti1Image(labels, segmentation.affine, segmentation.header)

    @staticmethod
    def _keep_largest_component(image: nib.Nifti1Image, target_label: int) -> nib.Nifti1Image:
        data = np.asanyarray(image.dataobj).copy()
        mask = data == target_label
        components, count = connected_components(mask)
        if count == 0:
            LOGGER.warning("Label %d is missing; component cleanup skipped", target_label)
            return image
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        keep = components == int(np.argmax(sizes))
        data[mask & ~keep] = 0
        return nib.Nifti1Image(data, image.affine, image.header)

    @staticmethod
    def _keep_labels(image: nib.Nifti1Image, labels: tuple[int, ...]) -> nib.Nifti1Image:
        data = np.asanyarray(image.dataobj)
        filtered = np.where(np.isin(data, labels), data, 0).astype(np.uint16)
        return nib.Nifti1Image(filtered, image.affine, image.header)

    @staticmethod
    def _extreme_point(
        data: np.ndarray,
        affine: np.ndarray,
        target_label: int,
        *,
        highest: bool,
    ) -> np.ndarray:
        voxels = np.argwhere(data == target_label)
        if voxels.size == 0:
            raise ValueError(f"Frankfort landmark label {target_label} is missing.")
        ras = nib.affines.apply_affine(affine, voxels)
        index = np.argmax(ras[:, 2]) if highest else np.argmin(ras[:, 2])
        return ras[index]

    def _find_frankfort_points(
        self, landmarks: nib.Nifti1Image
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = np.asanyarray(landmarks.dataobj)
        f1 = self._extreme_point(data, landmarks.affine, 17, highest=True)
        f2 = self._extreme_point(data, landmarks.affine, 1, highest=False)
        f3 = self._extreme_point(data, landmarks.affine, 16, highest=True)
        return f1, f2, f3

    @staticmethod
    def _component_extreme(data: np.ndarray, target_label: int, highest: bool) -> np.ndarray:
        mask = data == target_label
        components, count = connected_components(mask)
        if count == 0:
            raise ValueError(f"Required label {target_label} is missing.")
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        voxels = np.argwhere(components == int(np.argmax(sizes)))
        z = voxels[:, 2]
        return voxels[np.argmax(z) if highest else np.argmin(z)]

    def _split_head_and_neck(
        self, image: nib.Nifti1Image, *, offset_mm: float
    ) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
        if self.frankfort_points is None:
            raise RuntimeError("Frankfort landmarks have not been calculated.")

        data = np.asanyarray(image.dataobj)
        affine = np.asarray(image.affine)
        base_ijk = self._component_extreme(data, 50, highest=False)
        slope_ijk = self._component_extreme(data, 91, highest=False)
        base_world = nib.affines.apply_affine(affine, base_ijk)
        slope_world = nib.affines.apply_affine(affine, slope_ijk)

        f1, _, f3 = self.frankfort_points
        p1 = np.array([f1[0], f1[1], base_world[2]], dtype=float)
        p2 = np.array([f3[0], f3[1], base_world[2]], dtype=float)
        p3 = slope_world.copy()
        p3[2] -= offset_mm
        normal = np.cross(p2 - p1, p3 - p1)

        i, j = np.indices(data.shape[:2])
        zeros = np.zeros_like(i)
        world_at_k0 = nib.affines.apply_affine(
            affine, np.column_stack((i.ravel(), j.ravel(), zeros.ravel()))
        )
        if np.linalg.norm(normal) < 1e-12 or abs(normal[2]) < 1e-12:
            use_base_plane = True
        else:
            use_base_plane = False

        head = np.zeros_like(data)
        neck = np.zeros_like(data)
        for k in range(data.shape[2]):
            world = world_at_k0 + affine[:3, 2] * k
            if use_base_plane:
                threshold = base_world[2]
            else:
                plane_z = p1[2] - (
                    normal[0] * (world[:, 0] - p1[0])
                    + normal[1] * (world[:, 1] - p1[1])
                ) / normal[2]
                threshold = np.minimum(base_world[2], plane_z)
            above = (world[:, 2] >= threshold).reshape(data.shape[:2])
            head[:, :, k] = np.where(above, data[:, :, k], 0)
            neck[:, :, k] = np.where(~above, data[:, :, k], 0)

        head[data == 91] = 91
        neck[data == 50] = 50
        neck[data == 49] = 49
        neck[data == 79] = 79
        return (
            nib.Nifti1Image(head, affine, image.header),
            nib.Nifti1Image(neck, affine, image.header),
        )

    def _crop_head_and_neck(
        self, head: nib.Nifti1Image, neck: nib.Nifti1Image
    ) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
        head_data = np.asanyarray(head.dataobj)
        skull = np.argwhere(head_data == 91)
        if skull.size == 0:
            raise ValueError("Skull label 91 is missing; head/neck crop cannot be defined.")

        x0 = max(int(skull[:, 0].min()) - 15, 0)
        x1 = min(int(skull[:, 0].max()) + 15, head_data.shape[0] - 1)
        y0 = max(int(skull[:, 1].min()) - 25, 0)
        y1 = min(int(skull[:, 1].max()) + 25, head_data.shape[1] - 1)

        head_cropped = head_data[x0 : x1 + 1, :, :]
        head_affine = head.affine.copy()
        head_affine[:3, 3] += head.affine[:3, 0] * x0

        neck_data = np.asanyarray(neck.dataobj)[x0 : x1 + 1, y0 : y1 + 1, :].copy()
        neck_affine = neck.affine.copy()
        neck_affine[:3, 3] += neck.affine[:3, 0] * x0 + neck.affine[:3, 1] * y0

        c7 = np.argwhere(neck_data == 44)
        if c7.size == 0:
            raise ValueError("C7 label 44 is missing; the scan may not cover C7 completely.")
        c7_world = nib.affines.apply_affine(neck_affine, c7)
        lower_z = float(c7_world[:, 2].min())
        for k in range(neck_data.shape[2]):
            z_world = nib.affines.apply_affine(neck_affine, [0, 0, k])[2]
            if z_world < lower_z:
                neck_data[:, :, k] = 0

        return (
            nib.Nifti1Image(head_cropped, head_affine, head.header),
            nib.Nifti1Image(neck_data, neck_affine, neck.header),
        )

    def _patient_metadata(self) -> dict[str, str]:
        if not self.selected_dicom_files:
            return {}
        dataset = pydicom.dcmread(self.selected_dicom_files[0], stop_before_pixels=True)
        fields = (
            "PatientID",
            "PatientSex",
            "PatientAge",
            "StudyInstanceUID",
            "SeriesInstanceUID",
        )
        return {field: str(getattr(dataset, field, "")) for field in fields}

    @staticmethod
    def _calculate_inertia(image: nib.Nifti1Image) -> dict[str, dict]:
        data = np.asanyarray(image.dataobj)
        affine = np.asarray(image.affine, dtype=np.float64)
        voxel_basis = affine[:3, :3]
        voxel_volume_cm3 = abs(np.linalg.det(voxel_basis)) / 1000.0
        covariance = voxel_basis @ voxel_basis.T / 12.0
        voxel_inertia_per_kg = np.trace(covariance) * np.eye(3) - covariance

        records: list[tuple[float, np.ndarray, np.ndarray]] = []
        total_volume = 0.0
        total_mass = 0.0
        weighted_centroid = np.zeros(3)

        for target_label, density in DENSITY_G_CM3.items():
            ijk = np.argwhere(data == target_label)
            if ijk.size == 0:
                continue
            world = nib.affines.apply_affine(affine, ijk)
            centroid = world.mean(axis=0)
            volume = float(ijk.shape[0] * voxel_volume_cm3)
            mass = volume * density / 1000.0
            centered = world - centroid
            second_moment = centered.T @ centered
            local_inertia = (mass / ijk.shape[0]) * (
                np.trace(second_moment) * np.eye(3) - second_moment
            )
            local_inertia += mass * voxel_inertia_per_kg
            records.append((mass, centroid, local_inertia))
            total_volume += volume
            total_mass += mass
            weighted_centroid += mass * centroid

        if total_mass == 0:
            raise ValueError("The final mask contains no labels with configured densities.")

        center_of_mass = weighted_centroid / total_mass
        inertia = np.zeros((3, 3))
        for mass, centroid, local_inertia in records:
            displacement = centroid - center_of_mass
            inertia += local_inertia + mass * (
                np.dot(displacement, displacement) * np.eye(3)
                - np.outer(displacement, displacement)
            )
        inertia = (inertia + inertia.T) / 2.0
        moments, axes = np.linalg.eigh(inertia)

        return {
            "T1": {"name": "Volume", "value": total_volume, "unit": "cm³"},
            "T2": {"name": "Mass", "value": total_mass, "unit": "kg"},
            "T3": {
                "name": "Inertia Tensor (RAS)",
                "value": inertia.tolist(),
                "unit": "kg·mm²",
            },
            "T4": {
                "name": "Inertia Tensor (RAS)",
                "value": (inertia / 100.0).tolist(),
                "unit": "kg·cm²",
            },
            "T7": {
                "name": "Center of Mass (RAS)",
                "value": center_of_mass.tolist(),
                "unit": "mm",
            },
            "T8": {
                "name": "Principal Moments of Inertia",
                "value": moments.tolist(),
                "unit": "kg·mm²",
            },
            "T9": {
                "name": "Principal Axes (column vectors, RAS)",
                "value": axes.tolist(),
                "unit": "unit vectors",
            },
        }

    def _frankfort_frame(self) -> tuple[np.ndarray, np.ndarray]:
        if self.frankfort_points is None:
            raise RuntimeError("Frankfort landmarks have not been calculated.")
        f1, f2, f3 = self.frankfort_points
        origin = (f1 + f3) / 2.0
        y_axis = f3 - f1
        y_axis /= np.linalg.norm(y_axis)
        x_axis = f2 - origin
        x_axis -= np.dot(x_axis, y_axis) * y_axis
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.cross(y_axis, x_axis)
        z_axis /= np.linalg.norm(z_axis)
        axes = np.column_stack((x_axis, y_axis, z_axis))
        return axes, origin

    @staticmethod
    def _neck_origin(image: nib.Nifti1Image, target_label: int) -> np.ndarray:
        data = np.asanyarray(image.dataobj)
        ijk = np.argwhere(data == target_label)
        if ijk.size == 0:
            raise ValueError(f"Neck-origin label {target_label} is missing.")
        world = nib.affines.apply_affine(image.affine, ijk)
        x_mid = (world[:, 0].min() + world[:, 0].max()) / 2.0
        posterior = world[np.argmax(world[:, 1])]
        return np.array([x_mid, posterior[1], posterior[2]])

    @staticmethod
    def _transform_inertia(
        result: dict[str, dict], axes: np.ndarray, origin: np.ndarray
    ) -> dict[str, dict]:
        rotation = axes.T
        center = rotation @ (np.asarray(result["T7"]["value"]) - origin)
        inertia_ras = np.asarray(result["T3"]["value"])
        inertia_local = rotation @ inertia_ras @ axes
        principal_axes = rotation @ np.asarray(result["T9"]["value"])
        return {
            "T1": result["T1"],
            "T2": result["T2"],
            "T7_Frankfort": {
                "name": "Center of Mass (Frankfort)",
                "value": center.tolist(),
                "unit": "mm",
            },
            "T3_Frankfort": {
                "name": "Inertia Tensor (Frankfort)",
                "value": inertia_local.tolist(),
                "unit": "kg·mm²",
            },
            "T4_Frankfort": {
                "name": "Inertia Tensor (Frankfort)",
                "value": (inertia_local / 100.0).tolist(),
                "unit": "kg·cm²",
            },
            "T8": result["T8"],
            "T9_Frankfort": {
                "name": "Principal Axes (column vectors, Frankfort)",
                "value": principal_axes.tolist(),
                "unit": "unit vectors",
            },
        }

    @staticmethod
    def _write_result(
        path: Path, metadata: dict[str, str], result_key: str, result: dict[str, dict]
    ) -> None:
        payload = {"Patient Metadata": metadata, result_key: result}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Wrote %s", path)

