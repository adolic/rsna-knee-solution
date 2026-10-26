"""DICOM reading: one decode per instance, slice ordering, per-series 1-99 % window inputs.

Pixel decode with the two known malformed-pixel repairs in the RSNA training data, MONOCHROME1 inversion, modality LUT,
orientation / position / spacing validation, and slice ordering along the median slice normal. No crop, centre or
resize logic (that is keypoint based, pipeline.py).
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian

KNOWN_MALFORMED_PIXEL_REPAIRS = {
    "1.2.826.0.1.3680043.8.498.41947538458330624797291612781685746320": {
        "original_sha256": "cfe23f2a97ba3fc07066bfed047c6701d9982a54e4913205a6c30366906a455c",
        "rows": 640,
        "columns": 640,
    },
    "1.2.826.0.1.3680043.8.498.66460523372602171398246641420761543626": {
        "original_sha256": "c057554899ba81249d280e880226bf28b089b1e33235e3cfd101bfc36a399d7a",
        "rows": 416,
        "columns": 416,
    },
}

def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


PIXEL_REPAIR_METHOD = (
    "zero_extend_unsigned_u8_samples_to_explicit_vr_little_endian_u16_v1"
)

def _normalized(value: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return None
    return value / norm

def _pixel_array_with_known_repairs(
    dataset: pydicom.Dataset, path: Path, source_sha256: str | None = None
) -> tuple[np.ndarray, dict[str, Any] | None]:
    try:
        return np.asarray(dataset.pixel_array), None
    except ValueError as original_error:
        sop_uid = str(getattr(dataset, "SOPInstanceUID", ""))
        repair = KNOWN_MALFORMED_PIXEL_REPAIRS.get(sop_uid)
        if repair is None:
            raise
        expected = {
            "transfer_syntax_uid": str(ExplicitVRLittleEndian),
            "rows": int(repair["rows"]),
            "columns": int(repair["columns"]),
            "samples_per_pixel": 1,
            "photometric_interpretation": "MONOCHROME2",
            "number_of_frames": 1,
            "bits_allocated": 16,
            "bits_stored": 12,
            "high_bit": 11,
            "pixel_representation": 0,
            "pixel_data_vr": "OW",
        }
        observed = {
            "transfer_syntax_uid": str(dataset.file_meta.TransferSyntaxUID),
            "rows": int(dataset.Rows),
            "columns": int(dataset.Columns),
            "samples_per_pixel": int(dataset.SamplesPerPixel),
            "photometric_interpretation": str(dataset.PhotometricInterpretation),
            "number_of_frames": int(getattr(dataset, "NumberOfFrames", 1) or 1),
            "bits_allocated": int(dataset.BitsAllocated),
            "bits_stored": int(dataset.BitsStored),
            "high_bit": int(dataset.HighBit),
            "pixel_representation": int(dataset.PixelRepresentation),
            "pixel_data_vr": str(dataset["PixelData"].VR),
        }
        if observed != expected:
            raise ValueError(
                f"Known malformed DICOM contract changed for {path}: {observed}"
            ) from original_error
        pixel_data = bytes(dataset.PixelData)
        pixel_count = expected["rows"] * expected["columns"]
        if len(pixel_data) != pixel_count:
            raise ValueError(
                f"Known malformed DICOM payload changed for {path}: "
                f"{len(pixel_data)} != {pixel_count} bytes"
            ) from original_error
        digest = source_sha256 or _sha256(path)
        if digest != repair["original_sha256"]:
            raise ValueError(
                f"Known malformed DICOM hash changed for {path}: {digest}"
            ) from original_error
        pixels = np.frombuffer(pixel_data, dtype=np.uint8).astype(np.uint16)
        pixels = pixels.reshape(expected["rows"], expected["columns"])
        return pixels, {
            "sop_instance_uid": sop_uid,
            "source_sha256": digest,
            "method": PIXEL_REPAIR_METHOD,
        }

def read_once(path):
    """One source read and pixel decode; preserve headers when pixels are bad."""
    before = path.stat()
    record = {"path": path, "sop": path.stem, "instance": 0,
              "orientation": None, "position": None, "values": None, "repair": None,
              "source_bytes": before.st_size}
    try:
        source = path.read_bytes()
        ds = pydicom.dcmread(io.BytesIO(source))
        record.update(sop=str(getattr(ds, "SOPInstanceUID", path.stem)),
                      instance=int(getattr(ds, "InstanceNumber", 0) or 0),
                      study=str(getattr(ds, "StudyInstanceUID", "")),
                      series=str(getattr(ds, "SeriesInstanceUID", "")))
        # Knee-frame input (knee-frame-v1 catalogue rule): Laterality, else ImageLaterality, first letter R/L.
        side = str(getattr(ds, "Laterality", "") or "").strip() or str(getattr(ds, "ImageLaterality", "") or "").strip()
        record["laterality"] = side.upper()[:1] if side.upper()[:1] in ("R", "L") else None
        for key, tag, size in [("orientation", "ImageOrientationPatient", 6),
                               ("position", "ImagePositionPatient", 3)]:
            value = getattr(ds, tag, None)
            if value is not None and len(value) == size:
                array = np.asarray(value, dtype=np.float64)
                if np.isfinite(array).all():
                    record[key] = array
        try:
            instance = {"image_position_lps": record["position"], "image_orientation_lps": record["orientation"],
                        "pixel_spacing_mm": [float(x) for x in ds.PixelSpacing], "rows": int(ds.Rows), "columns": int(ds.Columns)}
            record["geometry"] = fov_geometry(instance)
            g = record["geometry"]
            record["acquired_centre"] = (g["ipp"] + g["row_dir"] * (g["cols"] - 1) / 2 * g["d_col"]
                                         + g["col_dir"] * (g["rows"] - 1) / 2 * g["d_row"])
        except (ValueError, TypeError, AttributeError):
            record["geometry"] = None
        known_hash = hashlib.sha256(source).hexdigest() if record["sop"] in KNOWN_MALFORMED_PIXEL_REPAIRS else None
        pixels, repair = _pixel_array_with_known_repairs(ds, path, source_sha256=known_hash)
        try:
            from pydicom.pixels import apply_modality_lut
        except ImportError:
            from pydicom.pixel_data_handlers.util import apply_modality_lut
        values = np.asarray(apply_modality_lut(pixels, ds), dtype=np.float32)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 2 or values.shape != (int(ds.Rows), int(ds.Columns)) or not np.isfinite(values).all():
            raise ValueError("Invalid decoded pixel geometry/values")
        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            values = values.max() - values
        spacing = [float(x) for x in getattr(ds, "PixelSpacing", (.5, .5))]
        if len(spacing) != 2 or not np.isfinite(spacing).all() or min(spacing) <= 0:
            raise ValueError("Invalid PixelSpacing")
        record.update(values=values, spacing=spacing, repair=repair)
    except (OSError, ValueError, TypeError, AttributeError, RuntimeError, NotImplementedError,
            pydicom.errors.InvalidDicomError) as error:
        record["pixel_error"] = f"{type(error).__name__}: {error}"
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed during inference: {path}")
    return record

def order_records(records):
    if len({r["sop"] for r in records}) != len(records):
        raise ValueError("Duplicate SOPInstanceUID in one series")
    complete = bool(records) and all(r["orientation"] is not None and r["position"] is not None for r in records)
    normal = None
    if complete:
        normals = [_normalized(np.cross(r["orientation"][:3], r["orientation"][3:])) for r in records]
        if all(n is not None for n in normals):
            reference = normals[0]
            aligned = [-n if float(np.dot(n, reference)) < 0 else n for n in normals]
            normal = _normalized(np.median(np.stack(aligned), axis=0))
            if normal is not None and normal[int(np.argmax(abs(normal)))] < 0:
                normal = -normal
    if normal is not None:
        records.sort(key=lambda r: (float(np.dot(r["position"], normal)), r["instance"], r["sop"]))
    else:
        records.sort(key=lambda r: (r["instance"], r["sop"]))
    return records

def fov_geometry(instance):
    """Validated per-slice LPS frame: increasing-column then increasing-row axes."""
    p = np.asarray(instance["image_position_lps"], np.float64)
    orient = np.asarray(instance["image_orientation_lps"], np.float64)
    spacing = np.asarray(instance["pixel_spacing_mm"], np.float64)
    h, w = int(instance["rows"]), int(instance["columns"])
    if p.shape != (3,) or orient.shape != (6,) or spacing.shape != (2,):
        raise ValueError("Incomplete pixel geometry")
    if not np.isfinite(np.r_[p, orient, spacing]).all() or (spacing <= 0).any() or min(h, w) < 1:
        raise ValueError("Invalid pixel geometry")
    u, v = orient[:3], orient[3:]
    if max(abs(np.linalg.norm(u)-1), abs(np.linalg.norm(v)-1), abs(u@v)) > 1e-3:
        raise ValueError("Nonorthonormal image geometry")
    return {"ipp": p, "row_dir": u, "col_dir": v,
            "d_row": float(spacing[0]), "d_col": float(spacing[1]),
            "rows": h, "cols": w}

# Per-series intensity window percentiles.
PCT_LO, PCT_HI = 1.0, 99.0
