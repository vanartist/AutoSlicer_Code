import logging
from pathlib import Path

from autoslicer import AutoSlicer


# Edit these two folders before running.
DICOM_FOLDER = Path(r"D:\CT\case001")
OUTPUT_FOLDER = Path(r"D:\results\case001")
DEVICE = "auto"
SERIES_UID = None
LOWER_HU = None
UPPER_HU = None
KEEP_INTERMEDIATE = False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pipeline = AutoSlicer(
        OUTPUT_FOLDER,
        device=DEVICE,
        lower_hu=LOWER_HU,
        upper_hu=UPPER_HU,
        keep_intermediate=KEEP_INTERMEDIATE,
    )
    outputs = pipeline.run(DICOM_FOLDER, series_uid=SERIES_UID)
    print("\nCompleted. Key outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
