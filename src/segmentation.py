import tifffile 
import numpy as np
from skimage import filters, morphology, measure
from scipy import ndimage
import pandas as pd
from pathlib import Path




def process_image(image_path, min_nucleus_size=100, save_labels_path=None):
    """
    Segment nuclei from a fluorescence image and compute per-nucleus measurements.

    This function performs background correction, nucleus segmentation, and
    instance labeling on a single 2D fluorescence image. Detected nuclei are
    labeled as distinct connected components and saved as an integer-valued
    label image. Quantitative measurements are computed per nucleus and can be
    exported for downstream analysis.

    Parameters
    ----------
    image_path : str or Path
        Path to a 2D fluorescence microscopy image containing nuclear signal.

    min_nucleus_size : int, optional
        Minimum size (in pixels) for a detected nucleus. Objects smaller than
        this threshold are removed to suppress noise and debris.

    save_labels_path : str or Path, optional
        Directory path where the label image will be saved. If None, the label
        image is not saved to disk. Default is None.

    Returns
    -------
    label_img : np.ndarray
        Integer-valued label image of the same spatial dimensions as the input
        image, where 0 denotes background and positive integers denote individual
        detected nuclei.

    df : per nucleus measurements table for the image 
    Notes
    -----
    - Background correction is performed using global median subtraction.
    - Nucleus segmentation is based on Otsu thresholding followed by
      morphological cleanup.
    - The label image is intended for quantitative analysis and visualization.
    - This function does not perform cytoplasm or whole-cell segmentation.
    """
    image_path = Path(image_path)
    image_id = image_path.stem

    # load image
    img = tifffile.imread(image_path)

    # background correction
    bg = np.median(img)
    fl_corr = np.clip(img.astype(float) - bg, 0, None)

    # segmentation
    thresh = filters.threshold_otsu(img - bg)
    nuc_mask = img > thresh
    nuc_mask = morphology.remove_small_objects(nuc_mask, min_size=min_nucleus_size)
    nuc_mask = ndimage.binary_fill_holes(nuc_mask)
    label_img = measure.label(nuc_mask)

    # save label image (optional)
    if save_labels_path is not None:
        save_labels_path = Path(save_labels_path)
        save_labels_path.mkdir(exist_ok=True, parents=True)
        tifffile.imwrite(
            save_labels_path / f"{image_id}_labels.tif",
            label_img.astype(np.int32)
        )

    # measurements
    props = measure.regionprops_table(
        label_img,
        intensity_image=fl_corr,
        properties=("label", "area", "mean_intensity")
    )

    df = pd.DataFrame(props)
    df["integrated_intensity"] = df["area"] * df["mean_intensity"]
    df["image_id"] = image_id

    return label_img, df