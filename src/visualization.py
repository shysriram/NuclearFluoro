
# Library imports 
import numpy as np
import matplotlib.pyplot as plt 
from skimage.segmentation import find_boundaries
from skimage.color import label2rgb
from pathlib import Path
import tifffile 



def save_overlay(image, label_img, output_path):
    """
    Save a visualization overlay of nuclear segmentation results.

    This function creates a human-readable overlay image by rendering
    instance-level nucleus labels on top of the raw fluorescence image.
    Each nucleus is shown in a distinct color for visual separation.
    The background image is normalized and included for visual context only.

    Parameters
    ----------
    image : np.ndarray
        Raw fluorescence image (2D array). This image is used only for
        visualization and does not affect segmentation or measurements.

    label_img : np.ndarray
        Integer-valued label image produced by the segmentation step,
        where 0 denotes background and positive integers denote individual nuclei.

    output_path : str or Path
        File path where the overlay image (PNG) will be saved.

    Notes
    -----
    - Overlay images are intended for qualitative inspection only.
    - Quantitative measurements are derived exclusively from `label_img`.
    - Visualization choices (normalization, colors, alpha blending)
      do not affect segmentation results or measurements.

    """
    overlay = label2rgb(label_img, image=image, bg_label=0, alpha=0.3)
    plt.imsave(output_path, overlay)

output_overlay = Path("overlay_results")
output_overlay.mkdir(exist_ok=True)



def save_boundaries(image, label_img, output_path):
    """
    Save a visualization of nuclear segmentation boundaries overlaid on the raw image.

    This function creates an image that highlights the boundaries of segmented nuclei
    on top of the original fluorescence image. The boundaries are shown in red for
    clear visibility against the grayscale background.

    Parameters
    ----------
    image : np.ndarray
        Raw fluorescence image (2D array). This image is used only for
        visualization and does not affect segmentation or measurements.

    label_img : np.ndarray
        Integer-valued label image produced by the segmentation step,
        where 0 denotes background and positive integers denote individual nuclei.

    output_path : str or Path
        File path where the boundary overlay image (PNG) will be saved.

    Notes
    -----
    - Boundary images are intended for qualitative inspection only.
    - Quantitative measurements are derived exclusively from `label_img`.
    - Visualization choices (normalization, colors, alpha blending)
      do not affect segmentation results or measurements.

    """
    image = image.astype(float)
    image_norm = (image - image.min()) / (image.max() - image.min()+1e-8)
    boundaries = find_boundaries(label_img, mode='outer')
    boundary_overlay = np.dstack([image_norm]*3)
    boundary_overlay[boundaries] = [1.0, 1.0, 1.0] 
    plt.imsave(output_path, boundary_overlay)


def save_label_mask(label_img, output_path):
    """
    Save the raw integer-valued label image produced by the segmentation step.

    This function writes the label image directly to disk without any
    visualization or color mapping. Each pixel's value corresponds to its
    assigned nucleus label (0 for background, positive integers for individual
    nuclei).

    Parameters
    ----------
    label_img : np.ndarray
        Integer-valued label image produced by the segmentation step.

    output_path : str or Path
        File path where the label image (TIFF) will be saved.

    Notes
    -----
    - The saved label image is intended for quantitative analysis and reuse.
    - The image may appear black in standard image viewers because it contains
      integer labels rather than grayscale intensities.
    """
    tifffile.imwrite(output_path, label_img.astype(np.int32))

