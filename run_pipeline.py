import argparse 
from pathlib import Path
import pandas as pd 
import tifffile 

from segmentation import process_image
from visualization import save_overlay, save_boundaries, save_label_mask


def run_pipeline(input_dir, output_dir, min_nucleus_size=100):
    """
    Run the full image analysis pipeline on all images in the input directory.

    This function orchestrates the entire workflow, including image loading,
    nucleus segmentation, measurement extraction, and visualization generation.
    Results are saved to the specified output directory for downstream analysis
    and quality control.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing raw fluorescence microscopy images to be analyzed.

    output_dir : str or Path
        Directory where results (measurements, overlays, boundaries) will be saved.

    min_nucleus_size : int, optional
        Minimum size (in pixels) for a detected nucleus. Objects smaller than
        this threshold are removed to suppress noise and debris.

    Notes
    -----
    - The function processes all TIFF images in the input directory.
    - Output includes per-nucleus measurements in CSV format and visualizations
      in PNG format for each processed image.
    - Quality control flags can be generated based on measurement summaries.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    labels_dir = output_dir / "labels"
    overlays_dir = output_dir / "overlays"
    boundaries_dir = output_dir / "boundaries"

    labels_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    boundaries_dir.mkdir(parents=True, exist_ok=True)   

    all_results = [] 

    for img_path in sorted(input_dir.glob("*.tif")):
        image_id = img_path.stem
        print(f"Processing {img_path.name}...")
        label_img, df = process_image(img_path, min_nucleus_size=min_nucleus_size, save_labels_path=labels_dir)

        image = tifffile.imread(img_path)

        save_overlay(
            image,
            label_img,
            overlays_dir / f"{image_id}_overlays.png"
        )

        save_boundaries(
            image,
            label_img,
            boundaries_dir / f"{image_id}_boundaries.png"
        )

        save_label_mask(
            label_img,
            labels_dir / f"{image_id}_labels.tif"
        )

        all_results.append(df)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(output_dir / "nucleus_measurements.csv", index=False)
        print(f"Saved measurements for {len(final_df)} nuclei to {output_dir 
                            / 'nucleus_measurements.csv'}")
    else:
        print("No images processed. No measurements saved.")



# Main function to allow running as a script

def main():
    print('Running main...')
    parser = argparse.ArgumentParser(description="Run nucleus segmentation and measurement pipeline.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--min_nucleus_size", 
                        type=int,
                          default=100, 
                          help="Minimum size (in pixels) for detected nuclei.")
    
    args = parser.parse_args()
    
    run_pipeline(args.input_dir, args.output_dir, args.min_nucleus_size)


if __name__ == "__main__":
    main()

    
