import pandas as pd 
import matplotlib.pyplot as plt


def summarize_measurements(df):
    """
    Compute basic summary statistics for nucleus measurements.
    """
    summary = {
        "num_nuclei": len(df),
        "mean_area": df["area"].mean(),
        "median_area": df["area"].median(),
        "min_area": df["area"].min(),
        "max_area": df["area"].max(),
        "mean_intensity": df["mean_intensity"].mean(),
    }
    return summary


def per_image_qc(df, min_nuclei = 5, max_nuclei = 5000):
    '''
    Flag images with unexpected nuclei counts for quality control purposes.
    '''
    counts = df.groupby('image_id').size()
    flags = []
    for image_id, count in counts.items():
        if count < min_nuclei:
            flags.append((image_id, 'Too few nuclei'))
        elif count > max_nuclei:
            flags.append((image_id, 'Too many nuclei'))

    return flags


def plot_nuclei_area_distribution(df, output_path):
    '''
    Generate histogram of nucleus area distribution across all images and save to file.
    '''
    plt.figure()
    df['area'].hist(bins=50)
    plt.xlabel('Nucleus Area (pixels)')
    plt.ylabel('Count')
    plt.title('Distribution of Nucleus Areas')
    plt.savefig(output_path)
    plt.close()



print(4)