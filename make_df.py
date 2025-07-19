import pandas as pd
import numpy as np
from s3_access import S3Access
import io
from typing import Optional


def load_image_dataframe(df: pd.DataFrame,
                         s3_access: S3Access) -> pd.DataFrame:
    """
    Load numpy arrays from S3 and add them as 'image' column to the DataFrame.

    Args:
        df: DataFrame with columns 'hash' and 'is_masc'
        s3_access: Instance of S3Access class

    Returns:
        DataFrame with added 'image' column containing numpy arrays
    """

    result_df = df.copy()

    # Initialize the image column with None values
    result_df['image'] = None

    # Track statistics
    successful_loads = 0
    failed_loads = 0
    failed_hashes = []

    print(f"Starting to load {len(result_df)} numpy arrays from S3...")

    # Iterate through each row and load the corresponding numpy array
    for idx, row in result_df.iterrows():
        hash = row['hash']

        try:
            # Construct the S3 key for the numpy file
            s3_key = f"numpys/{hash}.npy"

            # Retrieve the object from S3
            obj_data = s3_access.get_object(s3_key)

            if isinstance(obj_data, bytes):
                # If get_object returns bytes, use BytesIO
                numpy_array = np.load(io.BytesIO(obj_data))
            else:
                numpy_array = np.load(obj_data)

            # Add the numpy array to the DataFrame
            result_df.at[idx, 'image'] = numpy_array
            successful_loads += 1

            # Print progress every 50 items
            if (successful_loads + failed_loads) % 50 == 0:
                print(f"Processed \
                      {successful_loads + failed_loads}/{len(result_df)} \
                        items...")

        except Exception as e:
            print(f"Failed to load {hash}.npy: {str(e)}")
            failed_loads += 1
            failed_hashes.append(hash)
            # Keep None in the image column for failed loads

    # Print summary statistics
    print(f"\nLoading complete!")
    print(f"Successfully loaded: {successful_loads}")
    print(f"Failed to load: {failed_loads}")
    print(f"Success rate: \
          {successful_loads/(successful_loads + failed_loads)*100:.1f}%")

    if failed_hashes:
        print(f"Failed hash names: {failed_hashes[:10]}...")  # Show first 10

    return result_df


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame has the expected structure.
    """
    required_columns = ['hash', 'is_masc']

    if not all(col in df.columns for col in required_columns):
        print(f"Error: DataFrame must contain columns: {required_columns}")
        return False

    if df.empty:
        print("Error: DataFrame is empty")
        return False

    print(f"DataFrame validation passed. Shape: {df.shape}")
    return True


def main():
    """
    Main function to load and process the image classification DataFrame.
    """
    # Initialize S3Access (adjust parameters as needed for your setup)
    s3_access = S3Access()  # Add any required initialization parameters

    sample_data = {
        'hash': ['abc123', 'def456', 'ghi789'],
        'is_masc': [True, False, True]
    }
    df = pd.DataFrame(sample_data)

    # Validate the DataFrame structure
    if not validate_dataframe(df):
        return

    print(f"Original DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Load the image data
    enhanced_df = load_image_dataframe(df, s3_access)

    # Display information about the enhanced DataFrame
    print(f"\nEnhanced DataFrame shape: {enhanced_df.shape}")
    print(f"Columns: {list(enhanced_df.columns)}")

    # Check how many rows have successfully loaded images
    non_null_images = enhanced_df['image'].notna().sum()
    print(f"Rows with loaded images: {non_null_images}/{len(enhanced_df)}")

    print("\nSample of enhanced DataFrame:")
    sample_df = enhanced_df.head().copy()
    sample_df['image'] = sample_df['image'].apply(
        lambda x: f"numpy array shape: {x.shape}" if x is not None else "None"
    )
    print(sample_df)

    return enhanced_df


if __name__ == "__main__":
    enhanced_dataframe = main()
