import argparse
import os
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from s3_access import S3Access
from image_table_base import Image_table_base
from get_sql import get_hash_and_gender_dataframe
from make_df import load_image_dataframe

DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME', 'imags')
DB_USER = os.environ.get('DB_USER', 'imagetraineruser')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

# Set up the SQLAlchemy engine and sessionmaker globally
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"  # noqa E231
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

# connection to s3
s3access = S3Access(S3_BUCKET_NAME)


def test_sql_connection():
    """
    Test the SQL connection by performing 'SELECT * FROM images LIMIT 1'.
    Returns True and prints a happy message if successful,
    otherwise returns False and prints the error.
    """
    try:
        session = Session()
        session.query(Image_table_base).limit(1).all()
        print("\U0001F60A Successfully connected to the \
              database and queried the images table!")
        session.close()
        return True
    except Exception as e:
        print(f"\U0001F622 Failed to connect or query the database: {e}")
        return False


def check_numpys(dataframe, sample_size=15):
    """
    Take a random sample of <sample_size> from the dataframe and verify that \
        no two numpy arrays are identical.
    """
    print(f"\nVerifying {sample_size} random numpy \
          arrays for uniqueness...")
    sample_df = dataframe.sample(n=min(sample_size,
                                 len(dataframe)),
                                 random_state=42)
    arrays = sample_df['image'].tolist()
    duplicates_found = False
    for i in range(len(arrays)):
        for j in range(i+1, len(arrays)):
            if arrays[i] is not None and arrays[j] is not None:
                if np.array_equal(arrays[i], arrays[j]):
                    print(f"Duplicate found between indices \
                          {sample_df.index[i]} and {sample_df.index[j]}")
                    duplicates_found = True
    if not duplicates_found:
        print("\U0001F60A All sampled numpy arrays are unique!")
    else:
        print("\U0001F622 Duplicate numpy arrays found in the sample!")


def main():
    parser = argparse.ArgumentParser(
        description="Training script for ML console.")
    subparsers = parser.add_subparsers(
        dest="command", required=True)

    # 'frames' subcommand
    frames_parser = subparsers.add_parser("frames",
                                          help="Work with frames data")
    frames_parser.add_argument(
        "--sql-only",
        action="store_true",
        help="Only tests connection to SQL."
    )
    frames_parser.add_argument(
        "--verify-numpys",
        type=int,
        nargs="?",
        const=15,
        default=None,
        metavar="N",
        help="Verify that a random sample of N \
            numpy arrays are unique (default: 15)."
    )
    frames_parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="FILENAME",
        help="Save the resulting dataframe to\
        /mnt/ebs_volume/<filename> as a pickle file."
    )

    # 'train' subcommand
    train_parser = subparsers.add_parser("train", help="Train \
                                         a machine learning model")
    train_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["forest", "regression", "vector"],
        help="Specify which model to train: 'forest' \
            (Random Forest Classifier), 'regression' (Logistic Regression), \
                or 'vector' (Support Vector Machine)."
    )
    # The --model argument determines which ML model to use:
    #   'forest'    -> Random Forest Classifier
    #   'regression'-> Logistic Regression
    #   'vector'    -> Support Vector Machine

    args = parser.parse_args()

    if args.command == "frames":
        if args.sql_only:
            test_sql_connection()
        else:
            data_frame = get_hash_and_gender_dataframe(Session())
            if data_frame is not None:
                print("\U0001F60A Successfully retrieved the data frame!")
            complete = load_image_dataframe(data_frame, s3access)
            print("\nDataFrame info:")
            complete.info()
            print("\nDataFrame head:")
            print(complete.head())
            if args.verify_numpys is not None:
                check_numpys(complete, sample_size=args.verify_numpys)
            if args.save:
                save_path = f"/mnt/ebs_volume/{args.save}"
                complete.to_pickle(save_path)
                print(f"\U0001F4BE DataFrame saved\
                      to {save_path} as a pickle file.")
    elif args.command == "train":
        # Placeholder for training logic based on args.model
        pass


if __name__ == "__main__":
    main()
