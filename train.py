import argparse
import os
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
    elif args.command == "train":
        # Placeholder for training logic based on args.model
        pass


if __name__ == "__main__":
    main()
