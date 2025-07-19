import pandas as pd
from sqlalchemy.orm import Session
from image_table_base import Image_table_base


def get_hash_and_gender_dataframe(session: Session):
    """
    Query the database for hash and is_masc_human columns where
    is_masc_human is not null and deleted_at is null,
    and return the results as a pandas DataFrame.
    Also prints summary statistics.
    """
    results = session.query(
        Image_table_base.hash,
        Image_table_base.is_masc_human
    ).filter(
        Image_table_base.is_masc_human.isnot(None),
        Image_table_base.deleted_at.is_(None)
    ).all()

    # Convert results to DataFrame
    df = pd.DataFrame(results, columns=["hash", "is_masc_human"])

    # Print summary statistics
    total_rows = len(df)
    num_true = (df["is_masc_human"] == True).sum()  # noqa E712
    num_false = (df["is_masc_human"] == False).sum()  # noqa E712
    print(f"Total rows: {total_rows}")
    print(f"Total Male Presenting: {num_true}")
    print(f"Total Fem Presenting: {num_false}")

    return df
