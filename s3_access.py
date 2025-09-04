import boto3
from io import BytesIO
import numpy as np
from botocore.exceptions import ClientError, IncompleteReadError


class S3Access:
    """S3 access class for managing S3 bucket operations."""

    def __init__(self, bucket_name):
        """
        Initialize S3Access with a bucket name.

        @Args:
            bucket_name (str): Name of the S3 bucket to connect to
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')

    def get_object(self, key):
        """
        Get an object from S3 with the specified key.

        Args:
            key (str): Key name of the S3 object to retrieve

        Returns:
            bytes: File content as bytes, or None if error
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=key
            )

            file_content = response['Body'].read()
            print(f"Successfully retrieved object {key}")
            return file_content

        except IncompleteReadError:
            print(f"IncompleteReadError encountered. {key}")

            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=key
                )
                file_content = response['Body'].read()
                np_array = np.load(BytesIO(file_content))
                print(f"Successfully loaded numpy array from {key}.")
                return np_array
            except Exception as e:
                print(f"Error loading numpy array from {key}: {e}")
                return None

        except ClientError as e:
            print(f"Error retrieving object {key}: {e}")
            return None

    def object_exists(self, key):
        """
        Check if an object exists in S3 with the specified key.

        Args:
            key (str): Key name of the S3 object to check

        Returns:
            bool: True if object exists, False otherwise
        """
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=key
            )
            return True

        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                print(f"Error checking if object {key} exists: {e}")
                return False

    def delete_object(self, key):
        """
        Delete an object from S3 with the specified key.

        Args:
            key (str): Key name of the S3 object to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=key
            )

            print(f"Successfully deleted object {key}")
            return True

        except ClientError as e:
            print(f"Error deleting object {key}: {e}")
            return False
