from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
import boto3
import json

# Input and Output S3 paths
s3_input_bucket = "ccdout-bucket"
s3_input_folder = "/in_bdr_response"  # Adjusted to the correct folder path
s3_output_folder = "out_bdr_response"
s3_output_bucket = "ccdout-bucket"

# Add Hadoop configurations for S3
hadoop_conf = {
    "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
    "spark.hadoop.fs.s3a.awsAccessKeyId": "YOUR_ACCESS_KEY",
    "spark.hadoop.fs.s3a.awsSecretAccessKey": "YOUR_SECRET_KEY"
}

def invoke_bedrock_model(prompt):
    # Your existing function for invoking Bedrock model

if __name__ == "__main__":
    # Create a Spark session with Hadoop configurations
    with SparkSession.builder \
            .appName("ParquetToBedrock") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.awsAccessKeyId", "YOUR_ACCESS_KEY") \
            .config("spark.hadoop.fs.s3a.awsSecretAccessKey", "YOUR_SECRET_KEY") \
            .getOrCreate() as spark:

        # Your existing code for listing Parquet files and processing them
        parquet_files = (
            boto3.client("s3")
            .list_objects(Bucket=s3_input_bucket, Prefix=s3_input_folder)
            .get("Contents", [])
        )

        schema = StructType([StructField("first_name", StringType(), True)])
        transcripts_data_frames = [
            spark.read.schema(schema).parquet(f"s3a://{s3_input_bucket}/{file['Key']}")
            for file in parquet_files
        ]

        result_transcripts = " ".join(
            [row.first_name for df in transcripts_data_frames for row in df.collect()]
        )

        custom_prompt = "\n \n Please summarize the following call transcript input \n "
        bedrock_api_input = f"{custom_prompt} {result_transcripts}"

        print("The Bedrock API input is as below \n\n" + bedrock_api_input)
        print("\n \n The type of bedrock_input_variable is ", (type(bedrock_api_input)))

        bedrock_result = invoke_bedrock_model(bedrock_api_input)

        if bedrock_result is not None:
            print("\nBedrock API Result:")
            print(bedrock_result)

            s3_client = boto3.client("s3", region_name="us-east-1")
            s3_key = f"{s3_output_folder}/bedrock_output_s0glu.txt"

            try:
                s3_client.put_object(
                    Body=bedrock_result, Bucket=s3_output_bucket, Key=s3_key
                )
                print(f"Result uploaded to S3: {s3_key}")

            except Exception as e:
                print(f"Error uploading result to S3: {e}")

    # Stop the Spark session
    spark.stop()
