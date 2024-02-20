from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
import boto3
import json

# Hardcoded Input and Output S3 paths
s3_input_bucket = "ccdout-bucket"
#s3_input_folder="s3://ccdout-bucket/in_bdr_response/call.parquet"
s3_input_folder="s3://ccdout-bucket/in_bdr_response/" 
#s3_input_folder="/in_bdr_response/" 

s3_output_folder = "out_bdr_response"
s3_output_bucket = "ccdout-bucket"

def invoke_bedrock_model(prompt):
    # Create a Bedrock runtime client
    bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Bedrock model configuration
    model_id = "amazon.titan-text-express-v1"
    configs = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0,
            "topP": 1,
        },
    }

    # Convert configuration to JSON
    body = json.dumps(configs)
    accept = "application/json"
    content_type = "application/json"

    try:
        # Invoke the Bedrock model
        response = bedrock_runtime.invoke_model(
            body=body, modelId=model_id, accept=accept, contentType=content_type
        )

        # Parse the response
        response_body = json.loads(response["body"].read())

        # Return the output text
        return response_body.get("results")[0].get("outputText")

    except Exception as e:
        print(f"Error uploading result to S3: {e}")





def get_s3_input_files(bucket, prefix):
    # List all Parquet files in the specified S3 folder
    s3_objects = boto3.client("s3").list_objects(Bucket=bucket, Prefix=prefix)
    return [file["Key"] for file in s3_objects.get("Contents", [])]


def read_transcripts_column(spark, s3_folder_path):
    # Read only the 'transcripts/first_name' column from each parquet file into a DataFrame
    schema = StructType([StructField("first_name", StringType(), True)])
    df = spark.read.schema(schema).parquet(f"s3a://{s3_folder_path}")
    return df

def concatenate_transcripts(dfs):
    result_string = " ".join([row.first_name for df in dfs for row in df.collect()])
    return result_string

def write_result_to_s3(result_text, s3_output_path):
    # Create an S3 client
    s3_client = boto3.client("s3", region_name="us-east-1")

    # Specify the S3 path for storing the result
    s3_key = f"{s3_output_path}/bedrock_output_s0glu.txt"

    try:
        # Upload the result to S3
        s3_client.put_object(Body=result_text, Bucket=s3_output_bucket, Key=s3_key)
        print(f"Result uploaded to S3: {s3_key}")

    except Exception as e:
        print(f"Error uploading result to S3: {e}")
        
        


if __name__ == "__main__":
    # Create a Spark session
    with SparkSession.builder.appName("ParquetToBedrock").getOrCreate() as spark:
        # Get the list of Parquet files from the S3 bucket
        parquet_files = get_s3_input_files(s3_input_bucket, s3_input_folder)

        # Read only the 'transcripts' column from parquet files into DataFrame
        transcripts_data_frames = [
            read_transcripts_column(spark, file) for file in parquet_files
        ]

        # Join 'transcripts' content into a single string
        result_transcripts = concatenate_transcripts(transcripts_data_frames)

        # Custom prompt for Bedrock API
        custom_prompt = "Please summarize the following call transcript input \n \n"

        # Join custom prompt and 'transcripts' string for Bedrock API input
        bedrock_api_input = f"{custom_prompt} {result_transcripts}"

        # Print Bedrock API input
        print("The Bedrock API input is as below \n\n" + bedrock_api_input)
        print("\n \n The type of bedrock_input_variable is ", (type(bedrock_api_input)))

        # Invoke Bedrock model and print the result
        bedrock_result = invoke_bedrock_model(bedrock_api_input)

        if bedrock_result is not None:
            print("\nBedrock API Result:")
            print(bedrock_result)

            # Write the Bedrock result to S3
            write_result_to_s3(bedrock_result, s3_output_folder)

    # Stop the Spark session
    spark.stop()

