import boto3
import json
from pyspark.sql import SparkSession


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
        print(f"Error invoking Bedrock model: {e}")
        return None


def read_parquet_files(spark, s3_paths):
    df_list = []
    for s3_path in s3_paths:
        df = spark.read.parquet(s3_path)
        df_list.append(df)
    return df_list


def concatenate_transcripts(df_list, transcript_column):
    transcript_values = []
    for df in df_list:
        transcript_values.extend(df.select(transcript_column).collect())
    return " ".join([row[0] for row in transcript_values])


def write_result_to_s3(result_text, s3_output_path):
    # Create an S3 client
    s3_client = boto3.client("s3", region_name="us-east-1")

    # Specify the S3 path for storing the result
    s3_key = f"{s3_output_path}/bedrock_output_stestglu.txt"

    try:
        # Upload the result to S3
        s3_client.put_object(Body=result_text, Bucket=s3_output_bucket, Key=s3_key)
        print(f"Result uploaded to S3: {s3_key}")

    except Exception as e:
        print(f"Error uploading result to S3: {e}")


if __name__ == "__main__":
    # Initialize SparkSession
    spark = SparkSession.builder.appName("ReadParquetFiles").getOrCreate()

    # Define S3 folder locations
    # s3_input_folders = ["s3://ccdout-bucket/in_bdr_response/"]
    s3_input_folders = ["s3a://ccdout-bucket/in_call_transcripts/"]

    # Read Parquet files into DataFrame
    dfs = read_parquet_files(spark, s3_input_folders)

    # Concatenate transcript values
    # transcript_column = "email"
    transcript_column = "Text"
    concatenated_transcripts = concatenate_transcripts(dfs, transcript_column)

    # Custom prompt for Bedrock API
    custom_prompt = (
        "Please summarize the following call transcript input in 4 lines \n \n"
    )

    # Join custom prompt and 'transcripts' string for Bedrock API input
    bedrock_api_input = f"{custom_prompt} {concatenated_transcripts}"

    # Invoke Bedrock model and print the result
    bedrock_result = invoke_bedrock_model(bedrock_api_input)

    if bedrock_result is not None:
        print("\nBedrock API Result:")
        print(bedrock_result)

        # Write the Bedrock result to S3
        write_result_to_s3(bedrock_result, s3_output_folder)

    # Stop SparkSession
    spark.stop()



