import boto3
import json
import pandas as pd
import pyarrow.parquet as pq

def read_parquet_from_s3(bucket, path):
    # Read Parquet file from S3
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=path)
    parquet_object = pq.ParquetFile(response['Body'])
    
    # Convert Parquet data to a Pandas DataFrame
    parquet_df = parquet_object.read().to_pandas()
    
      # Concatenate all values into a single string
    pq_final_data = ' '.join(parquet_df.values.flatten().astype(str))
    
    return pq_final_data




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

    # Invoke the Bedrock model
    response = bedrock_runtime.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )

    # Parse the response
    response_body = json.loads(response["body"].read())

    # Return the output text
    return response_body.get("results")[0].get("outputText")

def lambda_handler(event, context):
    # S3 bucket and path where Parquet file is stored
    s3_bucket = "ccdin-bucket"
    s3_path = "s3://ccdin-bucket/call_transcript_sample.parquet"

    try:
        # Read Parquet file from S3 and generate prompt
        pq_data = read_parquet_from_s3(s3_bucket, s3_path)
        
         #  prompt with additional text if needed
        prompt_input_question = "summarize the  below  chat transcript "
        prompt_full = f"{pq_data} {prompt_input_question}"
        

        # Invoke Bedrock model with the prompt
        output_text = invoke_bedrock_model(prompt_full)

        # Print and return the output text
        print(output_text)
        return {"statusCode": 200, "body": output_text}
    except Exception as e:
        # Handle exceptions and return an error response
        error_message = str(e)
        print(f"Error: {error_message}")
        return {"statusCode": 500, "body": f"Error: {error_message}"}