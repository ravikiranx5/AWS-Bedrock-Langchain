import boto3
import json
import botocore

bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
prompt = "Hi ?"
configs = {"inputText": prompt}
body = json.dumps(configs)
modelId = "amazon.titan-embed-text-v1"
accept = "*/*"
contentType = "application/json"


try:
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    embedding = response_body.get("embedding")
    print(
        f"The embedding vector has {len(embedding)} values\n{embedding[0:3]+['...']+embedding[-3:]}"
    )

except botocore.exceptions.ClientError as error:
    if error.response["Error"]["Code"] == "AccessDeniedException":
        print(
            f"\x1b[41m{error.response['Error']['Message']}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n"
        )

    else:
        raise error
