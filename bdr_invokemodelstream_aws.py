import boto3
import json
import botocore


bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

prompt = "Write an essay for living on Mercury using 3 sentences."

configs = {
    "inputText": prompt,
    "textGenerationConfig": {
        # "maxTokenCount": 100,
        # "stopSequences": [],
        "temperature": 0,
        # "topP":1
    },
}
output = []


body = json.dumps(configs)
accept = "application/json"
contentType = "application/json"
modelId = "amazon.titan-text-express-v1"
# modelId = 'amazon.titan-tg1-large'


try:
    response = bedrock_runtime.invoke_model_with_response_stream(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    stream = response.get("body")

    i = 1
    if stream:
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                chunk_obj = json.loads(chunk.get("bytes").decode())
                text = chunk_obj["outputText"]
                output.append(text)
                print(f"\t\t\x1b[31m**Chunk {i}**\x1b[0m\n{text}\n")
                i += 1

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


print("\t\t\x1b[31m**COMPLETE OUTPUT**\x1b[0m\n")
complete_output = "".join(output)
print(complete_output)
