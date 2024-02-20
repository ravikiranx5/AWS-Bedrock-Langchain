import boto3
import json
import botocore


bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

prompt = """
Please provide a summary of the following call transcript. Do not add any information that is not mentioned in the text below.

<text>
Miguel: Hi Brant, I want to discuss the workstream  for our new product launch 
Brant: Sure Miguel, is there anything in particular you want to discuss? 
Miguel: Yes, I want to talk about how users enter into the product. 
Brant: Ok, in that case let me add in Namita. 
Namita: Hey everyone 
Brant: Hi Namita, Miguel wants to discuss how users enter into the product. 
Miguel: its too complicated and we should remove friction.  for example, why do I need to fill out additional forms?  I also find it difficult to find where to access the product when I first land on the landing page. 
Brant: I would also add that I think there are too many steps. 
Namita: Ok, I can work on the landing page to make the product more discoverable but brant can you work on the additonal forms? 
Brant: Yes but I would need to work with James from another team as he needs to unblock the sign up workflow.  Miguel can you document any other concerns so that I can discuss with James only once? 
Miguel: Sure.
</text>

"""

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
