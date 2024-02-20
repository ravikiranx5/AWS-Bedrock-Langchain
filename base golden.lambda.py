import boto3
import json

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
    # Sample meeting transcript prompt
    prompt = "Meeting transcript: Miguel: Hi Brant, I want to discuss the workstream  for our new product launch Brant: Sure Miguel, is there anything in particular you want to discuss? Miguel: Yes, I want to talk about how users enter into the product. Brant: Ok, in that case let me add in Namita. Namita: Hey everyone Brant: Hi Namita, Miguel wants to discuss how users enter into the product. Miguel: its too complicated and we should remove friction.  for example, why do I need to fill out additional forms?  I also find it difficult to find where to access the product when I first land on the landing page. Brant: I would also add that I think there are too many steps. Namita: Ok, I can work on the landing page to make the product more discoverable but brant can you work on the additonal forms? Brant: Yes but I would need to work with James from another team as he needs to unblock the sign up workflow.  Miguel can you document any other concerns so that I can discuss with James only once? Miguel: Sure."

    try:
        # Invoke Bedrock model with the prompt
        output_text = invoke_bedrock_model(prompt)

        # Print and return the output text
        print(output_text)
        return {"statusCode": 200, "body": output_text}
    except Exception as e:
        # Handle exceptions and return an error response
        error_message = str(e)
        print(f"Error: {error_message}")
        return {"statusCode": 500, "body": f"Error: {error_message}"}
