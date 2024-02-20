from langchain_community.llms import Bedrock

inference_modifiers = {"temperature": 0.3, "maxTokenCount": 512}

llm = Bedrock(
     #client = boto3_bedrock,
     model_id="amazon.titan-text-express-v1",
     model_kwargs =inference_modifiers
)

response = llm.invoke("What is the largest city in Vermont?")
print(response) 