
from langchain.llms.bedrock import Bedrock



inference_modifier = {
     # "maxTokenCount": 100,
     #"stopSequences": ["/"],
     "temperature": 0,
     # "topP":1
}



llm = Bedrock(model_id="amazon.titan-text-express-v1",region_name="us-east-1",model_kwargs=inference_modifier)

response = llm.invoke("Write an email from Bob, Customer Service Manager,to the customer John Doe that provided negative feedback on the service provided by our customer support engineer" )

print(response)


