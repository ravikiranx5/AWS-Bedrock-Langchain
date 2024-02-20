from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate



inference_modifier = {
    # "maxTokenCount": 100,
    # "stopSequences": ["/"],
    "temperature": 0,
    # "topP":1
}


llm = Bedrock(
    model_id="amazon.titan-text-express-v1",
    region_name="us-east-1",
    model_kwargs=inference_modifier,
)

multi_var_prompt = PromptTemplate(
    input_variables=["customerServiceManager", "customerName", "feedbackFromCustomer"],
    template="""Create an apology email from the Service Manager {customerServiceManager} to {customerName} in response to the following feedback that was received from the customer: 
{feedbackFromCustomer}
""",
)

# Pass in values to the input variables
prompt = multi_var_prompt.format(
    customerServiceManager="Bob",
    customerName="Rav",
    feedbackFromCustomer="""Hello Bob,
     I am very disappointed with the recent experience I had when I called your customer support.
     I was expecting an immediate call back but it took three days for us to get a call back.
     The first suggestion to fix the problem was incorrect. Ultimately the problem was fixed after three days.
     We are very unhappy with the response provided and may consider taking our business elsewhere.
     """,
)



response = llm.invoke(prompt)

email = response[response.index("\n") + 1 :]

print(email)
