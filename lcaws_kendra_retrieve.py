from langchain_community import AmazonKendraRetriever
from langchain_community import ConversationalRetrievalChain
from langchain_community import PromptTemplate
from langchain_community import Bedrock


llm = Bedrock(
     credentials_profile_name="ravi1",
     region_name = "us-east1",
     model_kwargs={"max_tokens_to_sample":300,"temperature":1,"top_k":250,"top_p":0.999},
     model_id="amazon.titan-text-express-v1"
)

retriever = AmazonKendraRetriever(index_id="96408c1f-9f88-4450-a60b-995f3c63d94f",top_k=5,region_name="us-east-1")

prompt_template = """ Human: This is a friendly conversation between a human and an AI. 
     The AI is talkative and provides specific details from its context but limits it to 240 tokens.
     If the AI does not know the answer to a question, it truthfully says it does not know.

     Assistant: OK, got it, I'll be a talkative truthful AI assistant.

     Human: Here are a few documents in <documents> tags:
     <documents>
     {context}
</documents>
     Based on the above documents, provide a detailed answer for, {question} 
     Answer "do not know" if not present in the document. 

      Assistant:
     """

PROMPT = PromptTemplate(
     template=prompt_template, input_variables=["context", "question"]
     )

response = ConversationalRetrievalChain.from_llm(
     llm=llm, 
     retriever=retriever, 
     return_source_documents=True, 
     combine_docs_chain_kwargs={"prompt":PROMPT},
     verbose=True)