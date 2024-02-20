import os
import boto3
import botocore

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch

index_name = os.environ["AOSS_INDEX_NAME"]
endpoint = os.environ["AOSS_COLLECTION_ENDPOINT"]

embeddings = BedrockEmbeddings( region_name="us-east-1",
     model_id="amazon.titan-embed-text-v1")

vector_store = OpenSearchVectorSearch(
          index_name=index_name,
          embedding_function=embeddings,
          opensearch_url=endpoint,
          http_auth=get_aws4_auth(),
          use_ssl=True,
          verify_certs=True,
          connection_class=RequestsHttpConnection,
     )
retriever = vector_store.as_retriever()














'''' RAG approach
You can convert the company-specific data, such as documents, into embeddings using the text embeddings model (described in the previous section).
You can then store the embeddings in a vector database.
You can extract the relevant documents based on the user request from the vector database.
You can then pass them to the LLM as context for an accurate response.'''