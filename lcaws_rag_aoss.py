from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.load.dump import dumps
from urllib.request import urlretrieve
import os

import numpy as np
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader


inference_modifier = {
    # "maxTokenCount": 100,
    # "stopSequences": ["/"],
    "temperature": 0,
    # "topP":1
}

# -  titan embed model
target_llm = Bedrock(
    model_id="amazon.titan-embed-text-v1",
    region_name="us-east-1",
    model_kwargs=inference_modifier,
)
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    service_name="bedrock-runtime", 
    region_name="us-east-1"
)


os.makedirs("data", exist_ok=True)
files = [
    "https://www.irs.gov/pub/irs-pdf/p1544.pdf",
    "https://www.irs.gov/pub/irs-pdf/p15.pdf",
    "https://www.irs.gov/pub/irs-pdf/p1212.pdf",
]
for url in files:
    file_path = os.path.join("data", url.rpartition("/")[2])
    urlretrieve(url, file_path)


loader = PyPDFDirectoryLoader("./data/")

documents = loader.load()
# Character split
text_splitter = RecursiveCharacterTextSplitter(
    # Set a chunk size,
    chunk_size=2000,
    chunk_overlap=200,
)
docs = text_splitter.split_documents(documents)


avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents]) // len(documents))
avg_char_count_pre = avg_doc_length(documents)
avg_char_count_post = avg_doc_length(docs)
print(
    f"Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters."
)
print(
    f"After the split we have {len(docs)} documents more than the original {len(documents)}."
)
print(
    f"Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters."
)


try:
    sample_embedding = np.array(bedrock_embeddings.embed_query(docs[0].page_content))
    modelId = bedrock_embeddings.model_id
    print("Embedding model Id :", modelId)
    print("Sample embedding of a document chunk: ", sample_embedding)
    print("Size of the embedding: ", sample_embedding.shape)

except ValueError as error:
    if "AccessDeniedException" in str(error):
        print(
            f"\x1b[41m{error}\
        \nTo troubeshoot this issue please refer to the following resources.\
         \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
         \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n"
        )

        class StopExecution(ValueError):
            def _render_traceback_(self):
                pass

        raise StopExecution
    else:
        raise error
