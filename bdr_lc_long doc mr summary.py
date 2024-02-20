from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain



inference_modifier = {
    "maxTokenCount": 4096,
    "stopSequences": [],
    "temperature": 0,
    "topP": 1,
}


llm = Bedrock(
    model_id="amazon.titan-text-express-v1",
    region_name="us-east-1",
    model_kwargs=inference_modifier,
)
# input file
input_file = "C:/wsletter.txt"
# open file to read
with open(input_file, encoding='utf8') as file:
    letter = file.read()

llm.get_num_tokens(letter)


# split file into small chunks
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"], chunk_size=4000, chunk_overlap=100
)

# splitting and recalculating toaken size for each chnunk
docs = text_splitter.create_documents([letter])
num_docs = len(docs)

num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)

print(
    f"we have {num_docs} documents split  and  each part  has {num_tokens_first_doc} tokens"
)

# entire summary by merging into 1 single,  after map reduce processS
summary_chain = load_summarize_chain(
    llm=llm, chain_type="map_reduce", verbose=False
)


output = ""
try:
    output = summary_chain.invoke(docs)

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


print(output)




'''


The text is too long to fit in the prompt, 
RecursiveCharacterTextSplitter in LangChain supports splitting long text into chunks
recursively until size of each chunk becomes smaller than chunk_size.
    

approx 6,000 characters per chunk, we can get summaries for each portion separately. The number of tokens, or word pieces,
 in a chunk depends on the text.

Summarizing chunks and combining them
Assuming that the number of tokens is consistent in the other docs we should be good to go.
 Let's use LangChain's load_summarize_chain to summarize the text.
   load_summarize_chain provides three ways of summarization: 
   1)stuff, 
   2)map_reduce, and
   3) refine.

stuff puts all the chunks into one prompt. Thus, this would hit the maximum limit of tokens

map_reduce summarizes each chunk, combines the summary, and summarizes the combined summary. 
If the combined summary is too large, it would raise error.

refine summarizes the first chunk, and then summarizes the second chunk with the first summary.
 The same process repeats until all chunks are summarized.


map_reduce and refine invoke LLM multiple times and takes time for obtaining final summary.

 map_reduce was used in above example.

'''