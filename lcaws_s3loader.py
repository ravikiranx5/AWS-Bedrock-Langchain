from langchain_community.document_loaders import S3FileLoader

loader = S3FileLoader("ccdin-bucket","sample-f1.docx")
data = loader.load()
print(data)

#unstructured-all library downloaded