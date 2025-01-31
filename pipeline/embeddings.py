from langchain_community.document_loaders import JSONLoader
import uuid
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

minilm = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

documents = ['/newsroom/release/train.jsonl', '/newsroom/release/test.jsonl', '/newsroom/release/dev.jsonl']

vectorstore = Chroma(
    collection_name="example_collection",
    embedding_function=minilm, 
    persist_directory="/vectorstores/newsroom-vecstore",  
)

for document in documents:
  loader = JSONLoader(
      file_path=document,
      jq_schema='.text',
      text_content=False,
      json_lines=True)

  raw_documents = loader.load()
  batch_size = 3000 # The maximum batch size allowed is 41666
  text_splitter = CharacterTextSplitter(separator = "\n")

  parsed_documents = text_splitter.split_documents(raw_documents)

  uuids = [str(uuid.uuid4()) for _ in parsed_documents]

  print(f"Sample parsed document: {parsed_documents[6]}")

  print(f"This document {document} has {len(parsed_documents)} documents.")
  for i in range(0, len(parsed_documents), batch_size):
    print(f"{(i*100.0/len(parsed_documents)):.2f}% done with processing {document}.")
    print("Making batch documents..")
    batch_documents = parsed_documents[i:i + batch_size]
    print("Making batch UUIDs..")
    batch_uuids = uuids[i:i + batch_size]
    print("Adding to vectorstore..")
    vectorstore.add_documents(documents=batch_documents, ids=batch_uuids)
print("Finished building vectorstore.") 