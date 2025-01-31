import os
os.environ['LANGCHAIN_TRACING_V2'] = 'your key'
os.environ['LANGCHAIN_ENDPOINT'] = 'your key'
os.environ['LANGCHAIN_API_KEY'] = 'your key'
os.environ['OPENAI_API_KEY'] = 'your key'
os.environ['TAVILY_API_KEY'] = 'your key'
os.environ["ANTHROPIC_API_KEY"] = 'your key'
os.environ["GOOGLE_API_KEY"] = 'your key'

from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import CharacterTextSplitter
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


from typing import Dict, TypedDict

from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
  """
  Represents the state of an agent in the conversation.
  """

  keys: Dict[str, any]

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tavily import TavilyClient, MissingAPIKeyError, InvalidAPIKeyError, UsageLimitExceededError

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

print("Defining nodes...")

def generate_queries(state):
  from langchain_core.documents import Document

  state_dict = state["keys"]
  hate_speech = state_dict["hate_speech"]

  template = """You are responding with counterspeech to the hate speech {question}
  Generate 3 precise queries you would research in order to generate a counterspeech for this hate speech.
  Include no preamble in your answer, only have the questions, and have each question only be separated by newlines.
  """

  query_template = ChatPromptTemplate.from_template(template)

  chain = query_template | llm | StrOutputParser()

  queries = chain.invoke({"question": hate_speech}).split("\n\n")

  return {"keys": {"hate_speech": hate_speech, "queries": queries}}

def retrieve(state):
  from langchain_core.documents import Document
  state_dict = state["keys"]
  hate_speech = state_dict["hate_speech"]
  queries = state_dict["queries"]

  retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) # k is how many docs are retrieved
  query_document_pairs = []

  for query in queries:
    # From Newsroom:
    retrieved_documents = retriever.get_relevant_documents(query)
    for document in retrieved_documents:
      query_document_pairs.append({"query": query, "document": document})
    # From Tavily:
    tavily_client = TavilyClient()
    response = tavily_client.search(query, max_results=1)
    for result in response["results"]:
      document = Document(
          page_content= result["content"],
          metadata={"title": result["title"],
                    "url": result["url"],
                    })
      query_document_pairs.append({"query": query, "document": document})

  return {"keys": {"hate_speech": hate_speech, "queries": queries, "query_document_pairs": query_document_pairs}}

def summarize_documents(state):
  from langchain_core.documents import Document
  state_dict = state["keys"]
  hate_speech = state_dict["hate_speech"]
  queries = state_dict["queries"]
  query_document_pairs = state_dict["query_document_pairs"]

  template = """Summarize this document in 2-3 sentences in a way that answers the query and retaining information relevant for countering the hate speech. Keep any relevant statistics, and sources.
  Query: \n {query} \n
  Hate speech: \n {hs} \n
  Document: \n {document} \n
  """

  prompt = ChatPromptTemplate.from_template(template)

  chain = prompt | llm | StrOutputParser()

  query_summarized_document_pairs = []

  for query_document_pair in query_document_pairs:
    document_summary = chain.invoke({"query":query_document_pair["query"],"document":query_document_pair["document"].page_content, "hs":hate_speech})
    query_summarized_document_pairs.append({"query": query_document_pair["query"], "document": document_summary, "metadata": query_document_pair["document"].metadata})

  return {"keys": {"hate_speech": hate_speech, "queries": queries, "query_document_pairs": query_document_pairs, "query_summarized_document_pairs": query_summarized_document_pairs}}



def grade_documents(state):
    from langchain_core.documents import Document
    state_dict = state["keys"]
    hate_speech = state_dict["hate_speech"]
    queries = state_dict["queries"]
    query_document_pairs = state_dict["query_document_pairs"]
    query_summarized_document_pairs = state_dict["query_summarized_document_pairs"]

    template = """You are a grader assessing the relevance of a retrieved document for use in constructing counterspeech to hate speech.
    Here is the retrieved document: \n\n {document} \n\n
    Here is the hate speech: \n\n {hs} \n\n
    Here is the query used to retrieve that document: \n\n {question} \n\n
    If the document is generally relevant to countering the hate speech, grade it as relevant.
    Grade it as irrelevant if it is generally irrelevant to countering the hate speech, or if it seems to agree with the hate speech.
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
    """

    prompt_template = ChatPromptTemplate.from_template(template)

    filtered_query_document_pairs = []

    for query_summarized_document_pair in query_summarized_document_pairs:
        query = query_summarized_document_pair["query"]
        document = query_summarized_document_pair["document"]

        chain = prompt_template | llm | StrOutputParser()
        score = chain.invoke({
            "document": document,
            "hs": hate_speech,
            "question": query
        }).strip()

        if score.lower() == "yes":
            print("Document is relevant!")
            filtered_query_document_pairs.append(query_summarized_document_pair)
        else:
            print("Document is irrelevant :(")

    return {
        "keys": {
            "hate_speech": hate_speech,
            "queries": queries,
            "query_document_pairs": query_document_pairs,
            "query_summarized_document_pairs": query_summarized_document_pairs,
            "filtered_query_document_pairs": filtered_query_document_pairs,

        }
    }


def generate_counterspeech(state):
  def format_documents(data):
    formatted_output = []
    for index, item in enumerate(data, start=1):
        document = item['document']
        metadata = item['metadata']
        
        formatted_string = f"{index}. {document}"
        
        if 'seq_num' not in metadata:
            source_info = ''
            if 'title' in metadata and 'url' in metadata:
                source_info = f"source: {{title: '{metadata['title']}', url: '{metadata['url']}'}}"
            elif 'source' in metadata:
                source_info = f"source: {metadata['source']}"
            formatted_string += f"\n{source_info}"
        
        formatted_output.append(formatted_string)
    
    return '\n\n'.join(formatted_output)
  
  def clean_response(response):
    cleaned = response.strip()
    
    # Remove specific prefixes that some LLMs generate
    if cleaned.startswith("Counterspeech:"):
        cleaned = cleaned.replace("Counterspeech:", "", 1).strip()
    elif cleaned.startswith("Counterspeech statement:"):
        cleaned = cleaned.replace("Counterspeech statement:", "", 1).strip()
    
    return cleaned
  

  from langchain_core.documents import Document
  state_dict = state["keys"]
  hate_speech = state_dict["hate_speech"]
  queries = state_dict["queries"]
  query_document_pairs = state_dict["query_document_pairs"]
  query_summarized_document_pairs = state_dict["query_summarized_document_pairs"]
  filtered_query_document_pairs = state_dict["filtered_query_document_pairs"]


  template = """Generate counterspeech to this hate speech, utilizing the following information fetched from various sources.
  Try and really utilize the information to build your narrative, and cite sources when possible: \n\n {context} \n\n

  Hate speech: \n\n {question} \n\n
  Keep your counterspeech to 4 sentences. Do not include any prolouge, additional commentary or explanation. Do not use markdown formatting. Return nothing but a single counterspeech statement.
  """

  prompt = ChatPromptTemplate.from_template(template)


  llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
  chain = prompt | llm | StrOutputParser()
  gpt_4o_counterspeech = chain.invoke({"context":format_documents(filtered_query_document_pairs),"question":hate_speech})


  return {"keys":
            {"hate_speech": hate_speech,
            "queries": queries,
            "query_document_pairs": query_document_pairs, 
            "filtered_query_document_pairs": filtered_query_document_pairs, 
            "query_summarized_document_pairs": query_summarized_document_pairs,
            "gpt_4o_counterspeech": gpt_4o_counterspeech,
            }}

from langgraph.graph import END, StateGraph
from langchain_core.documents import Document

pipeline = StateGraph(GraphState)

pipeline.add_node("generate_queries", generate_queries)
pipeline.add_node("retrieve", retrieve)
pipeline.add_node("summarize_documents", summarize_documents)
pipeline.add_node("grade_documents", grade_documents)
pipeline.add_node("generate_counterspeech", generate_counterspeech)

pipeline.set_entry_point("generate_queries")
pipeline.add_edge("generate_queries", "retrieve")
pipeline.add_edge("retrieve", "summarize_documents")
pipeline.add_edge("summarize_documents", "grade_documents")
pipeline.add_edge("grade_documents", "generate_counterspeech")

app = pipeline.compile()
print("Pipeline successfully built.")
import pandas as pd
import csv


csv_path = "/mt_conan_no_dupes.csv"
df = pd.read_csv(csv_path)
unique_hate_speech = df["HATE_SPEECH"].unique()

output_csv = "/final_data.csv"

existing_hate_speech = []
existing_entry_count = 0

# Load the hate speech entries that are already in the output file
try:
    existing_df = pd.read_csv(output_csv)
    existing_hate_speech = existing_df["hate_speech"].unique()
    # Check how many entries the file already has
    existing_entry_count = len(existing_df)
except FileNotFoundError:
    existing_hate_speech = []
    existing_entry_count = 0


with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Write the header if the file is newly created
    if len(existing_hate_speech) == 0:
        writer.writerow([
            "hate_speech",
            "counter_narrative",
            "query_document_pairs",
            "query_summarized_document_pairs",
            "filtered_query_document_pairs",
            "gpt_4o_counterspeech",
        ])
    
    for hate_speech in unique_hate_speech:
        print("Processing hate speech.")
        if hate_speech in existing_hate_speech:
            print(f"Skipping duplicate: {hate_speech}")
            continue  # Skip if the hate speech is already processed

        # Get the corresponding counter narrative
        counter_narrative = df.loc[df["HATE_SPEECH"] == hate_speech, "COUNTER_NARRATIVE"].values
        counter_narrative = counter_narrative[0] if len(counter_narrative) > 0 else ""

        inputs = {"keys": {"hate_speech": hate_speech}}
        
        # Running the pipeline:
        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "generate_counterspeech":
                    results = value["keys"]
                    writer.writerow([
                        results.get("hate_speech", ""),
                        counter_narrative,
                        str(results.get("query_document_pairs", [])),  
                        str(results.get("query_summarized_document_pairs", [])),  
                        str(results.get("filtered_query_document_pairs", [])),  
                        results.get("gpt_4o_counterspeech", ""),
                    ])
                    file.flush()
                    existing_entry_count += 1
                    print(f"Successfully processed and written: {hate_speech}")

print(f"Pipeline run complete.")

