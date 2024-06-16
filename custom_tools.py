from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st

# Tool to search the internet
@tool
def search_duckduckgo(query: str) -> str:
    """ Run internet search"""
    search = DuckDuckGoSearchResults()
    results = search.run(query)
    print(f"There was a duck duck go search for {query}")
    return results

# Tool to query the pinecone bible index
def search_bible(query:str) -> str:
    """ Search the bible for relevant passages for the user query. Returns passages for you to read and answer the user question. """
    # Get Open AI key from secrets.toml
    openai_api_key = st.secrets["openai_api_key"]

    # Create an OpenAIEmbeddings instance
    model_name = 'text-embedding-ada-002'
    embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

    # Embed your query
    query_vector = embeddings.embed_query(query)

    # Initialize the Pinecone client
    pc = Pinecone(api_key=st.secrets["pinecone_api_key"])

    # Connect to your index
    index_name = "full-bible-index"
    index = pc.Index(index_name)

    # Set k, this is the number of documents we want to return
    k = 3

    # Query the index using the query vector
    query_response = index.query(
        vector=query_vector,
        top_k=k,
        include_values=True,
        include_metadata=True
    )
    
    # Collect texts from the query response to use to interpret the data
    texts = []
    for i in range(k):
        text = query_response['matches'][i]['metadata']['text']
        texts.append(text)
    
    return texts