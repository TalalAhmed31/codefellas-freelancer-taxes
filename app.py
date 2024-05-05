import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GEMINI_API_KEY"]=os.getenv("GEMINI_API_KEY")
import streamlit as st
from pypdf import PdfReader

def load_pdf(file_path):
    """
    Reads the text content from a PDF file and returns it as a single string.

    Parameters:
    - file_path (str): The file path to the PDF file.

    Returns:
    - str: The concatenated text content of all pages in the PDF.
    """
    # Logic to read pdf
    reader = PdfReader(file_path)

    # Loop over each page and store it in a variable
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text

# replace the path with your file path
pdf_text = load_pdf(file_path="./finalfile.pdf")


import re
def split_text(text: str):
    """
    Splits a text string into a list of non-empty substrings based on the specified pattern.
    The "\n \n" pattern will split the document para by para
    Parameters:
    - text (str): The input text to be split.

    Returns:
    - List[str]: A list containing non-empty substrings obtained by splitting the input text.

    """
    split_text = re.split('\n', text)
    return [i for i in split_text if i != ""]

# chunked_text = split_text(text=pdf_text)
# len(chunked_text)



import google.generativeai as genai
import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from chromadb import EmbeddingFunction, Embeddings

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function using the Gemini AI API for document retrieval.

    This class extends the EmbeddingFunction class and implements the __call__ method
    to generate embeddings for a given set of documents using the Gemini AI API.

    Parameters:
    - input (list[str]): A list of documents to be embedded.

    Returns:
    - Embeddings: Embeddings generated for the input documents.
    """
    def __call__(self, input: list[str]) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]

import chromadb

# db,name =create_chroma_db(documents=chunked_text, 
#                           path="./", #replace with your path
#                           name="rag_experiment2")


def load_chroma_collection(path, name):
    """
    Loads an existing Chroma collection from the specified path with the given name.

    Parameters:
    - path (str): The path where the Chroma database is stored.
    - name (str): The name of the collection within the Chroma database.

    Returns:
    - chromadb.Collection: The loaded Chroma Collection.
    """
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    return db

db=load_chroma_collection(path="./", name="rag_experiment2")
# db


def get_relevant_passage(query, db, n_results):
  passage = db.query(query_texts=[query], n_results=n_results)['documents'][0]
  return passage

# #Example usage
# relevant_text = get_relevant_passage(query="how to start the process",db=db,n_results=3)
# relevant_text


def make_rag_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = ("""
  PASSAGE: '{relevant_passage}'
  Above is the paragraph about tax for freelancers. User will possibly ask you to check how much tax he or she has to pay, calculate it by getting percentage from passage and make it possible to tell the user tax amount. Person can also seek guidance regarding filling filer form. keep the response long. Do not give answer outside of the context.
  QUESTION: '{query}'

  ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt


import google.generativeai as genai
def generate_answer(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.0-pro-latest')
    answer = model.generate_content(prompt)
    return answer.text


def generate_answers(db,query):
    #retrieve top 3 relevant text chunks
    relevant_text = get_relevant_passage(query,db,n_results=3)
    prompt = make_rag_prompt(query, 
                             relevant_passage="".join(relevant_text)) # joining the relevant chunks to create a single passage
    answer = generate_answer(prompt)

    return answer


# db=load_chroma_collection(path="./", #replace with path of your persistent directory
#                           name="rag_experiment2") #replace with the collection name
# query = "i am a freelancer on upwork earning 10k dollars annually, how much tax would i pay annually? and what are the steps to file tax in pakistan"
# answer = generate_answers(db,query=query)
# print(answer)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Enable chat option if at least one file is connected
if True:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # response = f"Echo: {prompt}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # print(st.session_state.qa_chain)
            response_bot = generate_answers(db,query=prompt)
            # print metadata source of response
            print(response_bot)
            st.markdown(response_bot)
            st.session_state.chat_history.append((prompt, response_bot))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_bot})
else:
    st.warning("Connect at least one file to enable the chat option.")