from flask import jsonify, request
from flask import Flask, request
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import openai
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
import os
from langchain_google_vertexai import ChatVertexAI, VertexAI
import vertexai
from langchain.schema import ChatMessage
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

# credential_path = r"D:\download\refreshing-code-426111-t6-f6d56102511b.json"
# if not os.path.exists(credential_path):
#     raise Exception(f"Credentials file not found at {credential_path}")
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
# PROJECT_ID = "refreshing-code-426111-t6"  # @param {type:"string"}
# REGION = "asia-east2"  # @param {type:"string"}
# vertexai.init(project=PROJECT_ID, location=REGION)
# llm = ChatVertexAI(
#     model="gemini-pro",
#     temperature=0,
#     max_tokens=None,
#     max_retries=6,
#     stop=None,
# )
app = Flask(__name__)
folder_path = "db"

llm =  AzureChatOpenAI(
    api_key="d10679b382ab2c6a2e70edc11a28d1074ed13b6a6b1dc757b6e073665a567822f3ac9f7e67a716231d372af56d701b040d1a5e9f910df530bdda0ece717a88b174a9395b3199e54bddc995763d2ca9549b3fc9657ac298f38328de45d2e33957",
    api_version="2023-05-15",
    azure_endpoint="https://cityucsopenai.azurewebsites.net/api/"
)
deployment_name = "cs4514-gpt-4-32k"

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] 
    "You are a friendly Cupid AI assistant, here to help users navigate our dating app - Cupid with ease and find your perfect match. 
   ",
    [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

@app.route("/askAsistant", methods=["POST"])
def askAsistant():
    print("Post askAsistant called")
    json_content = request.json
    query = json_content.get("query")
    localFolderPath = folder_path
    localFolderPath = json_content.get("folderName")
    print(f"query: {query}")
    # Get the chat history from the request
    chat_history = json_content.get("history", [])
    print("Loading vector store")
    vector_store = Chroma(persist_directory=localFolderPath, embedding_function=embedding)
    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.1,
        },
    )
    document_chain = create_stuff_documents_chain(llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    input_data = {
        "input": query,
        "history": chat_history
    }

    result = chain.invoke(input_data)

    response_answer = {"answer": result["answer"]}
    return jsonify(response_answer), 201

@app.route("/askCupid", methods=["POST"])
def askCupid():
    print("Post askAsistant called")
    json_content = request.json
    query = json_content.get("query") 

    chat_history = json_content.get("history", [])

    # messages = [
    #     {"type": message["role"], "content": message["content"]}
    #     for message in chat_history
    # ]

   
    result = llm.invoke(chat_history)

    response_answer = {"answer": result.content}
    return jsonify(response_answer), 201


@app.route("/text", methods=["POST"]) # handle upload
def textPost():
    localFolderPath = folder_path
    localFolderPath = request.form['folderName']

    files = request.files.getlist("files")
    allFileNames = []
    if not os.path.exists(localFolderPath):
        # Create the directory if it doesn't exist
        os.makedirs(localFolderPath)
    for file in files:
        # Process each file individually
        filename = file.filename
        allFileNames.append(filename)
        file.save(os.path.join(localFolderPath, filename))

        with open(os.path.join(localFolderPath, filename), 'r') as f:
            text = f.read()
        docs = [Document(page_content=text)]
        chunks = text_splitter.split_documents(docs)
        print(f"chunks len={len(chunks)}")
        vector_store = Chroma.from_documents(
            documents=chunks, embedding=embedding, persist_directory=folder_path
        )
        vector_store.persist()
    response = {
        "status": "Successfully Uploaded",
        "filename": allFileNames,
    }
    return response

def start_app():
    app.run(debug=True)
if(__name__ =="__main__"):
    start_app()