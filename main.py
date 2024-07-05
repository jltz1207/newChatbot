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
import vertexai
from langchain.schema import ChatMessage
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
import logging
from InterestExtract import extract_interest
from PersonalExtract import extract_personality
from AiMatch import update_user_keywords, update_user_details, find_best_matches
from faceReg import find_similar_face
import base64
import io
import face_recognition
from PIL import Image

app = Flask(__name__)
#logging.basicConfig(level=logging.DEBUG)
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

@app.route("/askCupid", methods=["POST"]) #only change the keywords
def askCupid():
    print("Post askCupid called")
    json_content = request.json
    chat_history = json_content.get("history", [])
    categoryId = json_content.get("categoryId")
    userId = json_content.get("userId")
    user_objects = [obj for obj in chat_history if obj.get("role") == "user"]
    
    response_answer = {}
    if user_objects and categoryId is not None:
        last_user_object = user_objects[-1]
        user_query = last_user_object.get("content", "")
        if categoryId == 1:
            keywords = extract_interest(user_query)
        elif categoryId ==2 or categoryId ==3:
            keywords = extract_personality(user_query)
        else:
            print("categoryId is out of range")
        update_user_keywords(userId, categoryId, keywords)
        response_answer["keywords"] = keywords
    elif categoryId is None:
        print("No categoryId found ")
    else: 
        print("No user object found in chat_history")

    result = llm.invoke(chat_history)
    response_answer["answer"] =  result.content
    return jsonify(response_answer), 201

@app.route("/updateDetails", methods=["POST"]) #only change the keywords
def updateDetails():
    print("Post /updateDetails called")
    json_content = request.json
    userId = json_content.get("userId")
    userDetails = json_content.get("userDetails")
    update_user_details(userId, userDetails)
    response_answer = {"response":"Updated successfully"}
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


@app.route("/genAiResponse", methods=["POST"])
def genAiResponse():
    print("Post genAiResponse called")
    json_content = request.json
    prompt = json_content.get("prompt") 

    messages = [{
    "role" : "system",
    "content":prompt
    }]
   
   
    result = llm.invoke(messages)

    response_answer = {"answer": result.content}
    return jsonify(response_answer), 201


@app.route("/genAiBio", methods=["POST"])
def genAiBio():
    print("Post /genAiBio called")
    json_content = request.json
    history = json_content.get("history") 
    result = llm.invoke(history)
    response_answer = {"answer": result.content}
    return jsonify(response_answer), 201

@app.route("/getKeywords", methods=["POST"])
def getKeywords():
    print("Post getKeywords called")
    json_content = request.json
    user_query = json_content.get("passage")
    categoryId = json_content.get("categoryId")

    if categoryId == 1:
        keywords = extract_interest(user_query)
    elif categoryId ==2 or categoryId ==3:
        keywords = extract_personality(user_query)
    
    print("Extracted keywords:", keywords)
    response_answer = {"keywords":keywords }
    return jsonify(response_answer), 201

@app.route("/getMatches", methods=["POST"])
def getMatches():
    print("Post /getMatches called")
    json_content = request.json
    userId = json_content.get("userId")
    otherUserIds = json_content.get("otherUserIds")
    expectedMinAge = json_content.get("expectedMinAge") #null
    expectedMaxAge = json_content.get("expectedMaxAge") #null
    weight = json_content.get("weight") #null
    matches = find_best_matches(userId,otherUserIds,weight, expectedMinAge, expectedMaxAge)

    response_answer = {"sortedId":matches}
    return jsonify(response_answer), 201

@app.route("/genFaceMatch", methods=["POST"]) #only change the keywords
def genFaceMatch():
    print("Post /genFaceMatch called")
    
    # getting input
    json_content = request.json
    user_photos_database = json_content.get("user_photos_database")
    newImage_base64 = json_content.get("newImage_base64")

    # find match
    matchedId = find_similar_face(newImage_base64,user_photos_database )
        
    
    print(matchedId)
    if matchedId == None: # no face is detected 
        return jsonify({'error': 'No faces detected in the provided image'}), 400
   
    response_answer = {"answer":matchedId}
    return jsonify(response_answer), 201



def start_app():
    app.run(debug=True)
if(__name__ =="__main__"):
    start_app()

    