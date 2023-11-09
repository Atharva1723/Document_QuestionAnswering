# AUTHOR : TECHMAHINDRA MAKERS LAB #


import os
from abc import ABC, abstractmethod
import pandas as pd
from django.http import HttpResponse
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import os

from pdfminer.high_level import extract_text
from langchain.vectorstores import Chroma
from fwk.settings import BOT_FOLDER, MODEL_PATH, KEYWORDS_PATH
import re

class WordEmbeddingModel(ABC):
    @abstractmethod
    def save_embeddings(self, botid, upload_file):
        pass

    @abstractmethod
    def generate_response(self, text):
        pass

class BertDocumentModel(WordEmbeddingModel):
    def __init__(self, botid):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.botid = botid
        bot_folder = os.path.join(BOT_FOLDER, botid)
        self.embedding_file_path = os.path.join(bot_folder, 'bert_para_embeddings')
        print(self.embedding_file_path)
        if not os.path.exists(self.embedding_file_path):
            os.makedirs(self.embedding_file_path)

    def save_embeddings(self, upload_file):
        create_embeddings(upload_file, self.embedding_file_path)
        return "Document embeddings are created"

    def generate_response(self, query):
        answer = get_answer(self.botid, query)
        return answer

def create_embeddings(pdf_folder, embeddings_path):
    data = []

    # Loop through all the files in the directory
    for index, filename in enumerate(os.listdir(pdf_folder)):
        # Check if the file is a PDF
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(pdf_folder, filename)

            # Load the PDF and extract its text
            text = extract_text(full_path)

            # Create a dictionary to store the document information
            document_info = {
                "document_id": index,  # Unique identifier for the document
                "document_name": filename,  # Name of the document
                "page_content": text,
                "metadata": {
                    "source": full_path,
                    "page": len(data)  # Page number is the index of the document in 'all_data' list
                }
            }

            # Append the document information to the list
            data.append(document_info)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

    class DocumentData:
        def __init__(self, page_content, metadata, document_id, document_name):
            self.page_content = page_content
            self.metadata = metadata
            self.document_id = document_id
            self.document_name = document_name

    # Create a list to store all document data
    all_documents = []

    for document_info in data:
        page_content = document_info["page_content"]
        metadata = document_info["metadata"]
        document_id = document_info["document_id"]
        document_name = document_info["document_name"]

        document = DocumentData(page_content=page_content, metadata=metadata, document_id=document_id,
                                document_name=document_name)
        all_documents.append(document)

    # Split documents into chunks using the initialized text splitter
    docs = text_splitter.split_documents(all_documents)

    # Create the open-source embedding function
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Load documents into Chroma
    db = Chroma.from_documents(docs, embeddings, persist_directory=embeddings_path)
    db.persist()

    return HttpResponse("done")

def check_valid_question_1(question):
    keywords_df = pd.read_excel(KEYWORDS_PATH)
    keywords_dict = dict()

    for i in range(len(keywords_df)):
        resort_number = keywords_df["Resort Number"][i]
        temp = keywords_df["Keywords"][i].strip().split(',')
        keywords = set(temp)
        keywords_dict[resort_number] = keywords
    
    for resort_name in keywords_dict:
        if has_keyword(question, keywords_dict[resort_name]):
            return resort_name

    return None

def has_keyword(sentence, keywords):
    pattern = '|'.join(re.escape(keyword) for keyword in keywords)
    regex = re.compile(pattern, re.IGNORECASE)
    return bool(regex.search(sentence))

def docsearch_with_name_similarity(botid, query, relevant_document_names):
    bot_folder = os.path.join(BOT_FOLDER, botid)
    persist_directory = os.path.join(bot_folder, 'bert_para_embeddings')
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    query_with_names = query + " " + " ".join(relevant_document_names)
    print(query_with_names)
    return db.similarity_search(query_with_names)

def get_answer(botid, query):
    # Verbose is required to pass to the callback manager
    model_path = MODEL_PATH  # Path of the downloaded module
    n_gpu_layers = 40
    n_batch = 512

    # Load LlamaCpp model
    llm = LlamaCpp(
        model_path=model_path,
        max_tokens=256,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        n_ctx=1024,
        verbose=True,
    )

    chain = load_qa_chain(llm, chain_type="stuff")
    resort_number = check_valid_question_1(query)

    resort_number_to_name = {
        1: "Club Mahindra Kanha",
        2: "Club Mahindra Munnar",
        3: "Club Mahindra Shimla",
        4: "Club Mahindra Udaipur",
        5: "Club Mahindra Varca",
    }

    if resort_number:
        CURRENT_RESORT_NAME = resort_number_to_name[resort_number]
        docs = docsearch_with_name_similarity(botid, query, CURRENT_RESORT_NAME)
        ans = str(chain.run(input_documents=docs, question=query))
        return ans
    else:
        if not resort_number:
            ans = "Sorry, I did not understand that. Please mention the name of the resort in the question.\n"
            return ans
