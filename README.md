# Document Question Answering with Langchain and Llama2

![Project Logo](logo.png)

This project demonstrates how to perform document question answering using the Langchain and Llama2 libraries. It allows you to extract information from PDF documents, split them into chunks, and answer questions based on the content of those documents.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before running this project, you need to have the following prerequisites installed:

- Python 3.9 or greater
- Langchain library(langchain==0.0.300)
- Llama2 library(llama-2-13.bin)
- Other necessary dependencies (listed in requirements.txt)

## Project Structure

The project is organized as follows:

- `document_qa.py`: The main script for processing documents and answering questions.
- `pdf_files_directory`: Directory containing the PDF files.
- `data`: A list to store the extracted data from each PDF.
- `DocumentData`: A class to represent document information.
- `text_splitter`: A text splitter for dividing documents into smaller chunks.
- `all_documents`: A list to store all document instances.
- `embeddings`: Sentence embeddings using Hugging Face models.
- `db`: Chroma vector store for document embeddings.
- `docsearch_with_name_similarity`: Function for searching documents based on question and document names.
- `llm`: LlamaCpp model for question answering.
- `chain`: Question answering chain using LlamaCpp.

This structure helps organize the components of the project and provides a clear overview of the main files and functionalities.



