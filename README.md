# GoogleRAG - PDF Question Answering System

A powerful Retrieval-Augmented Generation (RAG) system that allows you to ask questions about your PDF documents using Google's Gemini AI.

## Features

- 📄 **Process Multiple PDFs** - Read and index all PDFs in a folder
- 🔍 **Semantic Search** - Find relevant content using vector embeddings
- 🤖 **Google Gemini Integration** - Get accurate answers using state-of-the-art AI
- 💬 **Interactive Q&A** - Ask questions in natural language
- 📊 **Source Attribution** - See which documents provided the information
- 💾 **Persistent Storage** - Only index documents once

## Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key ([Get one here](https://aistudio.google.com/))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/z1000biker/googlerag.git
cd googlerag
``` 

Install dependencies


```bash
pip install -r requirements.txt
``` 
Set up your PDFs

Create a folder named pdfs in the project directory

Add all your PDF files to this folder

Run the system

``` 
python gemini_rag.py

``` 
## Usage

Basic Usage

The system will automatically process all PDFs in the pdfs folder

Once indexing is complete, you can start asking questions

Type your questions and get answers based on your documents

Type quit to exit

## Example

Processing: ./pdfs/document1.pdf
Processing: ./pdfs/document2.pdf
Indexing 15 chunks...
Indexing complete!

Ready! Type 'quit' to exit.

Your question: What are the main features of the product?

## License:
 MIT
