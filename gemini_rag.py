import os
import glob
from PyPDF2 import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

class GeminiPDFRAG:
    def __init__(self, pdf_folder_path, google_api_key):
        self.pdf_folder = pdf_folder_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="pdf_documents")
        
        # Initialize the new Google AI client
        self.client = genai.Client(api_key=google_api_key)
        
    def load_and_chunk_pdfs(self):
        """Load PDFs with improved text extraction"""
        pdf_files = glob.glob(os.path.join(self.pdf_folder, "*.pdf"))
        all_chunks = []
        chunk_metadata = []
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file}")
            try:
                reader = PdfReader(pdf_file)
                full_text = ""
                
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        # Clean up the text
                        page_text = ' '.join(page_text.split())
                        full_text += f"\n--- Page {page_num+1} ---\n{page_text}"
                
                if not full_text.strip():
                    print(f"  Warning: No text extracted from {pdf_file}")
                    continue
                
                # Improved chunking
                paragraphs = [p.strip() for p in full_text.split('\n') if p.strip()]
                
                current_chunk = ""
                for paragraph in paragraphs:
                    # If this is a page break, consider flushing the chunk
                    if paragraph.startswith("--- Page"):
                        if current_chunk and len(current_chunk) > 50:
                            all_chunks.append(current_chunk)
                            chunk_metadata.append({
                                "source": os.path.basename(pdf_file),
                                "chunk_id": len(all_chunks)
                            })
                            current_chunk = paragraph
                        else:
                            current_chunk = paragraph
                    else:
                        if len(current_chunk + " " + paragraph) < 1500:
                            if current_chunk:
                                current_chunk += " " + paragraph
                            else:
                                current_chunk = paragraph
                        else:
                            if current_chunk and len(current_chunk) > 50:
                                all_chunks.append(current_chunk)
                                chunk_metadata.append({
                                    "source": os.path.basename(pdf_file),
                                    "chunk_id": len(all_chunks)
                                })
                            current_chunk = paragraph
                
                # Don't forget the last chunk
                if current_chunk and len(current_chunk) > 50:
                    all_chunks.append(current_chunk)
                    chunk_metadata.append({
                        "source": os.path.basename(pdf_file),
                        "chunk_id": len(all_chunks)
                    })
                        
                print(f"  Extracted chunks from {os.path.basename(pdf_file)}")
                        
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
        
        return all_chunks, chunk_metadata
    
    def index_documents(self):
        """Create embeddings and store in ChromaDB"""
        chunks, metadata = self.load_and_chunk_pdfs()
        
        if not chunks:
            print("No documents found or no text extracted!")
            return False
        
        print(f"Indexing {len(chunks)} chunks...")
        
        try:
            embeddings = self.embedding_model.encode(chunks).tolist()
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadata,
                ids=ids
            )
            print("✓ Indexing complete!")
            return True
        except Exception as e:
            print(f"✗ Indexing failed: {e}")
            return False
    
    def query_gemini(self, prompt, context):
        """Send query to Google Gemini with context using new SDK"""
        try:
            # Prepare the prompt for Gemini
            full_prompt = f"""Answer the following question based ONLY on the provided context. 
If the answer cannot be found in the context, say: "I cannot find this information in the provided documents."

CONTEXT:
{context}

QUESTION: {prompt}

ANSWER (based only on the context):"""
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",  # Using the flash model which is faster and cheaper
                contents=full_prompt
            )
            return response.text
        except Exception as e:
            return f"Error querying Gemini: {e}"
    
    def search_and_answer(self, question, n_results=3):
        """Search for relevant chunks and generate answer"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([question]).tolist()[0]
        
        # Search ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            if not results['documents'] or not results['documents'][0]:
                return "No relevant documents found."
            
            # Combine retrieved contexts
            context = "\n\n".join(results['documents'][0])
            
            print("Retrieved context:")
            print("-" * 60)
            for i, doc in enumerate(results['documents'][0]):
                source = results['metadatas'][0][i]['source']
                preview = doc[:200] + "..." if len(doc) > 200 else doc
                print(f"{i+1}. [From: {source}]\n   {preview}\n")
            print("-" * 60)
            
            # Generate answer using Gemini
            answer = self.query_gemini(question, context)
            return answer
            
        except Exception as e:
            return f"Search error: {e}"
    
    def get_collection_info(self):
        """Get info about stored documents"""
        try:
            return self.collection.count()
        except:
            return 0

def main():
    print("PDF RAG System with Google Gemini (New SDK)")
    print("=" * 50)
    
    # Your Google API Key
    GOOGLE_API_KEY = "YOUR GOOGLE API KEY"
    
    # Initialize RAG system
    rag = GeminiPDFRAG("./pdfs", GOOGLE_API_KEY)
    
    # Check if we need to index documents
    if rag.get_collection_info() == 0:
        print("No existing index found. Building document index...")
        success = rag.index_documents()
        if not success:
            print("Failed to index documents. Exiting.")
            return
    else:
        print(f"Found existing index with {rag.get_collection_info()} chunks.")
        reindex = input("Re-index documents? (y/n): ").strip().lower()
        if reindex == 'y':
            rag.index_documents()
    
    # Interactive Q&A
    print("\nReady for questions! Type 'quit' to exit.")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if question:
            answer = rag.search_and_answer(question)
            print(f"\nAnswer: {answer}")

if __name__ == "__main__":

    main()
