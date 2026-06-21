import os
import io
import json
import threading
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

FAISS_PATH = "faiss_store"
_embeddings = None
_lock = threading.Lock()
_query_cache = {}

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 128, "normalize_embeddings": True, "show_progress_bar": False}
        )
    return _embeddings

class RAGSystem:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=2048
        )
        self.vectorstore = None
        self.documents = {}
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        self._load_faiss()

    def _load_faiss(self):
        try:
            if os.path.exists(FAISS_PATH):
                self.vectorstore = FAISS.load_local(FAISS_PATH, self.embeddings, allow_dangerous_deserialization=True)
                print("FAISS loaded from disk")
        except Exception as e:
            print(f"FAISS load error: {e}")

    def _save_faiss(self):
        try:
            if self.vectorstore:
                self.vectorstore.save_local(FAISS_PATH)
        except Exception as e:
            print(f"FAISS save error: {e}")

    def process_pdf(self, contents, filename):
        try:
            pdf_reader = PdfReader(io.BytesIO(contents))
            text = ""
            for page in pdf_reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
            if not text.strip():
                return {"error": "No text found in PDF"}
            chunks = self.text_splitter.split_text(text)
            chunks = chunks[:100]
            with _lock:
                if self.vectorstore is None:
                    self.vectorstore = FAISS.from_texts(chunks, self.embeddings, metadatas=[{"source": filename}] * len(chunks))
                else:
                    new_vs = FAISS.from_texts(chunks, self.embeddings, metadatas=[{"source": filename}] * len(chunks))
                    self.vectorstore.merge_from(new_vs)
                self._save_faiss()
            self.documents[filename] = len(chunks)
            return {"message": "PDF processed successfully", "chunks": len(chunks), "total_documents": len(self.documents)}
        except Exception as e:
            return {"error": str(e)}

    def query(self, question, history=None, mode="normal", language="English"):
        if self.vectorstore is None:
            return {"error": "No documents uploaded yet. Please upload a PDF first."}
        cache_key = question.strip().lower() + mode + language
        if cache_key in _query_cache:
            return _query_cache[cache_key]
        try:
            docs = self.vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(question)
            context = "\n\n".join([d.page_content for d in docs])
            sources = list(set([d.metadata.get("source", "Unknown") for d in docs]))
            history_text = ""
            if history:
                for m in history[-6:]:
                    history_text += m["role"] + ": " + m["content"] + "\n"
            if mode == "student":
                style = "Explain in very simple words, use analogies and examples. Imagine explaining to a 15 year old. Be friendly and encouraging."
            elif mode == "professor":
                style = "Provide a detailed, academic, technical analysis with proper terminology and depth."
            elif mode == "summary":
                style = "Provide a concise bullet-point summary of the most important information."
            else:
                style = "Be helpful, clear, and professional. Use bullet points when listing multiple things."
            prompt = "You are DocuMind AI — the world most advanced document intelligence assistant.\n\nStyle: " + style + "\nResponse language: " + language + "\n\nContext from documents:\n" + context + "\n\nConversation history:\n" + history_text + "\nHuman: " + question + "\nDocuMind AI:"
            response = self.llm.invoke(prompt)
            answer = response.content
            confidence = min(95, max(45, len(docs) * 20 + (10 if question.lower() in context.lower() else 0)))
            result = {"answer": answer, "sources": sources, "confidence": confidence, "mode": mode}
            if len(_query_cache) < 100:
                _query_cache[cache_key] = result
            return result
        except Exception as e:
            return {"error": str(e)}

    def compare_documents(self, question, doc1, doc2):
        if self.vectorstore is None:
            return {"error": "No documents uploaded"}
        try:
            docs = self.vectorstore.as_retriever(search_kwargs={"k": 6}).invoke(question)
            doc1_context = "\n".join([d.page_content for d in docs if d.metadata.get("source") == doc1])
            doc2_context = "\n".join([d.page_content for d in docs if d.metadata.get("source") == doc2])
            prompt = "You are DocuMind AI. Compare these two documents.\n\nQuestion: " + question + "\n\nDocument 1 (" + doc1 + "):\n" + (doc1_context or "No relevant content found") + "\n\nDocument 2 (" + doc2 + "):\n" + (doc2_context or "No relevant content found") + "\n\nProvide structured comparison with similarities, differences, and recommendation.\n\nDocuMind AI:"
            response = self.llm.invoke(prompt)
            return {"answer": response.content, "sources": [doc1, doc2], "confidence": 85}
        except Exception as e:
            return {"error": str(e)}

    def generate_quiz(self, num_questions=5):
        if self.vectorstore is None:
            return {"error": "No documents uploaded"}
        try:
            docs = self.vectorstore.as_retriever(search_kwargs={"k": 8}).invoke("main concepts key points important facts")
            context = "\n\n".join([d.page_content for d in docs])
            prompt = "You are DocuMind AI. Generate " + str(num_questions) + " multiple choice questions from this document.\n\nContext:\n" + context + "\n\nReturn ONLY valid JSON:\n{\"questions\": [{\"question\": \"text?\", \"options\": [\"A) opt1\", \"B) opt2\", \"C) opt3\", \"D) opt4\"], \"correct\": \"A\", \"explanation\": \"why A is correct\"}]}"
            response = self.llm.invoke(prompt)
            text = response.content.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
            return {"error": "Could not generate quiz"}
        except Exception as e:
            return {"error": str(e)}

    def get_document_summary(self, filename):
        if self.vectorstore is None:
            return {"error": "No documents uploaded"}
        try:
            docs = self.vectorstore.as_retriever(search_kwargs={"k": 10}).invoke("summary overview main topics")
            context = "\n\n".join([d.page_content for d in docs if d.metadata.get("source") == filename])
            if not context:
                context = "\n\n".join([d.page_content for d in docs])
            prompt = "Analyze this document and return ONLY valid JSON:\n{\"title\": \"topic\", \"summary\": \"2-3 sentence summary\", \"key_points\": [\"point1\", \"point2\", \"point3\", \"point4\", \"point5\"], \"difficulty\": \"Beginner/Intermediate/Advanced\", \"topics\": [\"topic1\", \"topic2\", \"topic3\"], \"reading_time\": \"X minutes\"}\n\nDocument:\n" + context[:3000]
            response = self.llm.invoke(prompt)
            text = response.content.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
            return {"error": "Could not generate summary"}
        except Exception as e:
            return {"error": str(e)}

    def get_documents(self):
        return {"documents": list(self.documents.keys()), "total": len(self.documents)}

    def delete_document(self, doc_name):
        if doc_name in self.documents:
            del self.documents[doc_name]
            return {"message": "Document removed"}
        return {"error": "Document not found"}




