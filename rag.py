import os
import io
import json
import threading
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

FAISS_PATH = "faiss_store"
_lock = threading.Lock()
_query_cache = {}
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings

def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024
    )

class RAGSystem:
    def __init__(self):
        self.vectorstore = None
        self.documents = {}
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self._load_faiss()

    def _load_faiss(self):
        try:
            if os.path.exists(FAISS_PATH):
                self.vectorstore = FAISS.load_local(
                    FAISS_PATH, get_embeddings(),
                    allow_dangerous_deserialization=True
                )
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
            chunks = chunks[:50]
            with _lock:
                if self.vectorstore is None:
                    self.vectorstore = FAISS.from_texts(
                        chunks, get_embeddings(),
                        metadatas=[{"source": filename}] * len(chunks)
                    )
                else:
                    new_vs = FAISS.from_texts(
                        chunks, get_embeddings(),
                        metadatas=[{"source": filename}] * len(chunks)
                    )
                    self.vectorstore.merge_from(new_vs)
                self._save_faiss()
            self.documents[filename] = len(chunks)
            return {
                "message": "PDF processed successfully",
                "chunks": len(chunks),
                "total_documents": len(self.documents)
            }
        except Exception as e:
            return {"error": str(e)}

    def query(self, question, history=None, mode="normal", language="English"):
        if self.vectorstore is None:
            return {"error": "No documents uploaded yet."}
        cache_key = question.strip().lower()[:80] + mode + language
        if cache_key in _query_cache:
            return _query_cache[cache_key]
        try:
            docs = self.vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(question)
            context = "\n\n".join([d.page_content for d in docs])
            sources = list(set([d.metadata.get("source", "Unknown") for d in docs]))
            history_text = ""
            if history:
                for m in history[-4:]:
                    history_text += m["role"] + ": " + m["content"][:100] + "\n"
            if mode == "student":
                style = "Simple words, analogies, friendly."
            elif mode == "professor":
                style = "Detailed academic analysis."
            elif mode == "summary":
                style = "Bullet points only, concise."
            else:
                style = "Clear, helpful, professional."
            prompt = "You are DocuMind AI.\nStyle: " + style + "\nLanguage: " + language + "\n\nContext:\n" + context + "\n\nHistory:\n" + history_text + "\nUser: " + question + "\nDocuMind AI:"
            llm = get_llm()
            response = llm.invoke(prompt)
            answer = response.content
            confidence = min(92, max(50, len(docs) * 25))
            result = {"answer": answer, "sources": sources, "confidence": confidence, "mode": mode}
            if len(_query_cache) < 50:
                _query_cache[cache_key] = result
            return result
        except Exception as e:
            return {"error": str(e)}

    def compare_documents(self, question, doc1, doc2):
        if self.vectorstore is None:
            return {"error": "No documents uploaded"}
        try:
            docs = self.vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(question)
            doc1_ctx = "\n".join([d.page_content for d in docs if d.metadata.get("source") == doc1]) or "No content found"
            doc2_ctx = "\n".join([d.page_content for d in docs if d.metadata.get("source") == doc2]) or "No content found"
            prompt = "Compare these documents.\n\nQuestion: " + question + "\n\nDoc1 (" + doc1 + "):\n" + doc1_ctx[:2000] + "\n\nDoc2 (" + doc2 + "):\n" + doc2_ctx[:2000] + "\n\nGive similarities, differences, and recommendation.\n\nDocuMind AI:"
            llm = get_llm()
            response = llm.invoke(prompt)
            return {"answer": response.content, "sources": [doc1, doc2], "confidence": 85}
        except Exception as e:
            return {"error": str(e)}

    def generate_quiz(self, num_questions=5):
        if self.vectorstore is None:
            return {"error": "No documents uploaded"}
        try:
            docs = self.vectorstore.as_retriever(search_kwargs={"k": 4}).invoke("main concepts key points")
            context = "\n\n".join([d.page_content for d in docs])
            prompt = "Generate " + str(num_questions) + " MCQ questions. Return ONLY valid JSON:\n{\"questions\": [{\"question\": \"text?\", \"options\": [\"A) opt1\", \"B) opt2\", \"C) opt3\", \"D) opt4\"], \"correct\": \"A\", \"explanation\": \"why\"}]}\n\nContext:\n" + context[:3000]
            llm = get_llm()
            response = llm.invoke(prompt)
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
            docs = self.vectorstore.as_retriever(search_kwargs={"k": 4}).invoke("summary overview main topics")
            context = "\n\n".join([d.page_content for d in docs if d.metadata.get("source") == filename])
            if not context:
                context = "\n\n".join([d.page_content for d in docs])
            prompt = "Analyze document. Return ONLY valid JSON:\n{\"title\": \"topic\", \"summary\": \"2-3 sentences\", \"key_points\": [\"p1\",\"p2\",\"p3\",\"p4\",\"p5\"], \"difficulty\": \"Beginner/Intermediate/Advanced\", \"topics\": [\"t1\",\"t2\",\"t3\"], \"reading_time\": \"X min\"}\n\nDoc:\n" + context[:2000]
            llm = get_llm()
            response = llm.invoke(prompt)
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

