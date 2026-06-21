import os
import io
import json
import threading
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from groq import Groq

load_dotenv()

FAISS_PATH = "faiss_store"
_lock = threading.Lock()
_query_cache = {}
_groq_client = None
_embeddings = None

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 128, "normalize_embeddings": True, "show_progress_bar": False}
        )
    return _embeddings

def embed_texts(texts):
    client = get_groq_client()
    try:
        result = []
        for text in texts:
            response = client.embeddings.create(
                model="nomic-embed-text-v1_5",
                input=text[:512]
            )
            result.append(response.data[0].embedding)
        return result
    except:
        return get_embeddings().embed_documents(texts)

class RAGSystem:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=2048
        )
        self.vectorstore = None
        self.documents = {}
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80
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
            chunks = chunks[:80]
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
            return {"error": "No documents uploaded yet. Please upload a PDF first."}
        cache_key = question.strip().lower()[:100] + mode + language
        if cache_key in _query_cache:
            return _query_cache[cache_key]
        try:
            docs = self.vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(question)
            context = "\n\n".join([d.page_content for d in docs])
            sources = list(set([d.metadata.get("source", "Unknown") for d in docs]))
            history_text = ""
            if history:
                for m in history[-4:]:
                    history_text += m["role"] + ": " + m["content"][:200] + "\n"
            if mode == "student":
                style = "Explain simply with examples. Friendly tone."
            elif mode == "professor":
                style = "Detailed academic analysis with technical depth."
            elif mode == "summary":
                style = "Concise bullet-point summary only."
            else:
                style = "Helpful, clear, professional. Use bullets when needed."
            prompt = "You are DocuMind AI — world most advanced document assistant.\nStyle: " + style + "\nLanguage: " + language + "\n\nDocument Context:\n" + context + "\n\nChat History:\n" + history_text + "\nUser: " + question + "\nDocuMind AI:"
            response = self.llm.invoke(prompt)
            answer = response.content
            confidence = min(92, max(50, len(docs) * 25 + (15 if question.lower() in context.lower() else 0)))
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
            doc1_context = "\n".join([d.page_content for d in docs if d.metadata.get("source") == doc1])
            doc2_context = "\n".join([d.page_content for d in docs if d.metadata.get("source") == doc2])
            if not doc1_context:
                doc1_context = "No relevant content found for this document"
            if not doc2_context:
                doc2_context = "No relevant content found for this document"
            prompt = "You are DocuMind AI. Compare these two documents.\n\nQuestion: " + question + "\n\nDocument 1 (" + doc1 + "):\n" + doc1_context[:2000] + "\n\nDocument 2 (" + doc2 + "):\n" + doc2_context[:2000] + "\n\nProvide:\n- Key similarities\n- Key differences\n- Which is better for this topic and why\n\nDocuMind AI:"
            response = self.llm.invoke(prompt)
            return {"answer": response.content, "sources": [doc1, doc2], "confidence": 85}
        except Exception as e:
            return {"error": str(e)}

    def generate_quiz(self, num_questions=5):
        if self.vectorstore is None:
            return {"error": "No documents uploaded"}
        try:
            docs = self.vectorstore.as_retriever(search_kwargs={"k": 6}).invoke("main concepts key points important facts")
            context = "\n\n".join([d.page_content for d in docs])
            prompt = "You are DocuMind AI. Generate " + str(num_questions) + " MCQ questions from this document.\n\nContext:\n" + context[:4000] + "\n\nReturn ONLY valid JSON — no extra text:\n{\"questions\": [{\"question\": \"text?\", \"options\": [\"A) opt1\", \"B) opt2\", \"C) opt3\", \"D) opt4\"], \"correct\": \"A\", \"explanation\": \"why A is correct\"}]}"
            response = self.llm.invoke(prompt)
            text = response.content.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
            return {"error": "Could not generate quiz. Try again."}
        except Exception as e:
            return {"error": str(e)}

    def get_document_summary(self, filename):
        if self.vectorstore is None:
            return {"error": "No documents uploaded"}
        try:
            docs = self.vectorstore.as_retriever(search_kwargs={"k": 6}).invoke("summary overview main topics introduction")
            context = "\n\n".join([d.page_content for d in docs if d.metadata.get("source") == filename])
            if not context:
                context = "\n\n".join([d.page_content for d in docs])
            prompt = "Analyze this document. Return ONLY valid JSON — no extra text:\n{\"title\": \"document topic\", \"summary\": \"2-3 sentence summary\", \"key_points\": [\"point1\", \"point2\", \"point3\", \"point4\", \"point5\"], \"difficulty\": \"Beginner/Intermediate/Advanced\", \"topics\": [\"topic1\", \"topic2\", \"topic3\"], \"reading_time\": \"X minutes\"}\n\nDocument:\n" + context[:3000]
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
