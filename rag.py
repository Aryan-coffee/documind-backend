import os
import io
import json
import threading
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

_embeddings = None
_lock = threading.Lock()
_query_cache = {}
_session_stores = {}

class SimpleEmbeddings:
    def embed_documents(self, texts):
        import hashlib
        result = []
        for text in texts:
            words = text.lower().split()[:100]
            vec = [0.0] * 384
            for i, word in enumerate(words):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % 384] += 1.0
            norm = sum(x*x for x in vec) ** 0.5
            if norm > 0:
                vec = [x/norm for x in vec]
            result.append(vec)
        return result

    def embed_query(self, text):
        return self.embed_documents([text])[0]

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except:
            _embeddings = SimpleEmbeddings()
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
        self.global_documents = {}

    def get_session_store(self, session_id: str):
        if session_id not in _session_stores:
            _session_stores[session_id] = {
                "vectorstore": None,
                "documents": {},
                "history": []
            }
        return _session_stores[session_id]

    def process_pdf(self, contents, filename, session_id="default"):
        try:
            pdf_reader = PdfReader(io.BytesIO(contents))
            text = ""
            for page in pdf_reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
            if not text.strip():
                return {"error": "No text found in PDF"}

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_text(text)
            chunks = chunks[:80]

            session = self.get_session_store(session_id)

            with _lock:
                if session["vectorstore"] is None:
                    session["vectorstore"] = FAISS.from_texts(
                        chunks, get_embeddings(),
                        metadatas=[{"source": filename}] * len(chunks)
                    )
                else:
                    new_vs = FAISS.from_texts(
                        chunks, get_embeddings(),
                        metadatas=[{"source": filename}] * len(chunks)
                    )
                    session["vectorstore"].merge_from(new_vs)

            session["documents"][filename] = len(chunks)
            return {
                "message": "PDF processed successfully",
                "chunks": len(chunks),
                "total_documents": len(session["documents"])
            }
        except Exception as e:
            return {"error": str(e)}

    def query(self, question, session_id="default", history=None, mode="normal", language="English"):
        session = self.get_session_store(session_id)
        if session["vectorstore"] is None:
            return {"error": "No documents uploaded yet. Please upload a PDF first."}

        cache_key = session_id + question.strip().lower()[:80] + mode
        if cache_key in _query_cache:
            return _query_cache[cache_key]

        try:
            docs = session["vectorstore"].as_retriever(search_kwargs={"k": 2}).invoke(question)
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

            prompt = "You are DocuMind AI. Answer ONLY based on the provided document context. Do not use any external knowledge.\n\nStyle: " + style + "\nLanguage: " + language + "\n\nDocument Context:\n" + context + "\n\nHistory:\n" + history_text + "\nUser: " + question + "\nDocuMind AI:"

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

    def compare_documents(self, question, doc1, doc2, session_id="default"):
        session = self.get_session_store(session_id)
        if session["vectorstore"] is None:
            return {"error": "No documents uploaded"}
        try:
            docs = session["vectorstore"].as_retriever(search_kwargs={"k": 4}).invoke(question)
            doc1_ctx = "\n".join([d.page_content for d in docs if d.metadata.get("source") == doc1]) or "No content found"
            doc2_ctx = "\n".join([d.page_content for d in docs if d.metadata.get("source") == doc2]) or "No content found"
            prompt = "Compare these documents.\n\nQuestion: " + question + "\n\nDoc1 (" + doc1 + "):\n" + doc1_ctx[:2000] + "\n\nDoc2 (" + doc2 + "):\n" + doc2_ctx[:2000] + "\n\nGive similarities, differences, and recommendation.\n\nDocuMind AI:"
            llm = get_llm()
            response = llm.invoke(prompt)
            return {"answer": response.content, "sources": [doc1, doc2], "confidence": 85}
        except Exception as e:
            return {"error": str(e)}

    def generate_quiz(self, num_questions=5, session_id="default"):
        session = self.get_session_store(session_id)
        if session["vectorstore"] is None:
            return {"error": "No documents uploaded"}
        try:
            docs = session["vectorstore"].as_retriever(search_kwargs={"k": 4}).invoke("main concepts key points")
            context = "\n\n".join([d.page_content for d in docs])
            prompt = "Generate " + str(num_questions) + " MCQ questions from this document ONLY. Return ONLY valid JSON:\n{\"questions\": [{\"question\": \"text?\", \"options\": [\"A) opt1\", \"B) opt2\", \"C) opt3\", \"D) opt4\"], \"correct\": \"A\", \"explanation\": \"why\"}]}\n\nDocument:\n" + context[:3000]
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

    def get_document_summary(self, filename, session_id="default"):
        session = self.get_session_store(session_id)
        if session["vectorstore"] is None:
            return {"error": "No documents uploaded"}
        try:
            docs = session["vectorstore"].as_retriever(search_kwargs={"k": 4}).invoke("summary overview main topics")
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

    def get_documents(self, session_id="default"):
        session = self.get_session_store(session_id)
        return {"documents": list(session["documents"].keys()), "total": len(session["documents"])}

    def delete_document(self, doc_name, session_id="default"):
        session = self.get_session_store(session_id)
        if doc_name in session["documents"]:
            del session["documents"][doc_name]
            session["vectorstore"] = None
            return {"message": "Document removed — please re-upload other documents"}
        return {"error": "Document not found"}

    def clear_session(self, session_id="default"):
        if session_id in _session_stores:
            del _session_stores[session_id]
        return {"message": "Session cleared"}

