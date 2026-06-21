from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from rag import RAGSystem
from database import init_db, get_db, ChatHistory, Document, ImageHistory
from features import chat_with_website, chat_with_youtube, analyze_resume, analyze_data, extract_smart_alerts
from sqlalchemy.orm import Session
import uvicorn
import json
import pandas as pd
import io
from pypdf import PdfReader

app = FastAPI(title="DocuMind AI API v2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
rag = RAGSystem()

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
    mode: str = "normal"
    language: str = "English"

class CompareRequest(BaseModel):
    question: str
    doc1: str
    doc2: str
    session_id: str = "default"

class ImageRequest(BaseModel):
    prompt: str
    session_id: str = "default"
    style: str = "realistic"

class QuizRequest(BaseModel):
    num_questions: int = 5

class WebsiteRequest(BaseModel):
    url: str
    question: str
    session_id: str = "default"

class YouTubeRequest(BaseModel):
    video_url: str
    question: str
    session_id: str = "default"

class DataRequest(BaseModel):
    question: str
    session_id: str = "default"

class AlertRequest(BaseModel):
    filename: str

@app.get("/")
def root():
    return {"status": "DocuMind AI v2.0 Running", "features": ["chat","image","quiz","compare","website","youtube","resume","data","alerts"]}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": str(__import__("datetime").datetime.now())}

@app.post("/upload")
async def upload(file: UploadFile = File(...), session_id: str = "default", db: Session = Depends(get_db)):
    contents = await file.read()
    if file.filename.endswith(".pdf"):
        result = rag.process_pdf(contents, file.filename)
        if "error" not in result:
            db.add(Document(session_id=session_id, filename=file.filename, chunks=result.get("chunks", 0)))
            db.commit()
        return result
    elif file.filename.endswith((".csv", ".xlsx", ".xls")):
        try:
            if file.filename.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(contents))
            else:
                df = pd.read_excel(io.BytesIO(contents))
            return {"message": "Data file loaded", "rows": len(df), "columns": list(df.columns), "data_json": df.to_json(), "filename": file.filename}
        except Exception as e:
            return {"error": str(e)}
    else:
        raise HTTPException(400, "Supported: PDF, CSV, Excel")

@app.post("/chat")
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    history = db.query(ChatHistory).filter(ChatHistory.session_id == req.session_id).order_by(ChatHistory.timestamp).limit(10).all()
    history_list = [{"role": str(h.role), "content": str(h.content)} for h in history]
    result = rag.query(req.question, history_list, req.mode, req.language)
    if "error" not in result:
        db.add(ChatHistory(session_id=req.session_id, role="Human", content=req.question))
        db.add(ChatHistory(session_id=req.session_id, role="DocuMind AI", content=result["answer"], sources=json.dumps(result.get("sources", []))))
        db.commit()
    return result

@app.post("/compare")
async def compare(req: CompareRequest, db: Session = Depends(get_db)):
    result = rag.compare_documents(req.question, req.doc1, req.doc2)
    if "answer" in result:
        db.add(ChatHistory(session_id=req.session_id, role="Human", content="[COMPARE] " + req.question))
        db.add(ChatHistory(session_id=req.session_id, role="DocuMind AI", content=result["answer"]))
        db.commit()
    return result

@app.post("/quiz")
async def generate_quiz(req: QuizRequest):
    return rag.generate_quiz(req.num_questions)

@app.get("/summary/{filename}")
async def get_summary(filename: str):
    return rag.get_document_summary(filename)

@app.post("/generate-image")
async def generate_image(req: ImageRequest, db: Session = Depends(get_db)):
    try:
        import urllib.parse
        import random
        styles = {
            "realistic": "photorealistic ultra detailed 8k professional",
            "anime": "anime art style vibrant colorful detailed",
            "cartoon": "cartoon illustration colorful fun bold",
            "painting": "oil painting artistic impressionist masterpiece",
            "3d": "3D render CGI octane detailed lighting"
        }
        style_suffix = styles.get(req.style, "photorealistic ultra detailed")
        combined = req.prompt.strip() + ", " + style_suffix
        encoded = urllib.parse.quote(combined)
        seed = random.randint(1, 999999)
        image_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&enhance=true&seed={seed}&model=flux"
        db.add(ImageHistory(session_id=req.session_id, prompt=req.prompt, image_url=image_url))
        db.commit()
        return {"image_url": image_url, "prompt": req.prompt, "style": req.style}
    except Exception as e:
        return {"error": str(e)}

@app.post("/website-chat")
async def website_chat(req: WebsiteRequest, db: Session = Depends(get_db)):
    result = chat_with_website(req.url, req.question)
    if "answer" in result:
        db.add(ChatHistory(session_id=req.session_id, role="Human", content="[WEB] " + req.question))
        db.add(ChatHistory(session_id=req.session_id, role="DocuMind AI", content=result["answer"]))
        db.commit()
    return result

@app.post("/youtube-chat")
async def youtube_chat(req: YouTubeRequest, db: Session = Depends(get_db)):
    result = chat_with_youtube(req.video_url, req.question)
    if "answer" in result:
        db.add(ChatHistory(session_id=req.session_id, role="Human", content="[YT] " + req.question))
        db.add(ChatHistory(session_id=req.session_id, role="DocuMind AI", content=result["answer"]))
        db.commit()
    return result

@app.post("/analyze-resume")
async def analyze_resume_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    if file.filename.endswith(".pdf"):
        pdf = PdfReader(io.BytesIO(contents))
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    else:
        text = contents.decode("utf-8", errors="ignore")
    result = analyze_resume(text)
    return result

@app.post("/analyze-data")
async def analyze_data_endpoint(req: DataRequest):
    return {"error": "Send data_json with the request"}

@app.post("/analyze-data-json")
async def analyze_data_json(request: dict):
    question = request.get("question", "Analyze this data")
    data_json = request.get("data_json", "")
    return analyze_data(data_json, question)

@app.post("/smart-alerts")
async def smart_alerts(req: AlertRequest):
    result = rag.get_document_summary(req.filename)
    if "error" in result:
        return result
    content = " ".join(result.get("key_points", [])) + " " + result.get("summary", "")
    return extract_smart_alerts(content)

@app.get("/history/{session_id}")
async def get_history(session_id: str, db: Session = Depends(get_db)):
    history = db.query(ChatHistory).filter(ChatHistory.session_id == session_id).order_by(ChatHistory.timestamp).all()
    return {"history": [{"role": h.role, "content": h.content, "sources": h.sources, "timestamp": str(h.timestamp)} for h in history]}

@app.delete("/history/{session_id}")
async def clear_history(session_id: str, db: Session = Depends(get_db)):
    db.query(ChatHistory).filter(ChatHistory.session_id == session_id).delete()
    db.commit()
    return {"message": "History cleared"}

@app.get("/documents")
def get_documents():
    return rag.get_documents()

@app.delete("/documents/{doc_name}")
def delete_document(doc_name: str):
    return rag.delete_document(doc_name)

@app.post("/chat-stream")
async def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    history = db.query(ChatHistory).filter(ChatHistory.session_id == req.session_id).order_by(ChatHistory.timestamp).limit(10).all()
    history_list = [{"role": str(h.role), "content": str(h.content)} for h in history]

    async def generate():
        try:
            from langchain_groq import ChatGroq
            import os
            llm = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=2048,
                streaming=True
            )
            docs = rag.vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(req.question) if rag.vectorstore else []
            context = "\n\n".join([d.page_content for d in docs])
            sources = list(set([d.metadata.get("source", "Unknown") for d in docs]))
            history_text = ""
            for m in history_list[-6:]:
                history_text += m["role"] + ": " + m["content"] + "\n"

            if req.mode == "student":
                style = "Explain simply, use analogies, friendly tone."
            elif req.mode == "professor":
                style = "Detailed academic analysis with technical depth."
            elif req.mode == "summary":
                style = "Concise bullet-point summary."
            else:
                style = "Helpful, clear, professional."

            prompt = "You are DocuMind AI.\nStyle: " + style + "\nLanguage: " + req.language + "\n\nContext:\n" + context + "\n\nHistory:\n" + history_text + "\nHuman: " + req.question + "\nDocuMind AI:"

            full_response = ""
            async for chunk in llm.astream(prompt):
                if chunk.content:
                    full_response += chunk.content
                    yield f"data: {chunk.content}\n\n"

            sources_str = ",".join(sources)
            yield f"data: [SOURCES]{sources_str}[/SOURCES]\n\n"
            yield "data: [DONE]\n\n"

            db.add(ChatHistory(session_id=req.session_id, role="Human", content=req.question))
            db.add(ChatHistory(session_id=req.session_id, role="DocuMind AI", content=full_response, sources=str(sources)))
            db.commit()

        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)






