import os
import json
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=2048
)

def scrape_website(url: str):
    try:
        if not url.startswith("http"):
            url = "https://" + url
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside","iframe","noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
        content = "\n".join(lines[:300])
        title = soup.title.string.strip() if soup.title else url
        return {"title": title, "content": content, "url": url}
    except Exception as e:
        return {"error": f"Could not fetch website: {str(e)}"}

def chat_with_website(url: str, question: str):
    data = scrape_website(url)
    if "error" in data:
        return {"error": data["error"]}
    prompt = "You are DocuMind AI. Answer based on this website.\n\nWebsite: " + data["title"] + "\nURL: " + data["url"] + "\n\nContent:\n" + data["content"][:5000] + "\n\nQuestion: " + question + "\n\nDocuMind AI:"
    try:
        response = llm.invoke(prompt)
        return {"answer": response.content, "title": data["title"], "url": url}
    except Exception as e:
        return {"error": str(e)}

def get_youtube_transcript(video_url: str):
    try:
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            return {"error": "Invalid YouTube URL. Use format: youtube.com/watch?v=..."}
        from youtube_transcript_api import YouTubeTranscriptApi
        yapi = YouTubeTranscriptApi()
        try:
            fetched = yapi.fetch(video_id, languages=["en","hi","en-IN","en-GB","auto"])
        except:
            try:
                fetched = yapi.fetch(video_id)
            except:
                transcript_list = yapi.list(video_id)
                available = list(transcript_list)
                if available:
                    fetched = available[0].fetch()
                else:
                    return {"error": "No transcripts available for this video"}
        parts = []
        for t in fetched:
            if isinstance(t, dict):
                parts.append(t.get("text", ""))
            elif hasattr(t, "text"):
                parts.append(t.text)
            else:
                parts.append(str(t))
        transcript = " ".join(parts)
        return {"transcript": transcript, "video_id": video_id}
    except Exception as e:
        return {"error": f"Transcript error: {str(e)[:200]}"}

def get_youtube_info(video_url: str):
    try:
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            return {"error": "Invalid YouTube URL"}
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        r = requests.get(oembed_url, timeout=8)
        info = r.json() if r.status_code == 200 else {}
        return {"video_id": video_id, "title": info.get("title", "YouTube Video"), "author": info.get("author_name", "Unknown"), "url": f"https://www.youtube.com/watch?v={video_id}"}
    except Exception as e:
        return {"error": str(e)}

def chat_with_youtube(video_url: str, question: str):
    try:
        info = get_youtube_info(video_url)
        transcript_data = get_youtube_transcript(video_url)
        transcript = ""
        has_transcript = False
        if "error" not in transcript_data:
            transcript = transcript_data["transcript"][:6000]
            has_transcript = True
        context = f"Title: {info.get('title', 'Unknown')}\nChannel: {info.get('author', 'Unknown')}\n\nTranscript:\n{transcript if has_transcript else '[Transcript not available - answering from title/description]'}"
        prompt = "You are DocuMind AI. Answer based on this YouTube video.\n\nVideo Info:\n" + context + "\n\nQuestion: " + question + "\n\nIf transcript unavailable, answer based on title. Be helpful.\n\nDocuMind AI:"
        response = llm.invoke(prompt)
        return {"answer": response.content, "video_url": video_url, "title": info.get("title", ""), "has_transcript": has_transcript}
    except Exception as e:
        return {"error": str(e)}

def analyze_resume(content: str):
    prompt = """You are DocuMind AI — expert career coach. Analyze this resume. Return ONLY valid JSON:
{
  "name": "candidate name or Unknown",
  "score": 75,
  "strengths": ["strength1", "strength2", "strength3"],
  "weaknesses": ["weakness1", "weakness2"],
  "missing_skills": ["skill1", "skill2", "skill3"],
  "improvements": ["improvement1", "improvement2", "improvement3"],
  "best_roles": ["role1", "role2", "role3"],
  "summary": "2-3 sentence assessment",
  "ats_score": 70,
  "experience_level": "Fresher/Junior/Mid/Senior"
}

Resume Content:
""" + content[:5000]
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        return {"error": "Could not parse resume analysis"}
    except Exception as e:
        return {"error": str(e)}

def analyze_data(df_json: str, question: str):
    try:
        df = pd.read_json(df_json)
        stats = df.describe().to_string()
        columns = list(df.columns)
        shape = df.shape
        sample = df.head(5).to_string()
        prompt = "You are DocuMind AI — data analyst. Answer this question about the dataset.\n\nDataset: " + str(shape[0]) + " rows, " + str(shape[1]) + " columns\nColumns: " + str(columns) + "\n\nStats:\n" + stats[:2000] + "\n\nSample:\n" + sample[:2000] + "\n\nQuestion: " + question + "\n\nProvide clear insights and recommendations.\n\nDocuMind AI:"
        response = llm.invoke(prompt)
        return {"answer": response.content, "columns": columns, "rows": shape[0]}
    except Exception as e:
        return {"error": str(e)}

def extract_smart_alerts(content: str):
    prompt = """You are DocuMind AI. Extract important alerts from this document. Return ONLY valid JSON:
{
  "alerts": [
    {
      "type": "deadline/warning/action/date/important",
      "title": "Alert title",
      "description": "What needs attention",
      "priority": "high/medium/low",
      "date": "date if mentioned or null"
    }
  ],
  "summary": "Overall summary of key action items"
}

Document:
""" + content[:4000]
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        return {"error": "Could not extract alerts"}
    except Exception as e:
        return {"error": str(e)}
