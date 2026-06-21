import os
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
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
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 30]
        content = "\n".join(lines[:200])
        title = soup.title.string if soup.title else url
        return {"title": title, "content": content, "url": url}
    except Exception as e:
        return {"error": str(e)}

def chat_with_website(url: str, question: str):
    data = scrape_website(url)
    if "error" in data:
        return {"error": data["error"]}
    prompt = "You are DocuMind AI. Answer based on this website content.\n\nWebsite: " + data["title"] + "\nURL: " + data["url"] + "\n\nContent:\n" + data["content"][:4000] + "\n\nQuestion: " + question + "\n\nDocuMind AI:"
    response = llm.invoke(prompt)
    return {"answer": response.content, "title": data["title"], "url": url}

def get_youtube_info(video_url: str):
    try:
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            return {"error": "Invalid YouTube URL"}
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        r = requests.get(oembed_url, timeout=5)
        info = r.json() if r.status_code == 200 else {}
        page_url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        page = requests.get(page_url, headers=headers, timeout=10)
        soup = BeautifulSoup(page.content, "html.parser")
        description = ""
        for script in soup.find_all("script"):
            if script.string and "shortDescription" in str(script.string):
                import re
                match = re.search(r'"shortDescription":"(.*?)"(?:,|\})', str(script.string))
                if match:
                    description = match.group(1).replace("\\n", "\n").replace('\\"', '"')[:2000]
                    break
        title = info.get("title", "YouTube Video")
        author = info.get("author_name", "Unknown")
        return {
            "video_id": video_id,
            "title": title,
            "author": author,
            "description": description,
            "url": page_url
        }
    except Exception as e:
        return {"error": str(e)}

def chat_with_youtube(video_url: str, question: str):
    try:
        info = get_youtube_info(video_url)
        if "error" in info:
            return {"error": info["error"]}
        transcript = ""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            yapi = YouTubeTranscriptApi()
            try:
                fetched = yapi.fetch(info["video_id"], languages=["en","hi","en-IN","en-GB"])
            except:
                fetched = yapi.fetch(info["video_id"])
            parts = []
            for t in fetched:
                if isinstance(t, dict):
                    parts.append(t.get("text", ""))
                elif hasattr(t, "text"):
                    parts.append(t.text)
                else:
                    parts.append(str(t))
            transcript = " ".join(parts)[:5000]
        except Exception as te:
            transcript = f"[Transcript unavailable: {str(te)[:100]}]"
        context = f"Title: {info['title']}\nChannel: {info['author']}\nDescription: {info['description']}\n\nTranscript:\n{transcript}"
        prompt = "You are DocuMind AI — a smart video analyst. Answer the question based on this YouTube video information.\n\nVideo Info:\n" + context + "\n\nQuestion: " + question + "\n\nIf transcript is unavailable, answer based on title and description only. Be helpful.\n\nDocuMind AI:"
        response = llm.invoke(prompt)
        return {
            "answer": response.content,
            "video_url": video_url,
            "title": info["title"],
            "author": info["author"],
            "has_transcript": "[Transcript unavailable" not in transcript
        }
    except Exception as e:
        return {"error": str(e)}

def analyze_resume(content: str):
    prompt = """You are DocuMind AI — an expert career coach and HR professional. Analyze this resume and return ONLY valid JSON:
{
  "name": "candidate name",
  "score": 75,
  "strengths": ["strength1", "strength2", "strength3"],
  "weaknesses": ["weakness1", "weakness2"],
  "missing_skills": ["skill1", "skill2", "skill3"],
  "improvements": ["improvement1", "improvement2", "improvement3"],
  "best_roles": ["role1", "role2", "role3"],
  "summary": "2-3 sentence overall assessment",
  "ats_score": 70,
  "experience_level": "Fresher/Junior/Mid/Senior"
}

Resume:
""" + content[:4000]
    response = llm.invoke(prompt)
    text = response.content.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])
    return {"error": "Could not analyze resume"}

def analyze_data(df_json: str, question: str):
    try:
        df = pd.read_json(df_json)
        stats = df.describe().to_string()
        columns = list(df.columns)
        shape = df.shape
        sample = df.head(5).to_string()
        prompt = "You are DocuMind AI — a data analyst expert. Analyze this dataset and answer the question.\n\nDataset Info:\nShape: " + str(shape) + "\nColumns: " + str(columns) + "\n\nStatistics:\n" + stats + "\n\nSample Data:\n" + sample + "\n\nQuestion: " + question + "\n\nProvide insights, patterns, and actionable recommendations.\n\nDocuMind AI:"
        response = llm.invoke(prompt)
        return {"answer": response.content, "columns": columns, "rows": shape[0], "stats": df.describe().to_dict()}
    except Exception as e:
        return {"error": str(e)}

def extract_smart_alerts(content: str):
    prompt = """You are DocuMind AI. Extract all important alerts, deadlines, dates, action items, and warnings. Return ONLY valid JSON:
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
  "summary": "Overall document risk/action summary"
}

Document:
""" + content[:4000]
    response = llm.invoke(prompt)
    text = response.content.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])
    return {"error": "Could not extract alerts"}
