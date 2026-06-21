import os
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024
    )

def scrape_website(url: str):
    try:
        if not url.startswith("http"):
            url = "https://" + url
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, headers=headers, timeout=12)
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
        content = "\n".join(lines[:200])
        title = soup.title.string.strip() if soup.title else url
        return {"title": title, "content": content, "url": url}
    except Exception as e:
        return {"error": f"Could not fetch: {str(e)}"}

def chat_with_website(url: str, question: str):
    data = scrape_website(url)
    if "error" in data:
        return {"error": data["error"]}
    try:
        llm = get_llm()
        prompt = "You are DocuMind AI. Answer based on website.\n\nSite: " + data["title"] + "\n\nContent:\n" + data["content"][:4000] + "\n\nQuestion: " + question + "\n\nDocuMind AI:"
        response = llm.invoke(prompt)
        return {"answer": response.content, "title": data["title"], "url": url}
    except Exception as e:
        return {"error": str(e)}

def chat_with_youtube(video_url: str, question: str):
    try:
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            return {"error": "Invalid YouTube URL"}

        title = "YouTube Video"
        try:
            oembed = requests.get(f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json", timeout=5)
            if oembed.status_code == 200:
                title = oembed.json().get("title", title)
        except:
            pass

        transcript = ""
        has_transcript = False
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            yapi = YouTubeTranscriptApi()
            try:
                fetched = yapi.fetch(video_id, languages=["en","hi","en-IN","en-GB"])
            except:
                fetched = yapi.fetch(video_id)
            parts = []
            for t in fetched:
                if isinstance(t, dict):
                    parts.append(t.get("text",""))
                elif hasattr(t, "text"):
                    parts.append(t.text)
                else:
                    parts.append(str(t))
            transcript = " ".join(parts)[:5000]
            has_transcript = True
        except Exception as te:
            transcript = f"Transcript unavailable: {str(te)[:100]}"

        llm = get_llm()
        prompt = "You are DocuMind AI. Answer about this YouTube video.\n\nTitle: " + title + "\nTranscript:\n" + transcript + "\n\nQuestion: " + question + "\n\nDocuMind AI:"
        response = llm.invoke(prompt)
        return {"answer": response.content, "video_url": video_url, "title": title, "has_transcript": has_transcript}
    except Exception as e:
        return {"error": str(e)}

def analyze_resume(content: str):
    try:
        llm = get_llm()
        prompt = """Analyze this resume. Return ONLY valid JSON:
{"name":"name","score":75,"strengths":["s1","s2","s3"],"weaknesses":["w1","w2"],"missing_skills":["m1","m2","m3"],"improvements":["i1","i2","i3"],"best_roles":["r1","r2","r3"],"summary":"assessment","ats_score":70,"experience_level":"Fresher/Junior/Mid/Senior"}

Resume:
""" + content[:4000]
        response = llm.invoke(prompt)
        text = response.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        return {"error": "Could not parse"}
    except Exception as e:
        return {"error": str(e)}

def analyze_data(df_json: str, question: str):
    try:
        df = pd.read_json(df_json)
        stats = df.describe().to_string()
        columns = list(df.columns)
        sample = df.head(5).to_string()
        llm = get_llm()
        prompt = "You are DocuMind AI data analyst.\n\nDataset: " + str(df.shape[0]) + " rows, columns: " + str(columns) + "\nStats:\n" + stats[:1500] + "\nSample:\n" + sample[:1500] + "\n\nQuestion: " + question + "\n\nDocuMind AI:"
        response = llm.invoke(prompt)
        return {"answer": response.content, "columns": columns, "rows": df.shape[0]}
    except Exception as e:
        return {"error": str(e)}

def extract_smart_alerts(content: str):
    try:
        llm = get_llm()
        prompt = """Extract alerts from document. Return ONLY valid JSON:
{"alerts":[{"type":"deadline/warning/action","title":"title","description":"desc","priority":"high/medium/low","date":"date or null"}],"summary":"summary"}

Document:
""" + content[:3000]
        response = llm.invoke(prompt)
        text = response.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        return {"error": "Could not extract alerts"}
    except Exception as e:
        return {"error": str(e)}
