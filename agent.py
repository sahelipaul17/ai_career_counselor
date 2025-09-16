import os
import io
from fastapi import FastAPI,UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

from PyPDF2 import PdfReader
import docx

# Load .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

app = FastAPI(title="Interactive Career Counselor AI Agent")

# Allow CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define statements
STATEMENTS = [
    { "id": 1, "category": "Data", "text": "I enjoy exploring large datasets to find meaningful patterns." },
    { "id": 2, "category": "Data Analysis", "text": "I like using statistical tools to turn raw data into insights." },
    { "id": 3, "category": "ML Ops", "text": "Automating model deployment and monitoring excites me." },
    { "id": 4, "category": "Building Applications", "text": "Building end-to-end software products motivates me." },
    { "id": 5, "category": "Agents", "text": "Designing autonomous AI agents that act without constant oversight appeals to me." },
    { "id": 6, "category": "Chatbots", "text": "Crafting chatbots that hold natural conversations is rewarding." },
    { "id": 7, "category": "Evals & Testing", "text": "I enjoy stress-testing AI models to measure real-world performance." },
    { "id": 8, "category": "Cost Control & Reduction", "text": "Finding ways to cut compute costs in ML workflows motivates me." },
    { "id": 9, "category": "Fine-Tuning", "text": "Adapting pre-trained models to niche use cases interests me." },
    { "id": 10, "category": "Guardrails", "text": "Implementing robust safety and ethical guardrails for AI systems matters to me." }
]

#Pydantic models    
class Statement(BaseModel):
    id: int
    category: str
    text: str

class UserAnswer(BaseModel):
    statement_id: int
    rating: int  # -1, 0, 1

class UserAnswers(BaseModel):
    answers: List[UserAnswer]

#Endpoints
@app.get("/")
def root():
    return {"message": "Welcome to Interactive Career Counselor AI Agent!"}

# statements
@app.get("/statements")
def get_statements():
    """Return the list of statements one by one."""
    return STATEMENTS

# career suggestions based on user ratings
@app.post("/submit_answers")
def submit_answers(user_answers: UserAnswers):
    """
    Accept user ratings for each statement, calculate interest per category,
    and generate career suggestions using Gemini API.
    """
    # Calculate category scores
    category_scores: Dict[str, int] = {}
    for ans in user_answers.answers:
        stmt = next((s for s in STATEMENTS if s["id"] == ans.statement_id), None)
        if stmt:
            category_scores[stmt["category"]] = category_scores.get(stmt["category"], 0) + ans.rating

    top_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    top_list = [cat for cat, score in top_categories if score > 0]

    # Prepare prompt for Gemini
    system_prompt = """
    You are a career counsellor AI assistant helping users discover Gen-AI career paths.
    """
    user_prompt = f"""
    User has rated statements for career preferences. Top categories: {top_list}.
    Suggest 3-5 fitting career titles. For each, give:
    - Main Gen-AI area
    - Why it fits
    - Two practical starting steps
    Provide short, crisp sentences with bullet points.
    """

    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=400
    )

    return {
        "category_scores": category_scores,
        "top_categories": top_list,
        "career_suggestions": response.choices[0].message.content
    }
# resume upload    
# @app.post("/upload_resume")
# async def upload_resume(file: UploadFile = File(...)):
#     """
#     Accept resume file, analyze it using Gemini, and provide career suggestions.
#     """
#     content = await file.read()
#     text_content = content.decode(errors="ignore")  # crude text extraction
#     prompt = f"""
#     Analyze the following resume and suggest 3-5 Gen-AI career paths:
#     Resume Content: {text_content[:3000]}  # limit for API input
#     For each suggestion, include:
#     - Main Gen-AI area
#     - Why it fits
#     - Two practical starting steps
#     """

#     response = client.chat.completions.create(
#         model="gemini-1.5-flash",
#         messages=[
#             {"role": "system", "content": "You are a Gen-AI career advisor."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=400,
#     )

#     return {
#         "filename": file.filename,
#         "career_recommendations": response.choices[0].message.content
#     }    

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Accept resume file (PDF or DOC/DOCX), extract text, and provide career suggestions.
    """
    try:
        content = await file.read()

        # Extract text based on file type
        if file.filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(content))
            text_content = "\n".join([page.extract_text() or "" for page in reader.pages])
        elif file.filename.endswith((".doc", ".docx")):
            doc = docx.Document(io.BytesIO(content))
            text_content = "\n".join([para.text for para in doc.paragraphs])
        else:
            return {"error": "Unsupported file type. Only PDF and DOC/DOCX allowed."}

        # Limit text to first 3000 chars for API
        text_content = text_content[:3000]

        prompt = f"""
        Analyze the following resume and suggest 3-5 Gen-AI career paths:
        Resume Content: {text_content}
        For each suggestion, include:
        - Main Gen-AI area
        - Why it fits
        - Two practical starting steps
        """

        response = client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=[
                {"role": "system", "content": "You are a Gen-AI career advisor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
        )

        return {
            "filename": file.filename,
            "career_recommendations": response.choices[0].message.content
        }

    except Exception as e:
        return {"error": str(e)}