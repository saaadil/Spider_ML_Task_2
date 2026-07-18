import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent  # locate current script path
BASE = SCRIPT_DIR.parent  # get parent base directory
DATA_DIR = BASE / "data"  # set data folder
KB_DIR = DATA_DIR / "knowledge_base"  # knowledge base dir
CHROMA_DIR = DATA_DIR / "chroma_db"  # vector database storage path
ODIR = BASE / "outputs"  # output folder

for d in [DATA_DIR, KB_DIR, CHROMA_DIR, ODIR]:  # loop over required directories
    d.mkdir(parents=True, exist_ok=True)  # create folder if missing
print(f"\n[Setup] Python : {sys.version.split()[0]}")  # print python version
print(f"[Setup] Outputs -> {ODIR}")  # print output path
print(f"[Setup] ChromaDB -> {CHROMA_DIR}")  # print db path

EMBED_MODEL = "all-MiniLM-L6-v2"  # choose dense embedding model
OLLAMA_MODEL = "llama3.2"  # local llm name
OLLAMA_URL = "http://localhost:11434"  # local ollama endpoint
COLLECTION_NAME = "medical_kb"  # chroma collection name
CONFIG = {  # master configuration dictionary
    'embed_model': EMBED_MODEL,
    'ollama_model': OLLAMA_MODEL,
    'ollama_url': OLLAMA_URL,
    'collection_name': COLLECTION_NAME,
    'chunk_size': 500,
    'chunk_overlap': 50,
    'top_k': 5,
    'confidence_high': 0.75,
    'confidence_medium': 0.50,
    'batch_size': 64,
    'max_tokens': 512,
    'temperature': 0.1,
}
print(f"[Setup] Embed : {EMBED_MODEL}")  # show active embedder
print(f"[Setup] LLM : {OLLAMA_MODEL} via Ollama")  # show active llm

SOURCES = ['MedQuAD', 'WHO Guidelines', 'CDC Recommendations', 'NICE Guidelines']  # trusted medical sources
EMERGENCY_KEYWORDS = ["heart attack", "stroke", "chest pain", "can't breathe", "difficulty breathing", "not breathing", "suicide", "suicidal", "kill myself", "overdose", "unconscious", "seizure", "convulsion", "severe bleeding", "anaphylaxis", "allergic shock", "911", "emergency"]  # emergency trigger words
UNSAFE_PATTERNS = ["how many pills", "how much medication", "lethal dose", "stop taking my medication", "without prescription", "instead of my doctor", "self medicate", "diagnose myself", "how to stop taking", "skip my dose"]  # unsafe medical queries
SYSTEM_PROMPT = "You are a Healthcare Information Assistant. Answer ONLY using the retrieved medical context provided below. If the context does not contain enough information to answer, say: 'I cannot find sufficient information in my knowledge base to answer this question.' Do NOT use information from your training data beyond what is in the context. Do NOT provide specific dosages, prescriptions, or direct treatment decisions. Always recommend consulting a qualified healthcare professional for personal medical advice."  # strict grounding instructions
EMERGENCY_RESPONSE = "This appears to describe a medical emergency. Please call emergency services (112 / 911) immediately or go to the nearest emergency room. Do not rely on an AI assistant in an emergency situation."  # emergency fallback text
UNSAFE_RESPONSE = "This question involves medication management, dosing, or medical decisions that require professional guidance. Please consult your doctor or pharmacist before making any changes to your medication or treatment plan."  # unsafe fallback text