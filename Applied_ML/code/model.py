import requests
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from config import CONFIG, EMERGENCY_KEYWORDS, UNSAFE_PATTERNS, SYSTEM_PROMPT, EMERGENCY_RESPONSE, UNSAFE_RESPONSE

def check_ollama() -> bool:  # test llm connection
    try:  # attempt ping
        resp = requests.get(f"{CONFIG['ollama_url']}/api/tags", timeout=5)  # ping server
        return resp.status_code == 200  # verify ok
    except Exception:  # catch offline
        return False  # return fail
    
def generate_response(prompt: str) -> str:  # text generation call
    try:  # attempt post
        resp = requests.post(f"{CONFIG['ollama_url']}/api/generate", json={"model": CONFIG['ollama_model'], "prompt": prompt, "stream": False, "options": {"temperature": CONFIG['temperature'], "num_predict": CONFIG['max_tokens']}}, timeout=120)  # make api call
        resp.raise_for_status()  # check status
        return resp.json().get("response", "No response generated.").strip()  # return string
    except requests.exceptions.ConnectionError:  # handle missing server
        return "Error: Ollama is not running. Start it with: ollama serve && ollama pull llama3.2"  # print error
    except Exception as e:  # handle bad generation
        return f"Error generating response: {e}"  # print error
    
class HealthcareRAG:  # main system class
    def __init__(self, collection: chromadb.Collection, embedder: SentenceTransformer):  # initialize pipeline
        self.collection = collection  # store db reference
        self.embedder = embedder  # store model reference

    def retrieve(self, query: str) -> dict:  # vector search
        q_emb = self.embedder.encode([query])[0].tolist()  # embed query string
        results = self.collection.query(query_embeddings=[q_emb], n_results=CONFIG['top_k'], include=["documents", "metadatas", "distances"])  # fetch matches
        return results  # return query data
    
    def estimate_confidence(self, results: dict) -> tuple:  # calc heuristic confidence
        distances = results.get("distances", [[]])[0]  # get raw distances
        if not distances: return 0.0, "low"  # handle empty
        similarities = [1.0 - d for d in distances]  # convert to similarity
        score = float(np.mean(similarities[:3]))  # average top 3
        if score >= CONFIG['confidence_high']: level = "high"  # flag high
        elif score >= CONFIG['confidence_medium']: level = "medium"  # flag medium
        else: level = "low"  # flag low
        return round(score, 4), level  # return score and tag
    
    def build_citations(self, results: dict) -> list:  # format sources
        citations = []  # init array
        docs = results.get("documents", [[]])[0]  # get texts
        metas = results.get("metadatas", [[]])[0]  # get metadata
        dists = results.get("distances", [[]])[0]  # get metrics
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):  # loop results
            citations.append({"id": i + 1, "source": meta.get("source", "Unknown"), "focus": meta.get("focus", ""), "type": meta.get("type", "document"), "relevance": round(1.0 - dist, 4), "excerpt": doc[:200] + "..." if len(doc) > 200 else doc})  # format dict
        return citations  # return source list
    
    def check_safety(self, query: str) -> tuple:  # lexical filter
        q_lower = query.lower()  # normalize case
        for kw in EMERGENCY_KEYWORDS:  # loop bad words
            if kw in q_lower: return False, "emergency"  # trigger emergency
        for pattern in UNSAFE_PATTERNS:  # loop bad patterns
            if pattern in q_lower: return False, "unsafe"  # trigger unsafe
        return True, ""  # safe query
    
    def build_prompt(self, query: str, results: dict) -> str:  # construct llm input
        docs = results.get("documents", [[]])[0]  # extract texts
        metas = results.get("metadatas", [[]])[0]  # extract tags
        blocks = []  # init array
        for i, (doc, meta) in enumerate(zip(docs, metas)):  # process hits
            source = meta.get("source", "Unknown")  # get source
            focus = meta.get("focus", "")  # get topic
            header = f"[{i+1}] Source: {source}" + (f" | Topic: {focus}" if focus else "")  # make string
            blocks.append(f"{header}\n{doc}")  # append context
        context = "\n\n".join(blocks)  # join all
        return f"{SYSTEM_PROMPT}\n\nRetrieved Medical Context:\n{context}\n\nUser Question: {query}\n\nAnswer:"  # return full prompt
    
    def query(self, question: str) -> dict:  # run full chain
        is_safe, safety_type = self.check_safety(question)  # filter query
        if not is_safe:  # handle unsafe
            note = EMERGENCY_RESPONSE if safety_type == "emergency" else UNSAFE_RESPONSE  # fetch warning
            return {"answer": note, "confidence": 0.0, "confidence_level": "blocked", "sources": [], "is_safe": False, "safety_note": note, "safety_type": safety_type}  # return block
        results = self.retrieve(question)  # fetch data
        confidence, conf_level = self.estimate_confidence(results)  # calculate score
        citations = self.build_citations(results)  # format sources
        prompt = self.build_prompt(question, results)  # prep text
        answer = generate_response(prompt)  # run llm
        if conf_level == "low":  # handle weak data
            answer += "\n\n⚠️ Low confidence: Limited relevant information was found in the knowledge base for this query. Please consult a healthcare professional for accurate guidance."  # inject warning
        return {"answer": answer, "confidence": confidence, "confidence_level": conf_level, "sources": citations, "is_safe": True, "safety_note": "", "safety_type": "safe"}  # return full res
    
    def print_summary(self):  # print system config
        print("\n  ARCHITECTURE: Healthcare Information Assistant (RAG)")  # print title
        print(f"  Embedding Model : {CONFIG['embed_model']} (384-dim vectors)")  # print embedder
        print(f"  LLM : {CONFIG['ollama_model']} (Ollama, local inference)")  # print llm
        print(f"  Vector Store : ChromaDB (cosine similarity)")  # print db
        print(f"  Documents Indexed : {self.collection.count():,} chunks")  # print docs
        print(f"  Top-K Retrieval : {CONFIG['top_k']}")  # print fetch count
        print(f"  Confidence High : >= {CONFIG['confidence_high']}")  # print conf
        print(f"  Confidence Med : >= {CONFIG['confidence_medium']}")  # print conf
        print(f"  Emergency KWs : {len(EMERGENCY_KEYWORDS)} patterns")  # print safety count
        print(f"  Unsafe Patterns : {len(UNSAFE_PATTERNS)} patterns")  # print safety count
        print(f"  Pipeline : Query -> Safety -> Embed -> Retrieve -> Confidence -> Prompt -> Generate -> Cite -> Response\n")  # print flow
        
        if not check_ollama():  # check backend
            print(f"  WARNING: Ollama not reachable at {CONFIG['ollama_url']}")  # print error
            print(f"  Start it with: ollama serve && ollama pull {CONFIG['ollama_model']}\n")  # print fix