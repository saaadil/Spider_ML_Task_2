import sys
from pathlib import Path
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from config import DATA_DIR, KB_DIR, CHROMA_DIR, CONFIG

def print_download_instructions():  # handle missing data gracefully
    print("\n  MedQuAD DATASET NOT FOUND")  # print error header
    print(f"  Expected path: {DATA_DIR / 'medquad.csv'}")  # show expected path
    print("\n  Download steps:")  # list manual instructions
    print("  1. Go to https://www.kaggle.com/datasets/gpreda/medquad")  # provide link
    print("  2. Click Download")  # instruct user
    print("  3. Extract and rename the CSV to 'medquad.csv'")  # explain rename
    print(f"  4. Place it at: {DATA_DIR / 'medquad.csv'}")  # confirm final location
    print("\n  Optional: add WHO/CDC/NICE guideline PDFs to:")  # note optional docs
    print(f"  {KB_DIR}")  # print kb dir

def load_medquad(csv_path: Path) -> list:  # parse primary dataset
    print("[Ingest] Loading MedQuAD ...")  # log start
    df = pd.read_csv(csv_path)  # load pandas dataframe
    df.columns = [c.lower().strip() for c in df.columns]  # normalize column names
    q_col = next((c for c in df.columns if 'question' in c), None)  # find question column
    a_col = next((c for c in df.columns if 'answer' in c), None)  # find answer column
    f_col = next((c for c in df.columns if 'focus' in c or 'topic' in c), None)  # find topic column
    if q_col is None or a_col is None:  # check columns exist
        raise ValueError(f"Cannot find question/answer columns. Found: {list(df.columns)}")  # throw failure
    documents = []  # init document array
    for _, row in df.iterrows():  # iterate over dataframe
        q = str(row[q_col]).strip()  # clean question text
        a = str(row[a_col]).strip()  # clean answer text
        f = str(row[f_col]).strip() if f_col and pd.notna(row.get(f_col, None)) else ''  # extract focus cleanly
        if a in ('nan', '') or len(a) < 15: continue  # discard empty answers
        if q in ('nan', '') or len(q) < 5: continue  # discard empty questions
        content = f"Question: {q}\nAnswer: {a}"  # format raw content
        documents.append({'content': content, 'metadata': {'source': 'MedQuAD', 'focus': f, 'type': 'qa_pair'}})  # append with metadata
    print(f"[Ingest] Loaded {len(documents):,} QA pairs from MedQuAD")  # log count
    return documents  # return raw documents

def load_pdf_documents(pdf_dir: Path) -> list:  # process optional pdfs
    try:  # test import
        import pypdf
    except ImportError:  # handle missing library
        print("[Ingest] pypdf not installed - skipping PDF loading")  # print skip warning
        return []  # return empty list
    documents = []  # init array
    pdf_files = list(pdf_dir.glob("*.pdf"))  # scan for pdfs
    if not pdf_files:  # check if empty
        print(f"[Ingest] No PDFs found in {pdf_dir} - using MedQuAD only")  # print fallback warning
        return []  # return empty
    for pdf_path in pdf_files:  # loop through files
        try:  # attempt extraction
            reader = pypdf.PdfReader(str(pdf_path))  # initialize pdf reader
            text = "\n".join(page.extract_text() or "" for page in reader.pages)  # extract all pages
            source_name = pdf_path.stem.replace("_", " ").title()  # generate clean name
            chunks = _chunk_text(text)  # chunk raw text
            for chunk in chunks:  # loop parsed chunks
                documents.append({'content': chunk, 'metadata': {'source': source_name, 'focus': '', 'type': 'guideline'}})  # append document
            print(f"[Ingest] Loaded {len(chunks)} chunks from {pdf_path.name}")  # log success
        except Exception as e:  # handle parsing errors
            print(f"[Ingest] Failed to load {pdf_path.name}: {e}")  # print failure
    return documents  # return parsed pdfs

def _chunk_text(text: str) -> list:  # text slicer
    size = CONFIG['chunk_size']  # grab target size
    overlap = CONFIG['chunk_overlap']  # grab overlap size
    text = text.strip()  # clean string
    if not text: return []  # handle empty
    if len(text) <= size: return [text]  # handle short text
    chunks = []  # init array
    start = 0  # set start
    while start < len(text):  # loop until done
        end = min(start + size, len(text))  # find bound
        chunk = text[start:end].strip()  # slice string
        if len(chunk) > 30: chunks.append(chunk)  # keep valid size
        if end == len(text): break  # exit early
        start += size - overlap  # step forward
    return chunks  # return string array

def chunk_documents(documents: list) -> list:  # chunk full corpus
    chunks = []  # init array
    for doc in documents:  # loop items
        parts = _chunk_text(doc['content'])  # slice into parts
        for part in parts:  # loop parts
            chunks.append({'content': part, 'metadata': doc['metadata']})  # store pieces
    print(f"[Ingest] {len(documents):,} documents -> {len(chunks):,} chunks")  # log count
    return chunks  # return all chunks

def build_vectorstore(chunks: list, embedder: SentenceTransformer, client: chromadb.PersistentClient) -> chromadb.Collection:  # create vector db
    collection = client.get_or_create_collection(name=CONFIG['collection_name'], metadata={"hnsw:space": "cosine"})  # init collection
    texts = [c['content'] for c in chunks]  # map text
    metas = [c['metadata'] for c in chunks]  # map metadata
    ids = [f"chunk_{i}" for i in range(len(chunks))]  # generate ids
    bs = CONFIG['batch_size']  # get batch size
    print(f"[Ingest] Embedding {len(chunks):,} chunks (batch={bs}) ...")  # log start
    for i in range(0, len(chunks), bs):  # loop batches
        b_texts = texts[i : i + bs]  # slice texts
        b_metas = metas[i : i + bs]  # slice metadata
        b_ids = ids[i : i + bs]  # slice ids
        b_embs = embedder.encode(b_texts, show_progress_bar=False).tolist()  # run embedder
        collection.add(ids=b_ids, embeddings=b_embs, documents=b_texts, metadatas=b_metas)  # add batch
        done = min(i + bs, len(chunks))  # calculate progress
        if done % (bs * 20) == 0 or done == len(chunks):  # log interval
            print(f"  [{done:,}/{len(chunks):,}] chunks indexed")  # print progress
    print(f"[Ingest] Indexed {collection.count():,} chunks into ChromaDB")  # print final
    return collection  # return db

def get_collection() -> chromadb.Collection:  # load or build db
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))  # init client
    try:  # try loading
        collection = client.get_collection(CONFIG['collection_name'])  # fetch collection
        count = collection.count()  # count items
        if count > 0:  # verify filled
            print(f"[Index] Loaded existing collection ({count:,} chunks)")  # log success
            return collection  # return db
    except Exception:  # catch missing
        pass  # proceed to build
    csv_path = DATA_DIR / "medquad.csv"  # set path
    if not csv_path.exists():  # verify existence
        print_download_instructions()  # print reqs
        sys.exit(1)  # exit app
    embedder = SentenceTransformer(CONFIG['embed_model'])  # load embedder
    documents = load_medquad(csv_path)  # load core data
    pdf_docs = load_pdf_documents(KB_DIR)  # load pdfs
    documents.extend(pdf_docs)  # combine arrays
    chunks = chunk_documents(documents)  # chunk them all
    return build_vectorstore(chunks, embedder, client)  # build and return