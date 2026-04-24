import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

MODEL = SentenceTransformer("microsoft/codebert-base")
CLIENT = chromadb.Client()
COLLECTION = CLIENT.get_or_create_collection("codebase")

def chunk_file(filepath, chunk_size=50, overlap=10):
    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        chunk_text = "".join(lines[start:end])
        chunks.append({
            "text": chunk_text,
            "filename": filepath,
            "start_line": start + 1,
            "end_line": end
        })
        start += chunk_size - overlap

    return chunks

def index_codebase(repo_path):
    print(f"\nIndexing codebase at: {repo_path}")
    all_chunks = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in [".git", "venv", "__pycache__", "node_modules"]]
        for filename in files:
            if filename.endswith((".py", ".js", ".ts", ".java", ".md")):
                filepath = os.path.join(root, filename)
                chunks = chunk_file(filepath)
                all_chunks.extend(chunks)
                print(f"  Indexed: {filename} ({len(chunks)} chunks)")

    if not all_chunks:
        print("No files found to index.")
        return

    texts = [c["text"] for c in all_chunks]
    embeddings = MODEL.encode(texts).tolist()

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    metadatas = [{"filename": c["filename"], "start_line": c["start_line"], "end_line": c["end_line"]} for c in all_chunks]

    COLLECTION.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"\nTotal chunks indexed: {len(all_chunks)}")

def retrieve_context(diff_text, n_results=3):
    query_embedding = MODEL.encode([diff_text]).tolist()

    results = COLLECTION.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    context_chunks = []
    for i in range(len(results["documents"][0])):
        context_chunks.append({
            "text": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"],
            "start_line": results["metadatas"][0][i]["start_line"]
        })

    return context_chunks