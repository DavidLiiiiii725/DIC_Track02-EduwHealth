from pathlib import Path
import re
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

RAW_FILE = Path(r"C:\Users\Xudon\Desktop\agent\data\kb.txt")
OUT_DIR = Path("data")

MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384

# ✅ 控制内存的关键参数
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP = 200
ENCODE_BATCH_SIZE = 32       # 8/16/32 视内存调整
ADD_BATCH_CHUNKS = 512      # 每次处理多少个 chunk（越小越省内存）
NORMALIZE = True            # normalize embeddings（便于相似度更稳定）

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200):
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks

def load_or_create_index(index_path: Path):
    if index_path.exists():
        return faiss.read_index(str(index_path))
    return faiss.IndexFlatL2(DIM)

def count_existing_items(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0
    # 统计已写入条数（用于断点续跑）
    n = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n

def main():
    print(torch.cuda.is_available())
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw KB file not found: {RAW_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUT_DIR / "vector.index"
    jsonl_path = OUT_DIR / "vector_texts.jsonl"

    print("[1/5] Loading raw KB...")
    text = RAW_FILE.read_text(encoding="utf-8")
    chunks = chunk_text(text, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP)
    print(f"[2/5] Total chunks: {len(chunks)}")

    print("[3/5] Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME,device="cuda")

    print("[4/5] Loading/creating FAISS index...")
    index = load_or_create_index(index_path)

    #断点续跑：如果 jsonl 已有 N 行，则跳过前 N 个 chunks
    already = count_existing_items(jsonl_path)
    if already > 0:
        print(f"[RESUME] Detected {already} existing items in {jsonl_path.name}.")
        if already >= len(chunks):
            print("[DONE] Nothing to add.")
            return
        chunks_to_add = chunks[already:]
        start_id = already
    else:
        chunks_to_add = chunks
        start_id = 0

    print(f"[5/5] Encoding & adding to index (chunks to add: {len(chunks_to_add)})...")

    with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
        for base in range(0, len(chunks_to_add), ADD_BATCH_CHUNKS):
            batch_chunks = chunks_to_add[base: base + ADD_BATCH_CHUNKS]
            batch_ids = range(start_id + base, start_id + base + len(batch_chunks))

            #分批 encode
            emb = model.encode(
                batch_chunks,
                batch_size=ENCODE_BATCH_SIZE,
                normalize_embeddings=NORMALIZE,
                show_progress_bar=True
            )
            emb = np.array(emb, dtype=np.float32)

            index.add(emb)
            del emb
            torch.cuda.empty_cache()

            for cid, ch in zip(batch_ids, batch_chunks):
                obj = {"text": ch, "meta": {"source": RAW_FILE.name, "chunk_id": cid}}
                f_jsonl.write(json.dumps(obj, ensure_ascii=False) + "\n")

            #每批落盘
            faiss.write_index(index, str(index_path))

            done = base + len(batch_chunks)
            print(f"  - Added {done}/{len(chunks_to_add)} new chunks (total indexed: {index.ntotal})")

    print("\n[OK] Vector KB built/updated successfully.")
    print(f"- index:  {index_path}")
    print(f"- texts:  {jsonl_path}")
    print(f"- total vectors: {index.ntotal}")

if __name__ == "__main__":
    main()
