#!/usr/bin/env python3
"""
Module 3: RAG Corpus Ingestion Pipeline
Reads all corpus files, chunks them appropriately per category,
embeds using sentence-transformers, and stores in ChromaDB.
"""

import os
import re
import csv
import sys
import json
import fitz  # PyMuPDF
import docx
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CORPUS_BASE = Path("/home/ubuntu/ai-health-orchestration/data/corpus")

FOLDERS = {
    "public_health_recommendations": CORPUS_BASE / "public_health_recommendations",
    "nutrition_guidelines":          CORPUS_BASE / "nutrition_guidelines",
    "gym_programming":               CORPUS_BASE / "gym_programming",
    "food_and_recipes":              CORPUS_BASE / "food_and_recipes",
}

CHROMA_HOST = "localhost"
CHROMA_PORT = 8001

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking settings for prose documents
CHUNK_SIZE    = 400   # tokens (approximate via words)
CHUNK_OVERLAP = 60    # ~15% overlap


# ─────────────────────────────────────────────
# HELPERS — TEXT EXTRACTION
# ─────────────────────────────────────────────

def extract_pdf(path: Path) -> str:
    """Extract all text from a PDF file."""
    text = []
    try:
        doc = fitz.open(str(path))
        for page in doc:
            text.append(page.get_text())
        doc.close()
    except Exception as e:
        print(f"  [WARN] Could not read PDF {path.name}: {e}")
    return "\n".join(text).strip()


def extract_docx(path: Path) -> str:
    """Extract all text from a DOCX file."""
    try:
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        print(f"  [WARN] Could not read DOCX {path.name}: {e}")
        return ""


def extract_csv_as_text(path: Path) -> str:
    """Convert a CSV file to plain text rows."""
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
        lines = []
        for _, row in df.iterrows():
            line = ", ".join(f"{col}: {val}" for col, val in row.items()
                             if pd.notna(val) and str(val).strip())
            if line:
                lines.append(line)
        return "\n".join(lines)
    except Exception as e:
        print(f"  [WARN] Could not read CSV {path.name}: {e}")
        return ""


# ─────────────────────────────────────────────
# HELPERS — CHUNKING
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Used for prose documents (PDFs, DOCX).
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def chunk_by_paragraph(text: str, min_words: int = 30) -> list[str]:
    """
    Split text by double newlines (paragraph boundaries).
    Merges short paragraphs with the next one.
    Used for structured reference documents like lab ranges.
    """
    raw = re.split(r"\n{2,}", text)
    chunks = []
    buffer = ""
    for para in raw:
        para = para.strip()
        if not para:
            continue
        buffer = (buffer + " " + para).strip() if buffer else para
        if len(buffer.split()) >= min_words:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


# ─────────────────────────────────────────────
# HELPERS — USDA CSV PROCESSING
# ─────────────────────────────────────────────

def build_usda_nutrient_lookup(base_dir: Path) -> dict:
    """
    Build a lookup dict: fdc_id -> {nutrient_name: amount, ...}
    from food_nutrient.csv and nutrient.csv.
    """
    nutrient_names = {}
    nutrient_csv = base_dir / "nutrient.csv"
    if nutrient_csv.exists():
        df = pd.read_csv(nutrient_csv, on_bad_lines="skip")
        for _, row in df.iterrows():
            nutrient_names[str(row.get("id", ""))] = str(row.get("name", ""))

    lookup = {}
    food_nutrient_csv = base_dir / "food_nutrient.csv"
    if food_nutrient_csv.exists():
        df = pd.read_csv(food_nutrient_csv, on_bad_lines="skip",
                         usecols=["fdc_id", "nutrient_id", "amount"])
        for _, row in df.iterrows():
            fdc_id = str(row["fdc_id"])
            n_id   = str(row["nutrient_id"])
            amount = row["amount"]
            name   = nutrient_names.get(n_id, n_id)
            if fdc_id not in lookup:
                lookup[fdc_id] = {}
            lookup[fdc_id][name] = amount
    return lookup


# Key nutrients we care about for health/meal planning
KEY_NUTRIENTS = [
    "Protein", "Total lipid (fat)", "Carbohydrate, by difference",
    "Energy", "Fiber, total dietary", "Sugars, total including NLEA",
    "Calcium, Ca", "Iron, Fe", "Sodium, Na", "Vitamin C, total ascorbic acid",
    "Vitamin D (D2 + D3)", "Fatty acids, total saturated",
    "Fatty acids, total trans", "Cholesterol"
]


def build_usda_chunks(base_dir: Path, source_label: str) -> list[dict]:
    """
    Build one text chunk per food item with its key nutrients.
    Returns list of {text, metadata} dicts.
    """
    food_csv = base_dir / "food.csv"
    if not food_csv.exists():
        print(f"  [WARN] food.csv not found in {base_dir}")
        return []

    print(f"  Building nutrient lookup for {source_label}...")
    nutrient_lookup = build_usda_nutrient_lookup(base_dir)

    df_food = pd.read_csv(food_csv, on_bad_lines="skip",
                          usecols=["fdc_id", "description"])
    chunks = []

    for _, row in tqdm(df_food.iterrows(), total=len(df_food),
                       desc=f"  {source_label}"):
        fdc_id      = str(row["fdc_id"])
        description = str(row.get("description", "")).strip()
        if not description:
            continue

        nutrients = nutrient_lookup.get(fdc_id, {})
        nutrient_lines = []
        for key in KEY_NUTRIENTS:
            if key in nutrients and pd.notna(nutrients[key]):
                nutrient_lines.append(f"{key}: {nutrients[key]:.2f}")

        if not nutrient_lines:
            continue  # skip foods with no nutrient data

        text = (
            f"Food: {description}\n"
            f"USDA FDC ID: {fdc_id}\n"
            f"Nutrients per 100g:\n" +
            "\n".join(nutrient_lines)
        )
        chunks.append({
            "text": text,
            "metadata": {
                "source": source_label,
                "food_name": description,
                "fdc_id": fdc_id,
                "category": "food_and_recipes",
                "chunk_type": "usda_food_nutrient"
            }
        })
    return chunks


def build_recipe_chunks(csv_path: Path) -> list[dict]:
    """
    Build one chunk per recipe row from All_Diets.csv.
    """
    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip")
    except Exception as e:
        print(f"  [WARN] Could not read {csv_path.name}: {e}")
        return []

    chunks = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Recipes"):
        parts = []
        for col in df.columns:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                parts.append(f"{col}: {val}")
        if not parts:
            continue

        recipe_name = str(row.get("Recipe_name",
                          row.get("title",
                          row.get("name", "Unknown Recipe")))).strip()
        text = "\n".join(parts)
        chunks.append({
            "text": text,
            "metadata": {
                "source": csv_path.name,
                "recipe_name": recipe_name,
                "category": "food_and_recipes",
                "chunk_type": "recipe"
            }
        })
    return chunks


# ─────────────────────────────────────────────
# CORE — PROCESS EACH CATEGORY
# ─────────────────────────────────────────────

def process_prose_folder(folder: Path, category: str,
                         use_paragraph_chunks: bool = False) -> list[dict]:
    """
    Process a folder of PDFs and DOCX files into chunks.
    Returns list of {text, metadata} dicts.
    """
    all_chunks = []
    files = sorted(folder.glob("*"))

    for path in files:
        if path.suffix.lower() == ".pdf":
            print(f"  Reading PDF: {path.name}")
            text = extract_pdf(path)
        elif path.suffix.lower() == ".docx":
            print(f"  Reading DOCX: {path.name}")
            text = extract_docx(path)
        else:
            continue  # skip .gitkeep, .zip etc

        if not text.strip():
            print(f"  [WARN] Empty text from {path.name}, skipping")
            continue

        if use_paragraph_chunks:
            chunks = chunk_by_paragraph(text)
        else:
            chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": path.name,
                    "category": category,
                    "chunk_index": i,
                    "chunk_type": "prose",
                    "total_chunks": len(chunks)
                }
            })

    print(f"  → {len(all_chunks)} chunks from {category}")
    return all_chunks


def process_food_folder(folder: Path) -> list[dict]:
    """Process all food & recipe data sources."""
    all_chunks = []

    # All_Diets.csv — recipe dataset
    recipe_csv = folder / "All_Diets.csv"
    if recipe_csv.exists():
        print(f"  Reading recipe CSV: All_Diets.csv")
        all_chunks.extend(build_recipe_chunks(recipe_csv))

    # USDA Foundation Foods
    foundation_dir = (folder / "usda_foundation" /
                      "FoodData_Central_foundation_food_csv_2025-12-18")
    if foundation_dir.exists():
        print(f"  Reading USDA Foundation Foods...")
        all_chunks.extend(build_usda_chunks(foundation_dir, "USDA_Foundation"))

    # USDA SR Legacy
    sr_legacy_dir = (folder / "usda_sr_legacy" /
                     "FoodData_Central_sr_legacy_food_csv_2018-04")
    if sr_legacy_dir.exists():
        print(f"  Reading USDA SR Legacy...")
        all_chunks.extend(build_usda_chunks(sr_legacy_dir, "USDA_SR_Legacy"))

    print(f"  → {len(all_chunks)} chunks from food_and_recipes")
    return all_chunks


# ─────────────────────────────────────────────
# CORE — INGEST INTO CHROMADB
# ─────────────────────────────────────────────

def ingest_to_chromadb(collection_name: str, chunks: list[dict],
                       embedder: SentenceTransformer,
                       client: chromadb.HttpClient,
                       batch_size: int = 100):
    """
    Embed chunks and upsert into a ChromaDB collection.
    """
    if not chunks:
        print(f"  [SKIP] No chunks for collection: {collection_name}")
        return

    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    texts     = [c["text"]     for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids       = [f"{collection_name}_{i}" for i in range(len(chunks))]

    print(f"  Embedding {len(texts)} chunks for '{collection_name}'...")

    for start in tqdm(range(0, len(texts), batch_size),
                      desc=f"  Upserting {collection_name}"):
        end        = start + batch_size
        batch_txt  = texts[start:end]
        batch_meta = metadatas[start:end]
        batch_ids  = ids[start:end]

        embeddings = embedder.encode(batch_txt, show_progress_bar=False).tolist()

        collection.upsert(
            ids=batch_ids,
            documents=batch_txt,
            embeddings=embeddings,
            metadatas=batch_meta
        )

    final_count = collection.count()
    print(f"  ✓ '{collection_name}' — {final_count} documents in ChromaDB")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("AI Health Orchestration — RAG Ingestion Pipeline")
    print("=" * 60)

    # Connect to ChromaDB
    print("\n[1/6] Connecting to ChromaDB...")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    print(f"  ✓ Connected — ChromaDB heartbeat OK")

    # Load embedding model
    print("\n[2/6] Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  ✓ Model loaded: {EMBEDDING_MODEL}")

    # ── Public Health Recommendations ──────────────
    print("\n[3/6] Processing: public_health_recommendations")
    ph_chunks = process_prose_folder(
        FOLDERS["public_health_recommendations"],
        category="public_health_recommendations",
        use_paragraph_chunks=True  # lab reference docs chunk better by paragraph
    )
    ingest_to_chromadb("public_health_recommendations", ph_chunks,
                       embedder, client)

    # ── Nutrition Guidelines ────────────────────────
    print("\n[4/6] Processing: nutrition_guidelines")
    ng_chunks = process_prose_folder(
        FOLDERS["nutrition_guidelines"],
        category="nutrition_guidelines",
        use_paragraph_chunks=False  # prose research papers use sliding window
    )
    ingest_to_chromadb("nutrition_guidelines", ng_chunks, embedder, client)

    # ── Gym Programming ─────────────────────────────
    print("\n[5/6] Processing: gym_programming")
    gp_chunks = process_prose_folder(
        FOLDERS["gym_programming"],
        category="gym_programming",
        use_paragraph_chunks=False
    )
    ingest_to_chromadb("gym_programming", gp_chunks, embedder, client)

    # ── Food & Recipes ──────────────────────────────
    print("\n[6/6] Processing: food_and_recipes")
    fr_chunks = process_food_folder(FOLDERS["food_and_recipes"])
    ingest_to_chromadb("food_and_recipes", fr_chunks, embedder, client)

    # ── Summary ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE — Collection Summary")
    print("=" * 60)
    for name in ["public_health_recommendations", "nutrition_guidelines",
                 "gym_programming", "food_and_recipes"]:
        col   = client.get_or_create_collection(name)
        count = col.count()
        print(f"  {name}: {count} documents")
    print("=" * 60)


if __name__ == "__main__":
    main()