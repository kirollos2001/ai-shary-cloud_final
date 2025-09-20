import os
import json
import time
import fcntl
import shutil
import logging
import sqlite3
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings

import google.generativeai as genai
from google.cloud import storage

import variables  # لازم يحتوي GEMINI_API_KEY و CHROMA_PERSIST_DIR

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Google Cloud Storage (GCS)
# =========================
GCS_BUCKET_NAME = "sharyai2025-cache"
GCS_CHROMA_PREFIX = "chroma/"  # المسار داخل الباكِت حيث ملفات ChromaDB

try:
    gcs_client = storage.Client()
    gcs_bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    logger.info(f"✅ Connected to GCS bucket: {GCS_BUCKET_NAME}")
except Exception as e:
    logger.warning(f"⚠️ Failed to initialize GCS client: {e}. Using local ChromaDB only.")
    gcs_bucket = None

# =========================
# Gemini API (Embeddings)
# =========================
os.environ["GEMINI_API_KEY"] = variables.GEMINI_API_KEY
genai.configure(api_key=variables.GEMINI_API_KEY)

class GeminiEmbeddingFunction:
    """Custom embedding function using Gemini 3072-dim embeddings (read-only querying)."""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

    def name(self) -> str:
        return "gemini-embedding-001"

    def __call__(self, inputs: List[str]) -> List[List[float]]:
        if not inputs:
            return []
        out: List[List[float]] = []
        for text in inputs:
            try:
                result = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=str(text),
                    task_type="SEMANTIC_SIMILARITY",
                )
                vec = result["embedding"]
            except Exception as e:
                logger.error(f"❌ Error generating embedding: {e}")
                vec = [0.0] * 3072
            out.append(vec)
        return out

    def embed(self, text: str) -> List[float]:
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=str(text),
                task_type="SEMANTIC_SIMILARITY",
            )
            return result["embedding"]
        except Exception as e:
            logger.error(f"❌ Error generating single embedding: {e}")
            return [0.0] * 3072

# =========================
# Startup file lock
# =========================
CHROMA_INIT_LOCK = os.getenv("CHROMA_INIT_LOCK", "/tmp/chroma_init.lock")

class _StartupLock:
    def __enter__(self):
        os.makedirs(os.path.dirname(CHROMA_INIT_LOCK), exist_ok=True)
        self._f = open(CHROMA_INIT_LOCK, "w")
        fcntl.flock(self._f, fcntl.LOCK_EX)
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            fcntl.flock(self._f, fcntl.LOCK_UN)
            self._f.close()
        except Exception:
            pass

# =========================
# RAG (READ-ONLY)
# =========================
_rag_instance = None

class RealEstateRAG:
    """
    READ-ONLY Chroma client:
      - لا إنشاء لمجموعات
      - لا إدخال/إضافة داتا
      - تحميل ملفّات ChromaDB من GCS لو المجلد المحلي فاضي
    """

    def __init__(self, persist_directory: str = variables.CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        # نزّل ChromaDB من GCS لمرة أولى لو المجلد فاضي
        self._download_chroma_from_gcs_if_empty()

        self.embedder = GeminiEmbeddingFunction(variables.GEMINI_API_KEY)

        chroma_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=False,                   # مهم: لا نسمح بالـ reset تلقائي
            persist_directory=self.persist_directory,
        )
        self._chroma_settings = chroma_settings

        # Persistent client على الديركتوري المحلي
        self.client = self._create_client_readonly(chroma_settings)

        # حمّل collections الموجودة فقط (بدون create)
        self._load_or_get_collections()
        self._read_only = True
        logger.info("✅ ChromaDB loaded with existing collections (READ-ONLY).")

    # ---------- GCS helpers ----------
    def _download_from_gcs(self, gcs_path: str, local_path: str) -> bool:
        if not gcs_bucket:
            return False
        try:
            blob = gcs_bucket.blob(gcs_path)
            # في بعض الإصدارات blob.exists(client)؛ غالبًا .exists() تكفي
            if blob.exists():
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                blob.download_to_filename(local_path)
                logger.info(f"⬇️  Downloaded {gcs_path} -> {local_path} ({os.path.getsize(local_path)} bytes)")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to download {gcs_path}: {e}")
            return False

    def _download_chroma_from_gcs_if_empty(self):
        if not gcs_bucket:
            logger.info("GCS not configured; using local ChromaDB if present.")
            return

        # لو فيه ملفات محلّياً، ما ننزّلش
        if os.path.isdir(self.persist_directory) and os.listdir(self.persist_directory):
            logger.info(f"Local Chroma dir '{self.persist_directory}' already has files. Skipping GCS download.")
            return

        try:
            blobs = list(gcs_bucket.list_blobs(prefix=GCS_CHROMA_PREFIX))
            if not blobs:
                logger.warning(f"⚠️ No Chroma files found in gs://{GCS_BUCKET_NAME}/{GCS_CHROMA_PREFIX}")
                return

            downloaded = 0
            for blob in blobs:
                # تجاهل “الدلائل” الوهمية
                if blob.name.endswith("/"):
                    continue
                rel = os.path.relpath(blob.name, GCS_CHROMA_PREFIX)
                local_path = os.path.join(self.persist_directory, rel)
                if self._download_from_gcs(blob.name, local_path):
                    downloaded += 1

            if downloaded:
                logger.info(f"✅ Downloaded {downloaded} Chroma files from gs://{GCS_BUCKET_NAME}/{GCS_CHROMA_PREFIX} to {self.persist_directory}")
            else:
                logger.warning("⚠️ Found blobs but none downloaded (permissions or path mismatch).")
        except Exception as e:
            logger.error(f"❌ Error listing/downloading Chroma from GCS: {e}")

    # ---------- Client & Collections (read-only) ----------
    def _create_client_readonly(self, chroma_settings: Settings):
        with _StartupLock():
            try:
                return chromadb.PersistentClient(path=self.persist_directory, settings=chroma_settings)
            except Exception as e:
                msg = str(e).lower()
                # ما نعملش reset تلقائي عشان ما نمسحش الداتا
                raise RuntimeError(
                    f"❌ Failed to open Chroma persistent client at '{self.persist_directory}'. "
                    f"Make sure Chroma files exist locally or in GCS. Details: {e}"
                )

    def _load_or_get_collections(self):
        with _StartupLock():
            try:
                self.units_collection = self.client.get_collection(
                    name="real_estate_units",
                    embedding_function=self.embedder,
                )
                self.new_launches_collection = self.client.get_collection(
                    name="new_launches",
                    embedding_function=self.embedder,
                )
                logger.info("✅ Loaded existing collections: real_estate_units, new_launches")
            except Exception as e:
                # متعمّد: لا نعمل create هنا نهائيًا
                raise RuntimeError(
                    "❌ Chroma collections not found locally. "
                    "تأكد أن ملفات ChromaDB متاحة في المسار المحلي "
                    f"('{self.persist_directory}') أو تم تنزيلها من GCS."
                ) from e
    # ---------- Utility ----------
    @property
    def is_read_only(self) -> bool:
        return getattr(self, "_read_only", False)

    def get_collection_stats(self) -> Dict[str, int]:
        """Return basic statistics about the loaded collections."""
        try:
            units_count = self.units_collection.count() if self.units_collection else 0
        except Exception as exc:
            logger.error(f"❌ Failed to count units collection: {exc}")
            units_count = 0

        try:
            launches_count = (
                self.new_launches_collection.count() if self.new_launches_collection else 0
            )
        except Exception as exc:
            logger.error(f"❌ Failed to count new launches collection: {exc}")
            launches_count = 0

        return {
            "units_count": units_count,
            "new_launches_count": launches_count,
            "total_count": units_count + launches_count,
        }
    # ---------- Utility ----------
    def _parse_int(self, value, default=None):
        try:
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return int(value)
            import re
            m = re.search(r"\d+", str(value))
            if m:
                return int(m.group(0))
            return default
        except Exception:
            return default

    def _format_direct_results(self, results: Dict, n_results: int) -> List[Dict]:
        formatted = []
        if results and results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0][:n_results]):
                meta = {}
                if results.get("metadatas") and results["metadatas"][0] and i < len(results["metadatas"][0]):
                    meta = results["metadatas"][0][i]
                dist = None
                if results.get("distances") and results["distances"][0] and i < len(results["distances"][0]):
                    dist = results["distances"][0][i]
                formatted.append({"document": doc, "metadata": meta, "distance": dist})
        return formatted

    # ---------- Search (read-only) ----------
    def search_units(self, query: str, n_results: int = 20, filters: Dict = None) -> List[Dict]:
        start = time.time()
        where = None
        if filters:
            clauses = []
            if filters.get("price_min") is not None:
                clauses.append({"price_value": {"$gte": filters["price_min"]}})
            if filters.get("price_max") is not None:
                clauses.append({"price_value": {"$lte": filters["price_max"]}})
            if clauses:
                where = clauses[0] if len(clauses) == 1 else {"$and": clauses}

        try:
            res = self.units_collection.query(
                query_texts=[query],
                n_results=100,
                where=where,
                include=["metadatas", "distances", "documents"],
            )
            out = self._format_direct_results(res, n_results)
            logger.info(f"✅ Units search returned {len(out)} items in {time.time()-start:.2f}s")
            return out
        except Exception as e:
            logger.error(f"❌ Error searching units: {e}")
            return []

    def search_new_launches(self, query: str, n_results: int = 10, filters: Dict = None) -> List[Dict]:
        start = time.time()
        where = None
        if filters:
            clauses = []
            if filters.get("price_min") is not None:
                clauses.append({"price_value": {"$gte": filters["price_min"]}})
            if filters.get("price_max") is not None:
                clauses.append({"price_value": {"$lte": filters["price_max"]}})
            if clauses:
                where = clauses[0] if len(clauses) == 1 else {"$and": clauses}

        try:
            res = self.new_launches_collection.query(
                query_texts=[query],
                n_results=100,
                where=where,
                include=["metadatas", "distances", "documents"],
            )
            out = self._format_direct_results(res, n_results)
            logger.info(f"✅ New launches search returned {len(out)} items in {time.time()-start:.2f}s")
            return out
        except Exception as e:
            logger.error(f"❌ Error searching new launches: {e}")
            return []

    # ---------- Block any write/creation methods ----------
    def reset_collections(self):
        raise RuntimeError("READ-ONLY MODE: reset_collections() is disabled.")

    def store_units_in_chroma(self, *args, **kwargs):
        raise RuntimeError("READ-ONLY MODE: store_units_in_chroma() is disabled.")

    def store_new_launches_in_chroma(self, *args, **kwargs):
        raise RuntimeError("READ-ONLY MODE: store_new_launches_in_chroma() is disabled.")

# =========================
# Singleton getter
# =========================
def get_rag_instance() -> "RealEstateRAG":
    global _rag_instance
    if _rag_instance is None:
        with _StartupLock():
            if _rag_instance is None:
                _rag_instance = RealEstateRAG()
    return _rag_instance

# =========================
# CLI main (read-only)
# =========================
def main():
    logger.info("🚀 Starting ChromaDB RAG (READ-ONLY)...")
    rag = RealEstateRAG()

    # مثال سريع للتأكد من التحميل:
    try:
        units_cnt = rag.units_collection.count()
        launches_cnt = rag.new_launches_collection.count()
        logger.info(f"📊 Chroma stats — units: {units_cnt}, new_launches: {launches_cnt}, total: {units_cnt + launches_cnt}")
    except Exception as e:
        logger.error(f"❌ Failed to read collection counts: {e}")

if __name__ == "__main__":
    main()
