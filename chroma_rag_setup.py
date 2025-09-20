import os
import json
import time
import fcntl
import shutil
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import chromadb
from chromadb.config import Settings

import google.generativeai as genai
from google.cloud import storage

import variables  # يجب أن يحتوي GEMINI_API_KEY و CHROMA_PERSIST_DIR

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
    logger.error(f"❌ Failed to connect to GCS: {e}")
    gcs_bucket = None

class _StartupLock:
    """Context manager to prevent concurrent ChromaDB startup."""
    def __enter__(self):
        self.lockfile = os.path.join(variables.CHROMA_PERSIST_DIR, ".startup.lock")
        self.fd = open(self.lockfile, "w")
        try:
            fcntl.lockf(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            raise RuntimeError("❌ ChromaDB startup already in progress (lock held).")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            fcntl.lockf(self.fd, fcntl.LOCK_UN)
        finally:
            self.fd.close()
            try:
                os.unlink(self.lockfile)
            except OSError:
                pass

class GeminiEmbeddingFunction:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.EmbeddingModel("models/embedding-001")

    def __call__(self, texts: List[str]) -> List[List[float]]:
        try:
            result = self.model.get_embeddings(texts)
            return [embedding.values for embedding in result]
        except Exception as e:
            logger.error(f"❌ Embedding error: {e}")
            raise

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

        # أسماء وأيدي المجموعات قد تختلف (gemini أو legacy)
        self.units_collection: Optional[Any] = None
        self.units_collection_name: Optional[str] = None
        self.new_launches_collection: Optional[Any] = None
        self.new_launches_collection_name: Optional[str] = None

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

        if os.listdir(self.persist_directory):
            logger.info(f"📁 Local Chroma directory not empty: {self.persist_directory}")
            return

        try:
            blobs = gcs_bucket.list_blobs(prefix=GCS_CHROMA_PREFIX)
            downloaded = 0
            for blob in blobs:
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

    def _resolve_collection_handle(
        self,
        preferred_name: str,
        fallback_name: str,
        available_names: Optional[Set[str]] = None,
    ) -> Tuple[Any, str]:
        attempts: List[str] = []
        if available_names is not None:
            if preferred_name in available_names:
                attempts.append(preferred_name)
            if fallback_name in available_names:
                attempts.append(fallback_name)
        if not attempts:
            attempts = [preferred_name, fallback_name]

        last_exc: Optional[Exception] = None
        tried: Set[str] = set()
        for name in attempts:
            if name in tried:
                continue
            tried.add(name)
            try:
                collection = self.client.get_collection(
                    name=name,
                    embedding_function=self.embedder,
                )
                if name != preferred_name:
                    logger.info(
                        "ℹ️ Falling back to legacy collection '%s' (preferred '%s' not found).",
                        name,
                        preferred_name,
                    )
                return collection, name
            except Exception as exc:
                last_exc = exc

        attempted_names = ", ".join(attempts)
        raise RuntimeError(
            "❌ Chroma collection not found locally. "
            "تأكد أن ملفات ChromaDB متاحة في المسار المحلي "
            f"('{self.persist_directory}') أو تم تنزيلها من GCS. "
            f"Tried collections: {attempted_names}."
        ) from last_exc

    def _load_or_get_collections(self):
        with _StartupLock():
            available_names: Optional[Set[str]] = None
            try:
                available_names = {col.name for col in self.client.list_collections()}
                logger.info("📚 Available Chroma collections: %s", sorted(available_names))
            except Exception as exc:
                logger.warning("⚠️ Unable to list existing Chroma collections: %s", exc)

            units_collection, units_name = self._resolve_collection_handle(
                preferred_name="real_estate_units_gemini",
                fallback_name="real_estate_units",
                available_names=available_names,
            )
            launches_collection, launches_name = self._resolve_collection_handle(
                preferred_name="new_launches_gemini",
                fallback_name="new_launches",
                available_names=available_names,
            )

            self.units_collection = units_collection
            self.units_collection_name = units_name
            self.new_launches_collection = launches_collection
            self.new_launches_collection_name = launches_name

            logger.info(
                "✅ Loaded existing collections: %s, %s",
                self.units_collection_name,
                self.new_launches_collection_name,
            )

    # ---------- Utility ----------
    @property
    def is_read_only(self) -> bool:
        return getattr(self, "_read_only", False)

    def get_collection_stats(self) -> Dict[str, Union[int, str, None]]:
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
            "units_collection_name": self.units_collection_name,
            "new_launches_collection_name": self.new_launches_collection_name,
            "units_count": units_count,
            "new_launches_count": launches_count,
            "total_count": units_count + launches_count,
        }

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
                formatted.append({"document": doc, "metadata": meta})
        return formatted

    # ---------- Read-only API ----------
    def query_units(self, query: str, n_results: int = 5) -> List[Dict]:
        if not self.units_collection:
            return []
        try:
            results = self.units_collection.query(query_texts=[query], n_results=n_results)
            return self._format_direct_results(results, n_results)
        except Exception as e:
            logger.error(f"❌ Query error (units): {e}")
            return []

    def query_new_launches(self, query: str, n_results: int = 5) -> List[Dict]:
        if not self.new_launches_collection:
            return []
        try:
            results = self.new_launches_collection.query(query_texts=[query], n_results=n_results)
            return self._format_direct_results(results, n_results)
        except Exception as e:
            logger.error(f"❌ Query error (new launches): {e}")
            return []

    def store_units_in_chroma(self, *args, **kwargs):
        raise RuntimeError("READ-ONLY MODE: store_units_in_chroma() is disabled.")

    def store_new_launches_in_chroma(self, *args, **kwargs):
        raise RuntimeError("READ-ONLY MODE: store_new_launches_in_chroma() is disabled.")

# =========================
# Singleton getter
# =========================
_rag_instance = None

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
        logger.info(
            "📊 Chroma stats — %s: %s, %s: %s, total: %s",
            rag.units_collection_name or "units",
            units_cnt,
            rag.new_launches_collection_name or "new_launches",
            launches_cnt,
            units_cnt + launches_cnt,
        )
    except Exception as e:
        logger.error(f"❌ Failed to read collection counts: {e}")

if __name__ == "__main__":
    main()
