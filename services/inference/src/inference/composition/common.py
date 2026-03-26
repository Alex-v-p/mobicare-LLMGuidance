from __future__ import annotations

from functools import lru_cache

from inference.application.pipelines.factory import build_guidance_pipeline
from inference.application.pipelines.guidance_pipeline import GuidancePipeline
from inference.application.pipelines.steps import (
    AnswerGenerator,
    ExampleResponseBuilder,
    QueryPlanner,
    QueryRewriter,
    RetrievalOrchestrator,
    ResponseVerifier,
)
from inference.embeddings.ollama_embeddings import OllamaEmbeddingsClient
from inference.indexing.document_loader import DocumentLoader
from inference.indexing.document_preparer import DocumentPreparationService
from inference.indexing.ingestion_service import IngestionService
from inference.indexing.vector_indexer import VectorIndexingService
from inference.infrastructure.http.clients.ollama_client import OllamaClient
from inference.jobstore.redis_guidance_job_store import RedisGuidanceJobStore
from inference.jobstore.redis_ingestion_job_store import RedisIngestionJobStore
from inference.retrieval.dense import DenseRetriever
from inference.retrieval.hybrid import HybridRetriever
from inference.storage.minio_documents import MinioDocumentStore
from inference.storage.minio_guidance_job_results import MinioGuidanceJobResultStore
from inference.storage.minio_ingestion_job_results import MinioIngestionJobResultStore
from inference.storage.qdrant_store import QdrantVectorStore
from shared.config import InferenceSettings, get_inference_settings as resolve_inference_settings


def get_inference_settings() -> InferenceSettings:
    return resolve_inference_settings()


@lru_cache(maxsize=1)
def get_document_store() -> MinioDocumentStore:
    return MinioDocumentStore(settings=get_inference_settings())


@lru_cache(maxsize=1)
def get_document_loader() -> DocumentLoader:
    return DocumentLoader(get_document_store())


@lru_cache(maxsize=1)
def get_embedding_client() -> OllamaEmbeddingsClient:
    return OllamaEmbeddingsClient(settings=get_inference_settings())


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(settings=get_inference_settings())


@lru_cache(maxsize=1)
def get_dense_retriever() -> DenseRetriever:
    return DenseRetriever(
        embedding_client=get_embedding_client(),
        vector_store=get_vector_store(),
        settings=get_inference_settings(),
    )


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        embedding_client=get_embedding_client(),
        vector_store=get_vector_store(),
    )


@lru_cache(maxsize=1)
def get_retrieval_orchestrator() -> RetrievalOrchestrator:
    return RetrievalOrchestrator(
        retriever=get_dense_retriever(),
        hybrid_retriever=get_hybrid_retriever(),
    )


@lru_cache(maxsize=1)
def get_query_planner() -> QueryPlanner:
    return QueryPlanner()


@lru_cache(maxsize=1)
def get_ollama_client() -> OllamaClient:
    return OllamaClient(settings=get_inference_settings())


@lru_cache(maxsize=1)
def get_query_rewriter() -> QueryRewriter:
    return QueryRewriter(get_ollama_client())


@lru_cache(maxsize=1)
def get_answer_generator() -> AnswerGenerator:
    return AnswerGenerator(get_ollama_client())


@lru_cache(maxsize=1)
def get_response_verifier() -> ResponseVerifier:
    return ResponseVerifier(get_ollama_client())


@lru_cache(maxsize=1)
def get_example_response_builder() -> ExampleResponseBuilder:
    return ExampleResponseBuilder()


@lru_cache(maxsize=1)
def get_guidance_pipeline() -> GuidancePipeline:
    return build_guidance_pipeline(
        retriever=get_dense_retriever(),
        hybrid_retriever=get_hybrid_retriever(),
        ollama_client=get_ollama_client(),
        query_planner=get_query_planner(),
        query_rewriter=get_query_rewriter(),
        retrieval_orchestrator=get_retrieval_orchestrator(),
        answer_generator=get_answer_generator(),
        response_verifier=get_response_verifier(),
        example_response_builder=get_example_response_builder(),
    )


@lru_cache(maxsize=1)
def get_document_preparer() -> DocumentPreparationService:
    return DocumentPreparationService()


@lru_cache(maxsize=1)
def get_vector_indexer() -> VectorIndexingService:
    return VectorIndexingService(
        embedding_client=get_embedding_client(),
        vector_store=get_vector_store(),
    )


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        document_store=get_document_store(),
        document_loader=get_document_loader(),
        embedding_client=get_embedding_client(),
        vector_store=get_vector_store(),
        document_preparer=get_document_preparer(),
        vector_indexer=get_vector_indexer(),
    )


@lru_cache(maxsize=1)
def get_guidance_job_result_store() -> MinioGuidanceJobResultStore:
    return MinioGuidanceJobResultStore(settings=get_inference_settings())


@lru_cache(maxsize=1)
def get_ingestion_job_result_store() -> MinioIngestionJobResultStore:
    return MinioIngestionJobResultStore(settings=get_inference_settings())


def get_guidance_job_store() -> RedisGuidanceJobStore:
    return RedisGuidanceJobStore(settings=get_inference_settings())


def get_ingestion_job_store() -> RedisIngestionJobStore:
    return RedisIngestionJobStore(settings=get_inference_settings())
