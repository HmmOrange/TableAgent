"""Stable public boundaries for external TableAgent integrations."""

from .models import OpenAICompatibleLLM, create_model_client
from .qa import TableQAEngine, TableQARequest, TableQAResponse
from .qa_package import (
    QA_PACKAGE_SCHEMA_VERSION,
    TableQAPackage,
    load_qa_request,
    qa_response_payload,
)

__all__ = [
    "OpenAICompatibleLLM",
    "QA_PACKAGE_SCHEMA_VERSION",
    "TableQAEngine",
    "TableQAPackage",
    "TableQARequest",
    "TableQAResponse",
    "create_model_client",
    "load_qa_request",
    "qa_response_payload",
]
