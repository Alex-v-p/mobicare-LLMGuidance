from shared.config import ApiSettings


def test_settings_parse_allowed_lists_and_compute_minio_endpoint():
    settings = ApiSettings(
        minio_endpoint="https://minio.internal:9000",
        document_allowed_extensions_csv="pdf, txt ,.DOCX",
        document_allowed_content_types_csv="application/pdf, text/plain ",
    )

    assert settings.minio_client_endpoint == "minio.internal:9000"
    assert settings.document_allowed_extensions == {"pdf", "txt", "docx"}
    assert settings.document_allowed_content_types == {"application/pdf", "text/plain"}
