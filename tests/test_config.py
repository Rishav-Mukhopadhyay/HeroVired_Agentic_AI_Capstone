"""Smoke test: settings load and have sensible local defaults."""

from mediscribe.config import settings


def test_local_service_urls_have_defaults():
    assert settings.fhir_base_url.startswith("http")
    assert settings.neo4j_uri.startswith("bolt://")
    assert "postgresql" in settings.postgres_url
