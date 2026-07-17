"""Embedding-backend drills: Azure batching + config-driven selection with
safe local fallback. Offline — the Azure client is faked."""

from types import SimpleNamespace

import adaptiverag.ingest.embedder as emb_mod
from adaptiverag.ingest.embedder import AzureEmbedder, build_embedder_from_settings


class FakeEmbeddingsAPI:
    def __init__(self, log):
        self._log = log

    def create(self, input, model):  # noqa: A002 — mirrors the openai SDK signature
        texts = [input] if isinstance(input, str) else input
        self._log.append({"n": len(texts), "model": model})
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5] * 1536)
                                     for _ in texts])


class FakeAzureClient:
    def __init__(self, azure_endpoint=None, api_key=None, api_version=None):
        self.calls: list[dict] = []
        self.embeddings = FakeEmbeddingsAPI(self.calls)


def _azure_embedder(monkeypatch, deployment="text-embedding-3-small"):
    monkeypatch.setattr(emb_mod, "AzureOpenAI", FakeAzureClient)
    e = AzureEmbedder(endpoint="https://x", api_key="k", deployment=deployment)
    return e, e._client


class TestAzureEmbedder:
    def test_known_model_dimension_no_probe(self, monkeypatch):
        e, client = _azure_embedder(monkeypatch)
        assert e.dimension == 1536
        assert client.calls == []                    # dimension came from the table

    def test_unknown_model_probes_once(self, monkeypatch):
        e, client = _azure_embedder(monkeypatch, deployment="mystery-model")
        assert e.dimension == 1536                   # probed via one API call
        assert e.dimension == 1536                   # cached — still one call
        assert len(client.calls) == 1

    def test_batch_respects_api_cap(self, monkeypatch):
        e, client = _azure_embedder(monkeypatch)
        vectors = e.embed_batch([f"t{i}" for i in range(5000)])
        assert len(vectors) == 5000                  # order + count preserved
        assert [c["n"] for c in client.calls] == [2048, 2048, 904]

    def test_uses_the_embed_deployment_name(self, monkeypatch):
        e, client = _azure_embedder(monkeypatch, deployment="my-embed-deploy")
        e.embed("hello")
        assert client.calls[0]["model"] == "my-embed-deploy"


class TestRateLimitPatience:
    def test_retries_then_succeeds(self, monkeypatch):
        from openai import RateLimitError
        e, client = _azure_embedder(monkeypatch)
        naps: list[int] = []
        monkeypatch.setattr(emb_mod.time, "sleep", lambda s: naps.append(s))
        real_create, tantrums = client.embeddings.create, [0]

        def moody_create(input, model):  # noqa: A002
            if tantrums[0] < 2:
                tantrums[0] += 1
                raise RateLimitError("quota exceeded")
            return real_create(input=input, model=model)
        client.embeddings.create = moody_create

        vectors = e.embed_batch(["a", "b"])
        assert len(vectors) == 2          # succeeded on the 3rd try
        assert naps == [30, 60]           # waited out the quota window, growing

    def test_gives_up_after_patience_spent(self, monkeypatch):
        from openai import RateLimitError
        import pytest as _pytest
        e, client = _azure_embedder(monkeypatch)
        monkeypatch.setattr(emb_mod.time, "sleep", lambda s: None)

        def always_429(input, model):  # noqa: A002
            raise RateLimitError("quota exceeded")
        client.embeddings.create = always_429
        with _pytest.raises(RateLimitError):
            e.embed_batch(["a"])          # error surfaces → ingest job reports it


def _settings(provider="azure_openai", endpoint="https://x", api_key="k",
              embed_deployment="text-embedding-3-small"):
    return SimpleNamespace(
        embeddings=SimpleNamespace(provider=provider),
        azure=SimpleNamespace(endpoint=endpoint, api_key=api_key,
                              embed_deployment=embed_deployment),
    )


class TestBackendSelection:
    def _patch_classes(self, monkeypatch):
        # Markers instead of real classes: LocalEmbedder would download a
        # model; AzureEmbedder would build a client. Selection is the test.
        monkeypatch.setattr(emb_mod, "AzureEmbedder",
                            lambda **kw: ("azure", kw["deployment"]))
        monkeypatch.setattr(emb_mod, "LocalEmbedder",
                            lambda model_name=None: ("local", model_name))

    def test_full_azure_config_selects_azure(self, monkeypatch):
        self._patch_classes(monkeypatch)
        assert build_embedder_from_settings(_settings()) == (
            "azure", "text-embedding-3-small")

    def test_missing_embed_deployment_falls_back_to_local(self, monkeypatch):
        self._patch_classes(monkeypatch)
        result = build_embedder_from_settings(_settings(embed_deployment=""))
        assert result[0] == "local"

    def test_missing_creds_fall_back_to_local(self, monkeypatch):
        self._patch_classes(monkeypatch)
        assert build_embedder_from_settings(_settings(api_key=""))[0] == "local"

    def test_local_provider_selects_local(self, monkeypatch):
        self._patch_classes(monkeypatch)
        # The future fully-local build: provider flag alone flips the backend.
        assert build_embedder_from_settings(_settings(provider="local"))[0] == "local"
