"""Compatibility checks for the public model package imports."""

import importlib
import pickle

import pytest

import model


@pytest.mark.parametrize(
    ("name", "module_name"),
    [
        ("ROPEEmbedding", "rope"),
        ("Attention", "attention"),
        ("FeedForward", "feed_forward"),
        ("MOE", "moe"),
        ("LLMLayer", "llm_layer"),
        ("LLM", "llm"),
        ("VLM", "vlm"),
        ("KVCache", "cache"),
        ("PastKeyValues", "cache"),
        ("VLMCache", "cache"),
    ],
)
def test_public_imports_preserve_component_identity(name, module_name):
    component_module = importlib.import_module(f"model.{module_name}")
    assert getattr(model, name) is getattr(component_module, name)


@pytest.mark.parametrize(
    "name",
    ["ROPEEmbedding", "Attention", "FeedForward", "MOE", "LLMLayer",
     "LLM", "VLM", "VLMCache"],
)
def test_legacy_pickled_class_references_still_resolve(name):
    # Protocol 0 GLOBAL references match the paths used before the split.
    reference = f"cmodel\n{name}\n.".encode("ascii")
    assert pickle.loads(reference) is getattr(model, name)
