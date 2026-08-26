def is_tie_word_embeddings(model) -> bool:
    """Return whether the model config ties input embeddings and LM head."""
    config = getattr(model, "config", None)
    return bool(getattr(config, "tie_word_embeddings", False))

