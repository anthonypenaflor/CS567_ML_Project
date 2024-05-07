from transformers import AutoModelForCausalLM, AutoTokenizer


def encode_and_truncate(
    texts: list[str], tokenizer: AutoTokenizer, n_prompt_tokens: int = None, device: str = "cuda"
):
    """Encode each text as a list of token ids, and truncate to the first n_prompt_tokens tokens.
    Args:
        texts: list[str]
            List of texts to encode.
        tokenizer: AutoTokenizer
            Tokenizer to use.
        n_prompt_tokens: int
            Number of tokens to keep for each text. If the text has fewer than this many tokens, it will be padded. Default is None.
        device: str
            Device to use for encoding. Default is "cuda".
    Returns:
        dict[str, torch.Tensor]
            Dictionary of tensors, with keys "input_ids", "attention_mask", "token_type_ids".
    """
    all_encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)

    if n_prompt_tokens is None:
        return all_encoded
    return {key: value[:, :n_prompt_tokens] for key, value in all_encoded.items()}


def sample_from_model(
    texts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sampling_kwargs: dict,
    min_words=55,
    n_prompt_tokens=None,
) -> list[str]:
    """Sample from the model using the provided texts as prompts, until each sample has at least min_words words.
    Args:
        texts: list[str]
            List of texts to use as prompts.
        model: AutoModelForCausalLM
            Model to sample from.
        tokenizer: AutoTokenizer
            Tokenizer to use.
        min_words: int
            Minimum number of words for each sample.
        n_prompt_tokens: int
            Number of tokens to use for each prompt.
    Returns:
        list[str]
            List of samples generated from the model.
    """
    all_encoded = encode_and_truncate(texts, tokenizer, n_prompt_tokens)
    decoded = ["" for _ in range(len(texts))]

    # sample from the model until we get a sample with at least min_words words for each example
    # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
    tries = 0
    while (m := min(len(x.split()) for x in decoded)) < min_words:
        if tries != 0:
            print()
            print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

        outputs = model.generate(**all_encoded, **sampling_kwargs)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tries += 1

    # Remove the prompt text from the generated text
    cleaned = []
    for prompt, gen in zip(texts, decoded):
        cleaned.append(gen.replace(prompt, "").strip())

    return cleaned


def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(" ")), len(textb.split(" ")))
    texta = " ".join(texta.split(" ")[:shorter_length])
    textb = " ".join(textb.split(" ")[:shorter_length])
    return texta, textb
