"""Language token utilities for multilingual translation.

This module provides functionality to add language-specific prefix tokens to
HuggingFace tokenizers for multilingual translation training.
"""

import json
import os
from mammoth.utils.logging import logger


def extract_language_tokens_from_config(opts):
    """
    Extract all language prefix tokens from task configurations.

    Args:
        opts: Configuration object with tasks attribute

    Returns:
        Set of language tokens extracted from src_prefix and tgt_prefix fields
    """
    language_tokens = set()

    if not hasattr(opts, 'tasks') or not opts.tasks:
        logger.warning("No tasks found in configuration")
        return language_tokens

    for task_id, task_config in opts.tasks.items():
        if 'src_prefix' in task_config and task_config['src_prefix']:
            language_tokens.add(task_config['src_prefix'])
            logger.debug(f"Found src_prefix '{task_config['src_prefix']}' in task {task_id}")

        if 'tgt_prefix' in task_config and task_config['tgt_prefix']:
            language_tokens.add(task_config['tgt_prefix'])
            logger.debug(f"Found tgt_prefix '{task_config['tgt_prefix']}' in task {task_id}")

    logger.info(f"Extracted {len(language_tokens)} unique language tokens from config: {sorted(language_tokens)}")
    return language_tokens


def add_language_tokens_to_tokenizer(tokenizer_path: str, language_tokens, output_path: str, overwrite_input=False):
    """
    Add language prefix tokens to a HuggingFace tokenizer and save the result to output_path.

    The input tokenizer file is not modified unless overwrite_input is set, so other trainings
    that use the original tokenizer are unaffected.

    Args:
        tokenizer_path: Path to the tokenizer.json file to read
        language_tokens: Iterable of language tokens to add (e.g., {'<to_hi>', '<hi>', ...})
        output_path: Path where the extended tokenizer.json is written
        overwrite_input: Allow output_path == tokenizer_path. Only for copies previously written by
            save_tokenizers_with_language_tokens, never for a user-provided original.

    Returns:
        Number of tokens added. If 0, nothing is written.
    """
    if not overwrite_input and os.path.abspath(tokenizer_path) == os.path.abspath(output_path):
        raise ValueError(f"Refusing to overwrite the original tokenizer: {tokenizer_path}")

    if not language_tokens:
        logger.info("No language tokens to add")
        return 0

    logger.info(f"Adding {len(language_tokens)} language tokens to tokenizer: {tokenizer_path}")

    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)

    vocab = tokenizer_data['model']['vocab']
    original_vocab_size = len(vocab)
    logger.debug(f"Original vocabulary size: {original_vocab_size}")

    # Find the maximum token ID in the vocabulary
    max_id = max(vocab.values())
    logger.debug(f"Maximum token ID: {max_id}")

    # Add language tokens starting from max_id + 1
    added_tokens = []
    next_id = max_id + 1

    # Sort tokens for consistent ordering
    for token in sorted(language_tokens):
        if token in vocab:
            logger.debug(f"Token '{token}' already exists with ID {vocab[token]}")
        else:
            vocab[token] = next_id
            added_tokens.append((token, next_id))
            next_id += 1

    if not added_tokens:
        logger.info("All language tokens already exist in tokenizer")
        return 0

    # Update the vocabulary in the tokenizer data
    tokenizer_data['model']['vocab'] = vocab

    # Add to the added_tokens section for special handling
    if 'added_tokens' not in tokenizer_data:
        tokenizer_data['added_tokens'] = []

    for token, token_id in added_tokens:
        tokenizer_data['added_tokens'].append({
            "id": token_id,
            "content": token,
            "single_word": True,  # Treat as single token, don't split
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True  # Mark as special token
        })

    # Write atomically: several nodes may write the same file on a shared filesystem
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_path = f"{output_path}.tmp.{os.getpid()}"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)

    logger.info(
        f"Saved tokenizer with {len(added_tokens)} added language tokens to: {output_path} "
        f"(vocabulary size {len(vocab)}, was {original_vocab_size})"
    )
    if logger.isEnabledFor(10):  # DEBUG level
        logger.debug("Added tokens:")
        for token, token_id in added_tokens:
            logger.debug(f"  {token:12s} → ID {token_id}")

    return len(added_tokens)


def save_tokenizers_with_language_tokens(opts, langs_by_side, language_tokens, output_dir: str,
                                         extend_existing=False):
    """
    Write an extended copy of every HuggingFace tokenizer used by the tasks into output_dir,
    and point opts.src_vocab / opts.tgt_vocab at the copies.

    Each unique source tokenizer is written once, to output_dir/<source parent dir>/tokenizer.json,
    so languages that share a tokenizer keep sharing it. Tokenizers that already contain all
    language tokens keep their original path.

    A source tokenizer that is itself a copy inside output_dir (the config points at the saved copy)
    is extended in place. With extend_existing (resuming), an existing copy is extended instead of
    being rebuilt from the original. In both cases tokens already in the copy keep their ids and
    new tokens are appended, so the ids stay consistent with the checkpoint vocab.

    Args:
        opts: Configuration object with src_vocab / tgt_vocab dicts (updated in place)
        langs_by_side: Dict mapping 'src' / 'tgt' to the languages used on that side
        language_tokens: Iterable of language tokens to add
        output_dir: Directory for the extended tokenizers (normally next to the saved model)
        extend_existing: Build on an existing copy in output_dir instead of the original

    Returns:
        Dict mapping original tokenizer path to the path now used for training.
    """
    output_dir = os.path.abspath(output_dir)
    new_paths = {}
    used_names = set()
    for side in ('src', 'tgt'):
        vocab_paths = getattr(opts, f'{side}_vocab')
        for lang in langs_by_side[side]:
            original_path = vocab_paths[lang]
            if not original_path.endswith('.json'):
                continue
            if original_path not in new_paths:
                source_dir = os.path.dirname(os.path.abspath(original_path))
                if os.path.dirname(source_dir) == output_dir:
                    # Config already points at a copy we wrote: extend it in place
                    used_names.add(os.path.basename(source_dir))
                    add_language_tokens_to_tokenizer(
                        original_path, language_tokens, original_path, overwrite_input=True
                    )
                    new_paths[original_path] = original_path
                else:
                    name = os.path.basename(source_dir) or 'tokenizer'
                    unique_name, i = name, 1
                    while unique_name in used_names:
                        i += 1
                        unique_name = f'{name}_{i}'
                    used_names.add(unique_name)
                    output_path = os.path.join(output_dir, unique_name, 'tokenizer.json')
                    if extend_existing and os.path.exists(output_path):
                        logger.info(f"Extending existing tokenizer copy {output_path} (source: {original_path})")
                        add_language_tokens_to_tokenizer(
                            output_path, language_tokens, output_path, overwrite_input=True
                        )
                        new_paths[original_path] = output_path
                    else:
                        n_added = add_language_tokens_to_tokenizer(original_path, language_tokens, output_path)
                        new_paths[original_path] = output_path if n_added else original_path
            vocab_paths[lang] = new_paths[original_path]
    return new_paths


def verify_language_tokens(tokenizer_path: str, language_tokens):
    """
    Verify that language tokens are properly accessible in a tokenizer.

    Args:
        tokenizer_path: Path to the tokenizer.json file
        language_tokens: Iterable of tokens to test

    Returns:
        True if all checks pass, False otherwise
    """
    try:
        from tokenizers import Tokenizer
    except ImportError:
        logger.warning("HuggingFace tokenizers library not available, skipping verification")
        return False

    if not language_tokens:
        logger.info("No language tokens to verify")
        return True

    logger.debug(f"Verifying language tokens in: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    all_found = True
    for token in sorted(language_tokens):
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            logger.warning(f"Language token '{token}' not found in tokenizer")
            all_found = False
        else:
            logger.debug(f"Verified token {token:12s} → ID {token_id}")

    # Test encoding for a sample (should use single token ID, not split)
    test_sample = list(sorted(language_tokens))[:3]
    for token in test_sample:
        encoding = tokenizer.encode(token)
        if len(encoding.ids) != 1:
            logger.warning(
                f"Token '{token}' was split into {len(encoding.ids)} tokens: {encoding.ids}"
            )
            all_found = False

    if all_found:
        logger.info("All language token checks passed!")
    else:
        logger.warning("Some language token checks failed")

    return all_found
