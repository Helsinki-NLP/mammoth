"""Language token utilities for multilingual translation.

This module provides functionality to add language-specific prefix tokens to
HuggingFace tokenizers for multilingual translation training.
"""

import json
from pathlib import Path
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


def add_language_tokens_to_tokenizer(tokenizer_path: str, language_tokens, inplace=True):
    """
    Add language prefix tokens to an existing HuggingFace tokenizer.

    Args:
        tokenizer_path: Path to the tokenizer.json file
        language_tokens: Iterable of language tokens to add (e.g., {'<to_hi>', '<hi>', ...})
        inplace: If True, modify the tokenizer file in place. If False, return modified data

    Returns:
        Number of tokens added (if inplace=True) or tuple (num_added, modified_data) if inplace=False
    """
    if not language_tokens:
        logger.info("No language tokens to add")
        return 0 if inplace else (0, None)

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
        return 0 if inplace else (0, tokenizer_data)

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

    if inplace:
        # Create backup of original file
        backup_path = str(Path(tokenizer_path).with_suffix('.json.backup'))
        if not Path(backup_path).exists():
            logger.info(f"Creating backup: {backup_path}")
            import shutil
            shutil.copy2(tokenizer_path, backup_path)

        # Save the modified tokenizer
        logger.info(f"Saving modified tokenizer to: {tokenizer_path}")
        with open(tokenizer_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Successfully added {len(added_tokens)} language tokens! "
            f"New vocabulary size: {len(vocab)} (was {original_vocab_size})"
        )
        if added_tokens and logger.isEnabledFor(10):  # DEBUG level
            logger.debug("Added tokens:")
            for token, token_id in added_tokens:
                logger.debug(f"  {token:12s} → ID {token_id}")

        return len(added_tokens)
    else:
        return len(added_tokens), tokenizer_data


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
