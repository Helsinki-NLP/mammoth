"""
HuggingFace → MAMMOTH Conversion

This module converts HuggingFace pretrained models to MAMMOTH format.

Main entry point:
    hfBART2mammoth.py - Convert HF BART model to MAMMOTH checkpoint
"""

from .hfBART2mammoth import convert_hf_to_mammoth

__all__ = ['convert_hf_to_mammoth']
