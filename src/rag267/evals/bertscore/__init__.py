"""
BERTScore evaluators for measuring semantic similarity between generated answers and reference answers.
"""

from .evaluator import bertscore_evaluator

__all__ = ["bertscore_evaluator"]