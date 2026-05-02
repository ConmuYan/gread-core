"""CLI entry points for GReaD-Core training pipeline.

Stage 1: train_detector -- base detector warm-up (no LLM)
Stage 2: generate_err -- offline ERR generation + verification (only stage calling LLM)
Stage 3: train_reasoner -- reasoner distillation (accepted ERR only)
"""
