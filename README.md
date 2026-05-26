# Transformer From Scratch in PyTorch

A complete implementation of the Transformer architecture from scratch using PyTorch, inspired by the original paper:

> “Attention Is All You Need”.

This project was built for deep conceptual understanding of how Transformers work internally, including attention mechanisms, positional encoding, masking, encoder-decoder architecture, and sequence-to-sequence training.

---

# Features

Implemented completely from scratch without using HuggingFace or high-level Transformer APIs.

## Implemented Components

- Multi-Head Self Attention
- Scaled Dot Product Attention
- Positional Encoding (Sinusoidal)
- Encoder Architecture
- Decoder Architecture
- Residual Connections
- Layer Normalization
- Feed Forward Network
- Source Padding Mask
- Target Causal Mask
- Seq2Seq Training Loop
- Token Embeddings
- English → Hindi Toy Translation Dataset

---

# Architecture Overview

![Transformer](Images/transformer_architecture.webp)

The model follows the original Transformer encoder-decoder architecture:

Input Sentence
↓
Encoder Stack
↓
Contextual Representations
↓
Decoder Stack
↓
Next Token Prediction

---

# Key Concepts Learned

This implementation helped in understanding:

- Why attention matrices are `(T × T)`
- How Query, Key and Value interact
- Why masking is required in decoders
- Difference between source and target masks
- Why positional encoding is needed
- How tensor reshaping and broadcasting work
- Why softmax is applied over key dimensions
- Difference between logits and probabilities
- How CrossEntropyLoss works internally

---

# Project Structure

```bash
transformer-from-scratch/

│── notebooks/
│   └── transformer_from_scratch.ipynb

│── README.md