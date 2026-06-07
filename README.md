# Multimodal Land Cover Segmentation

Term project for DI 725 — Transformers and Attention-Based Deep Networks, METU Graduate School of Informatics.

## Overview

This project investigates whether AI-generated scene descriptions can improve satellite image segmentation accuracy. A SegFormer encoder is combined with a BERT text encoder via a gated cross-attention fusion module. The model is trained on a 10,000-sample dataset of georeferenced satellite images with seven land cover classes: Tree, Shrub, Grass, Crop, Built-up, Barren, and Water.

## Project Structure

| Phase | Focus | Best mIoU |
|-------|-------|-----------|
| Phase 1 | Dataset exploration, baseline SegFormer (MiT-B0) | — |
| Phase 2 (Exp 1–8) | Backbone scaling, plain cross-attention fusion, random caption ablation | 0.7079 (B2) |
| Phase 3 (Exp 9–13) | Gated cross-attention, BERT fine-tuning & injection stage ablation | **0.7460** (B2 + gated fusion) |

## Key Findings

- Backbone capacity (MiT-B0 → B2) is the dominant accuracy driver
- Random caption ablation confirmed captions are genuinely read: shuffled captions degrade mIoU by −0.50 pp vs. the B0 baseline
- Gated fusion (alpha initialized to 0) outperforms plain cross-attention by preventing forced noise injection
- BERT fine-tuning and mid-stage injection do not help at this dataset scale