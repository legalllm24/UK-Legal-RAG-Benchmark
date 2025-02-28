
# UK Legal RAG Benchmark

A benchmark dataset for evaluating Retrieval-Augmented Generation (RAG) systems on UK legal information, developed in collaboration with Durham University Law School.

## Overview

This benchmark consists of 20 legal queries spanning various jurisdictions within the UK (England, Wales, Scotland, and Northern Ireland) and covering diverse legal domains including employment law, housing legislation, domestic abuse protection, and family law.

The benchmark is designed to evaluate how different RAG configurations perform on legal information retrieval tasks.

## Dataset Structure

The benchmark is provided as a CSV file with the following columns:
- **Country**: Specifies the relevant UK jurisdiction
- **Question**: Contains a detailed legal scenario
- **Actual Answer**: Provides the model response with legislative citations

## Usage

This benchmark can be used to:
1. Evaluate RAG systems on legal information retrieval
2. Compare performance across different retrieval parameters
3. Assess accuracy in jurisdiction-specific legal contexts

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{yourname2025optimizing,
  title={Optimizing Retrieval Parameters in RAG Systems for Legislative Information: A Comparative Performance Analysis Using DeepSeek and Llama as Generation Models},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the Conference},
  year={2025}
}
