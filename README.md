# UK Legal RAG Benchmark

A dataset for evaluating Retrieval-Augmented Generation (RAG) systems on UK legal information, developed in collaboration with Durham University Law School.

## Overview

This benchmark consists of 20 legal queries spanning various jurisdictions within the UK (England, Wales, Scotland, and Northern Ireland) and covering diverse legal domains including employment law, housing legislation, domestic abuse protection, and family law.

The dataset is designed to evaluate how different RAG configurations perform on legal information retrieval tasks.

## Dataset Structure

- **Country**: Specifies the relevant UK jurisdiction
- **Question**: Contains a detailed legal scenario
- **Actual Answer**: Contains the ground truth answer with correct legislative citations that should be referenced

## Dataset Purpose

This serves as a gold standard for evaluating RAG systems on legal information retrieval. The ground truth answers include specific legislative provisions that a properly functioning system should retrieve and reference in its responses.

## Usage

1. Evaluate RAG systems on legal information retrieval
2. Compare performance across different retrieval parameters
3. Assess accuracy in jurisdiction-specific legal contexts
4. Measure metrics such as relevancy, faithfulness, context precision, and context recall

## How to Cite

This benchmark is part of ongoing research. If you use it in your work, please check back later for citation information, or temporarily cite it as:
