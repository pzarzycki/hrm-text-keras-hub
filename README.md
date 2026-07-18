# HRM-Text-1B KerasHub evidence

This repository contains two reproducible Colab notebooks for the proposed
KerasHub HRM-Text port:

- [`hrm_text_1b_colab.ipynb`](hrm_text_1b_colab.ipynb) is the numerical
  verification artifact: weight conversion, full-logit parity, and cached
  PrefixLM generation.
- [`hrm_text_1b_usage.ipynb`](hrm_text_1b_usage.ipynb) is a practical
  user-facing introduction: conversion, recommended prompt modes, configurable
  generation, complete continuation-only answers, and PrefixLM training inputs.

## Run in Colab

Open either notebook in Google Colab, select a GPU runtime, and run all cells.
A T4 is sufficient. The verification sequence is short; the usage notebook's
paragraph-length examples take roughly one to one-and-a-half minutes each on a
T4 because every generated token traverses 128 logical recurrent cache slots.
The implementation is currently under review, so both notebooks convert the
official checkpoint locally instead of relying on a hosted Keras preset.

The notebook checks out the exact KerasHub implementation commit
`5ff84000167698f0c412fb9cb8121c878e932749`, pins the official Apache-2.0
`sapientinc/HRM-Text-1B` checkpoint revision
`9f082d68b8cd0ebc56e33f1c88c45609174c272c`, and then:

1. converts every Hugging Face weight into KerasHub;
2. asserts full-logit parity with Transformers at `2e-4` tolerance; and
3. runs PrefixLM cached generation using the official 128-slot KV cache.

The initial clean-T4 run measured a maximum absolute logit error of
`1.1444091796875e-05` and generated the continuation `The capital of France`.

HRM-Text-1B is a base pre-alignment model, not a chat model. The notebook uses
the upstream PrefixLM condition format only to demonstrate correct execution;
it is not an evaluation of model quality.
