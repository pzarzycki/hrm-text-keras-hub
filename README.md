# HRM-Text-1B KerasHub evidence

This repository contains two reproducible Colab notebooks for HRM-Text-1B in
KerasHub:

- [`hrm_text_1b_colab.ipynb`](hrm_text_1b_colab.ipynb) is the numerical
  verification artifact: strict weight and parameter accounting, tiny gradient
  parity, full-logit parity, cached/full agreement, preset round-trip, runtime
  measurements, and PrefixLM generation.
- [`hrm_text_1b_usage.ipynb`](hrm_text_1b_usage.ipynb) is a practical
  user-facing introduction: conversion, recommended prompt modes, configurable
  generation, complete continuation-only answers, and PrefixLM training inputs.

## Run in Colab

Open either notebook in Google Colab, select a GPU runtime, and run all cells.
A T4 is sufficient. The verification sequence is short; the usage notebook's
paragraph-length examples take roughly one to one-and-a-half minutes each on a
T4 because every generated token traverses 128 logical recurrent cache slots.
HRM-Text-1B is not yet distributed as a hosted KerasHub preset, so both
notebooks convert the official checkpoint locally from a pinned source
revision.

The notebook checks out the exact KerasHub implementation commit
`9721c146c87850e3908e7e8876757c4c11294335`, pins the official Apache-2.0
`sapientinc/HRM-Text-1B` checkpoint revision
`9f082d68b8cd0ebc56e33f1c88c45609174c272c`, and then:

1. converts every Hugging Face weight into KerasHub;
2. asserts the exact `1,182,795,264` parameter count;
3. compares tiny PrefixLM forward results and routed gradients with
   Transformers;
4. asserts official full-logit and four-step cached/full parity at `2e-4`;
5. saves and reloads a complete local Keras preset; and
6. records prefill, cached decoding, and peak-memory measurements before
   running PrefixLM generation through the official 128-slot KV cache.

## Clean-T4 validation

The current pinned revision was executed end-to-end on a clean Tesla T4 Colab
runtime. It produced a maximum absolute full-logit error of
`1.1444091796875e-05`; its four cached/full decode-step errors were all at most
`1.52587890625e-05`. The exact parameter count, 128 logical cache slots, tiny
gradient parity, local-preset round trip, and PrefixLM preprocessing all
passed.

The verification notebook measured float32 prefill throughput of 104.4 tok/s
at 128 tokens, 385.6 tok/s at 512 tokens, and 418.7 tok/s at 1,024 tokens.
Cached decoding measured 0.71 tok/s. The latter is a diagnostic measurement,
not a deployment benchmark: each generated token executes all 128 logical
recurrent cache slots.

The user-facing notebook also completed three deterministic examples: a
paragraph-length explanation of why the sky is blue (52 generated
tokens), `Tokyo` for a few-shot factual question (3 tokens), and a two-sentence
exercise explanation (47 tokens). Each stopped at the official end token.

HRM-Text-1B is a base pre-alignment model, not a chat model. The notebook uses
the upstream PrefixLM condition format only to demonstrate correct execution;
it is not an evaluation of model quality.
