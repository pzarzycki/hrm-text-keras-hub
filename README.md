# HRM-Text-1B KerasHub evidence

This repository contains the reproducible numerical-verification notebook for
the proposed KerasHub HRM-Text port.

## Run in Colab

Open [`hrm_text_1b_colab.ipynb`](hrm_text_1b_colab.ipynb) in Google Colab,
select a GPU runtime, and run all cells. A T4 is sufficient for the short
validation sequence.

The notebook checks out the exact KerasHub implementation commit
`91475efb2401642073e84e39cc3dc7090b8b19db`, pins the official Apache-2.0
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
