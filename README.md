# HANDS-ON LLM WATERMARKING

## What you will learn in this workshop:
– implementation of the greenlist/redlist watermarking technique, which is employed by LLM providers to watermark outputs. 

– development of an interactive web app, which will showcase the active detector you built before.

# Setup
## Installation

Install uv:

Run `curl -LsSf https://astral.sh/uv/install.sh | sh`


## Setup

Then, run: `uv sync`

Finally, activate your environment using: `source .venv/bin/activate`

# Workshop program

## Overview of passive vs active detectors
1. Presentation by Pierre Fernandez, Research Scientist at Meta FAIR.
2. Interactive quizz: can you detect AI-generated content?

## Practical session: build watermarking engines

We developed several difficulty levels tailored to different levels of expertise!

First, choose the branch you want to challenge:

### main branch: hard mode

`git checkout main` or `git checkout hard`


### intermediate branch: medium mode

`git checkout intermediate`


### easy branch: easy mode
`git checkout easy`

### Find the correction
`git checkout correction`

### Where shall I write code

There are two files you have to complete:
1. `generator.py`: include watermarking in the generation process using two methods, OpenAI and Maryland
Head to the `sample_next` methods under the `MarylandGenerator` and `OpenaiGenerator` classes and fill the code.
The `logits_preprocessor` method has to be filled for the `MarylandGenerator` too.

2. `detector.py`: detect your own watermark post-generation using the `OpenaiDetector` and `MarylandDetector`, in theses classes, you can fill the `score_tok` methods to get it working.


## Running the watermarked generations

```bash
uv run /src/fernandez_llmwatermarking/main.py --model_name smollm2-360m
--prompt "You can write your first prompt here"
--temperature 0.8
--top_p 0.95
--max_gen_len 256
--method openai
--method_detect same
--seed 22
--ngram 1 \ 
```

```bash
uv run /src/fernandez_llmwatermarking/main.py --model_name smollm2-360m
--prompt "You can write your first prompt here"
--temperature 0.8
--top_p 0.95
--max_gen_len 256
--method maryland
--method_detect same
--seed 22
--ngram 1 \ 
```

You can try modifying the parameters' value.

## App building

Then, try to build your own chat interface a incorporating both watermarking in the generation as well as a watermark detector.

Some frameworks you can use:
- streamlit
- gradio
...

Now try to fool your own watermark detector, is it easy?
