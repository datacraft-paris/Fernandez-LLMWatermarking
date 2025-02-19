
# HANDS-ON LLM WATERMARKING
– implementation of the greenlist/redlist watermarking technique, which is employed by LLM providers to watermark outputs.
– development of an interactive web app, which will showcase the active detector you built before.


# Goal
branch main : 
branch intermediate : `git checkout intermediate`
branch correction : `git checkout correction`

the app : 


# Install uv
Run `curl -LsSf https://astral.sh/uv/install.sh | sh`
Then, run: `uv sync`
Finally, activate your environment using: `source .venv/bin/activate`


# Run 
uv run /src/fernandez_llmwatermarking/main.py  --model_name smollm2-360m \
  --prompt "You can write your first prompt here" \
  --temperature 0.8 \
  --top_p 0.95 \
  --max_gen_len 256 \
  --method openai \
  --method_detect same \
  --seed 22 \
  --ngram 1 \ 

uv run /src/fernandez_llmwatermarking/main.py  --model_name smollm2-360m \
  --prompt "You can write your first prompt here" \
  --temperature 0.8 \
  --top_p 0.95 \
  --max_gen_len 256 \
  --method maryland \
  --method_detect same \
  --seed 22 \
  --ngram 1 \ 

You can try modifying the parameters' value.