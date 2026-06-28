
 # Block Diffusion

  Block Diffusion language model. This repo supports unconditional language modeling and
  seq2seq STAR graph prediction, where the model receives a prompt and generates the target answer block by block.

  ## Installation

  ```bash
  git clone https://github.com/rjagalamari/Block-Diffusion.git
  cd Block-Diffusion

  python3 -m venv .venv_block_diffusion
  source .venv_block_diffusion/bin/activate

  pip install -e .

  ## Main seq2seq experiment

  xlm job_type=train \
    job_name=star_easy_seq2seq_block_diffusion_debug \
    experiment=star_easy_seq2seq_block_diffusion \
    debug=overfit

  ## Generate

  xlm job_type=generate \
    job_name=star_easy_seq2seq_block_diffusion_generate \
    experiment=star_easy_seq2seq_block_diffusion \
    generation.ckpt_path=/path/to/checkpoint.ckpt

  ## Seq2seq setup

  For seq2seq training, the prompt is kept clean and the target answer is diffusion-masked.

  input  = prompt + noisy_answer
  target = prompt + clean_answer

  During inference, the model receives the prompt and generates the answer block by block.


