# Pixels to Predictions: Deep Learning Kaggle Final Competition

This repository contains our final project for the **Pixels to Predictions: DL Vision Challenge**, a Kaggle competition for multimodal science multiple-choice question answering.

The goal of the competition was to use the required `HuggingFaceTB/SmolVLM-500M-Instruct` checkpoint to predict the correct answer choice for each test example using both an image and textual question context.

## Project Summary

We fine-tuned `HuggingFaceTB/SmolVLM-500M-Instruct` using parameter-efficient fine-tuning with **DoRA**, while keeping the number of trainable parameters below the competition limit of 5 million.

Our final system uses:

- Structured multiple-choice prompts
- Question-conditioned image captions
- Image augmentation
- Prompt variant sampling
- DoRA adapter fine-tuning
- Next-token answer-letter scoring
- Multi-prompt inference
- Test-time augmentation

Instead of using free-form generation, our inference pipeline directly scores valid answer letters such as `A`, `B`, `C`, `D`, and `E`, then converts the selected letter into the required 0-indexed integer answer format.

## Final Results

| Metric | Result |
|---|---:|
| Best validation checkpoint accuracy | 75.00% |
| Final inference validation accuracy | 74.05% |
| Final Kaggle leaderboard score | 0.77665 |
| Final Kaggle leaderboard rank | 70th |

## Repository Structure

```text
.
├── README.md
├── final_report.pdf
├── data/
│   ├── README_data.md
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   ├── final_notebook.ipynb
│   └── final_notebook.pdf
├── submission/
│   └── submission.csv
└── weights/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── README_weights.md

## Files

### `final_report.pdf`

Final written report describing the dataset, methodology, training setup, inference pipeline, ablations, results, limitations, and reproducibility details.

### `notebooks/final_notebook.ipynb`

Main notebook used for:

- Loading and preprocessing the competition data
- Generating question-conditioned image captions
- Fine-tuning SmolVLM with DoRA
- Evaluating on the validation split
- Running multi-prompt inference and test-time augmentation
- Generating the final Kaggle `submission.csv`

### `notebooks/final_notebook.pdf`

PDF export of the final notebook for easier review.

### `submission/submission.csv`

Final Kaggle submission file. It contains exactly two columns:

```text
id,answer
```

where `answer` is the predicted 0-indexed answer choice.

### `weights/`

Contains the trained DoRA adapter checkpoint.

Important files:

```text
adapter_model.safetensors
adapter_config.json
```

These are the trained adapter weights and adapter configuration. They are loaded on top of the required base model, not used as a standalone full model.

## Model Details

| Component | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| Base parameters | 507.5M |
| Fine-tuning method | DoRA / LoRA via PEFT |
| LoRA rank | 10 |
| LoRA alpha | 20 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `v_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | 4,638,720 |
| Trainable-parameter limit | 5,000,000 |

## Training Setup

| Hyperparameter | Value |
|---|---:|
| Epochs | 6 |
| Batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Learning rate | `1e-4` |
| Weight decay | 0.01 |
| Warmup ratio | 0.08 |
| Max gradient norm | 1.0 |
| Precision | `bfloat16` |
| Image longest edge | 1280 |
| Seed | 42 |

## Inference Setup

| Component | Setting |
|---|---|
| Prediction method | Next-token answer-letter scoring |
| Prompt variants | 3 |
| Test-time augmentation passes | 5 |
| Inference batch size | 4 |
| Captioning | Question-conditioned |
| Image splitting | Disabled |
| Output format | 0-indexed integer answer |

## Dataset

The competition dataset contains multimodal science multiple-choice examples.

| Split | Number of Examples |
|---|---:|
| Train | 3,109 |
| Validation | 1,048 |
| Test | 1,008 |

Each example includes an image, a question, multiple answer choices, and optional metadata such as subject, topic, grade, category, and skill.

The number of answer choices varies across examples, so inference restricts scoring to only the valid option letters for each question.

## Loading the Trained Adapter

The trained adapter should be loaded on top of the required base model:

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
import torch

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
ADAPTER_PATH = "weights"

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    do_image_splitting=False,
    size={"longest_edge": 1280},
)

base_model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
```

## Reproducibility

The full training and inference pipeline is contained in `notebooks/final_notebook.ipynb`.

To reproduce the final workflow:

1. Download the competition data from Kaggle.
2. Place `train.csv`, `val.csv`, `test.csv`, and `images.zip` in the expected notebook directory.
3. Run the notebook cells in order.
4. Load the best adapter checkpoint from `weights/` or from the saved output directory.
5. Generate `submission.csv`.

The notebook uses a fixed random seed:

```text
SEED = 42
```

The pipeline uses only the provided competition data and the required `HuggingFaceTB/SmolVLM-500M-Instruct` checkpoint.

## Google Drive Backup

The trained DoRA adapter checkpoint is also available here:

[Trained Adapter Checkpoint](https://drive.google.com/drive/u/1/folders/1VbOUbAfxy_bc7f_fNhMPfP2xK9xiVM7E)

This folder contains:

```text
adapter_config.json
adapter_model.safetensors
README.md
val_history.json
test_logprobs.pkl
```

## Authors

Ryan Fleishman  
New York University  
`rmf9265@nyu.edu`

Bram Simonnet  
New York University  
`bs4486@nyu.edu`
