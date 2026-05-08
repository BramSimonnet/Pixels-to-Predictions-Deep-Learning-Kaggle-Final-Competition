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
├── notebooks/
│   ├── final_notebook.ipynb
│   └── final_notebook.pdf
├── submission/
│   └── submission.csv
└── weights/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── README_weights.md
