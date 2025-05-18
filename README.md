# Multimodal Visual Question Answering with Amazon Berkeley Objects Dataset

# Visual Recognition Mini Project-2
## Contributors: 
<h3 align = "center">
Varsha Yamsani(IMT2022506): Yamsani.Varsha@iiitb.ac.in<br> 
R Harshavardhan(IMT2022515): R.Harshavardhan@iiitb.ac.in<br>
Keshav Goyal(IMT2022560) Keshav.Goyal560@iiitb.ac.in</h3>

# Introduction

This assignment involves creating a multiple-choice Visual Question Answering (VQA)
dataset using the Amazon Berkeley Objects (ABO) dataset, evaluating baseline models,
fine-tuning using Low-Rank Adaptation (LoRA), and assessing performance using standard metrics.

# Report

Please refer to the report [`IMT2022506_515_560_MiniProject2.pdf`](./IMT2022506_515_560_MiniProject2.pdf) located in the root directory of this repository for detailed analysis, methodology, and results.

---

# Inference Instructions

To run the inference:

1. Download the [`sample-submission`](./sample-submission) folder.
2. Set up a Python virtual environment and install required dependencies for your `run_inference_for_all.py` first.

# File Structure of the Repo

```
├── dataset/                        # Contains the final dataset containing of train test and validation split
├── generated_questions/           # Auto-generated visual question files
├── inference-setup/               # Setup code and scripts for inference
├── sample-submission/IMT2022506/  # Folder for final inference code submission
├── model_blip_r16/                # BLIP model fine-tuned with LoRA r=16
├── model_blip_r32/                # BLIP model fine-tuned with LoRA r=32
├── model_blip_r8/                 # BLIP model fine-tuned with LoRA r=8
├── IMT2022506_515_560_MiniProject2.pdf  # Final project report
├── README.md                      # Project overview and instructions
├── Vilt-Lora32-finetuning.ipynb   # LoRA finetuning script for ViLT (r=32)
├── Vilt_BaselineModel.ipynb       # Baseline inference script for ViLT
├── baseline-Blip.ipynb            # Baseline inference for BLIP
├── baseline-bakLlava.ipynb        # Baseline inference for BakLlava
├── baseline-clip.ipynb            # Baseline inference for CLIP
├── blip-2-baseline.ipynb          # Baseline inference for BLIP-2
├── curation.ipynb                 # Dataset curation and preprocessing
├── granitevision_baseline.ipynb   # Baseline inference for Granite Vision
├── VR Mini Project Two            # The problem statement given 
