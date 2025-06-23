# EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding

## Overview
The 2025 EEG Foundation Challenge (NeurIPS 2025 Competition Track) aims to advance EEG decoding by addressing two critical challenges:

1. **Cross-Task Transfer Learning**: Developing models that can effectively transfer knowledge from passive EEG tasks to active tasks
2. **Subject Invariant Representation**: Creating robust representations that generalize across different subjects while predicting clinical factors

## Competition Details
- **Deadline**: August 01, 2025
- **Dataset**: HBN-EEG dataset (3000+ participants, 128-channel EEG)
- **Data Format**: BIDS (Brain Imaging Data Structure)
- **Prizes**: $2,500 + NeurIPS 25 competition presentation opportunity

## Challenges

### Challenge 1: Cross-Task Transfer Learning
- **Type**: Supervised learning (regression + classification)
- **Input**: EEG from passive task (Surround Suppression - SuS) + demographic data
- **Output**: 
  - Response time (regression)
  - Success rate (classification)
- **Goal**: Predict behavioral performance in active task (Contrast Change Detection - CCD) using passive EEG data

### Challenge 2: Psychopathology Factor Prediction
- **Type**: Supervised regression
- **Input**: EEG recordings from multiple experimental paradigms
- **Output**: 4 continuous psychopathology scores:
  - p-factor
  - internalizing
  - externalizing
  - attention
- **Goal**: Predict mental health factors from brain activity across subjects and tasks

## EEG Tasks in Dataset

### Passive Tasks
- **Resting State (RS)**: Eyes open/closed conditions with fixation cross
- **Surround Suppression (SuS)**: Four flashing peripheral disks with contrasting background
- **Movie Watching (MW)**: Four short films with different themes

### Active Tasks
- **Contrast Change Detection (CCD)**: Identifying dominant contrast in co-centric flickering grated disks
- **Sequence Learning (SL)**: Memorizing and reproducing sequences of flashed circles
- **Symbol Search (SyS)**: Computerized version of WISC-IV subtest

## Data Structure
- **Participants**: 3000+ 
- **Channels**: 128-channel EEG
- **Format**: BIDS compliant
- **Psychopathology**: 4 CBCL-derived dimensions
- **Demographics**: Age, sex, handedness

## Project Structure
```
Project/
├── README.md
├── data/
│   ├── raw/           # Raw HBN-EEG data
│   ├── processed/     # Preprocessed data
│   └── features/      # Extracted features
├── src/
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Model architectures
│   ├── training/      # Training scripts
│   └── evaluation/    # Evaluation utilities
├── configs/           # Configuration files
├── experiments/       # Experiment scripts
├── notebooks/         # Exploratory analysis
└── results/          # Model outputs and results
```

## Setup Instructions
1. Clone and setup environment
2. Link to HBN-EEG dataset
3. Install dependencies
4. Run data exploration
5. Begin model development

## Resources
- [Competition Website](https://eeg2025.github.io/)
- [HBN Dataset Documentation](https://neuromechanist.github.io/data/hbn/)
- [BIDS Format](https://bids.neuroimaging.io/)
- [HED Tags](https://www.hedtags.org/)

## Approach Strategy
- Leverage existing BENDR and EEGPT foundation model experience
- Focus on cross-task and cross-subject generalization
- Implement self-supervised pretraining strategies
- Fine-tune for specific supervised objectives 