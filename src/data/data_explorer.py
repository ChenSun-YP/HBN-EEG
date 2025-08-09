#!/usr/bin/env python3
"""
HBN-EEG Dataset Explorer for EEG Foundation Challenge
Explores dataset structure and provides insights for competition tasks.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict, Counter
import mne
import warnings
warnings.filterwarnings('ignore')

class HBNDatasetExplorer:
    """Explore HBN-EEG dataset structure and extract relevant information."""
    
    def __init__(self, dataset_path):
        """Initialize with path to HBN_BIDS_EEG directory."""
        self.dataset_path = Path(dataset_path)
        self.releases = []
        self.participants_data = {}
        self.task_info = {}
        
    def explore_dataset_structure(self):
        """Explore the overall dataset structure."""
        print("=== HBN-EEG Dataset Structure Exploration ===\n")
        
        # Find all release directories
        self.releases = [d.name for d in self.dataset_path.iterdir() 
                        if d.is_dir() and d.name.startswith('cmi_bids_')]
        
        print(f"Found {len(self.releases)} releases:")
        for release in sorted(self.releases):
            print(f"  - {release}")
        
        # Analyze each release
        for release in sorted(self.releases):
            self._analyze_release(release)
            
        return self.releases
    
    def _analyze_release(self, release_name):
        """Analyze a specific release directory."""
        release_path = self.dataset_path / release_name
        print(f"\n--- Analyzing {release_name} ---")
        
        # Check for participants.tsv
        participants_file = release_path / 'participants.tsv'
        if participants_file.exists():
            df = pd.read_csv(participants_file, sep='\t')
            self.participants_data[release_name] = df
            
            print(f"Participants: {len(df)}")
            print(f"Age range: {df['age'].min():.1f} - {df['age'].max():.1f} years")
            print(f"Sex distribution: {df['sex'].value_counts().to_dict()}")
            
            # Competition-relevant columns
            psycho_cols = ['p_factor', 'attention', 'internalizing', 'externalizing']
            available_psycho = [col for col in psycho_cols if col in df.columns]
            print(f"Psychopathology factors available: {available_psycho}")
            
            # Task availability
            task_cols = [col for col in df.columns if any(task in col.lower() for task in 
                        ['resting', 'despicable', 'funwithfractals', 'thepresent', 'diary',
                         'contrast', 'surround', 'seqlearning', 'symbol'])]
            print(f"Task columns: {len(task_cols)}")
            
        # Count subjects
        subject_dirs = [d for d in release_path.iterdir() 
                       if d.is_dir() and d.name.startswith('sub-')]
        print(f"Subject directories: {len(subject_dirs)}")
        
        # Sample first few subjects to understand structure
        if subject_dirs:
            sample_subject = subject_dirs[0]
            self._analyze_subject_structure(sample_subject, release_name)
    
    def _analyze_subject_structure(self, subject_path, release_name):
        """Analyze structure of a sample subject."""
        eeg_dir = subject_path / 'eeg'
        if not eeg_dir.exists():
            print(f"  No EEG directory found for {subject_path.name}")
            return
            
        eeg_files = list(eeg_dir.glob('*.set'))
        print(f"  Sample subject {subject_path.name}: {len(eeg_files)} EEG files")
        
        # Extract task information
        task_counts = Counter()
        for eeg_file in eeg_files:
            # Extract task name from filename
            filename = eeg_file.name
            if '_task-' in filename:
                task = filename.split('_task-')[1].split('_')[0]
                task_counts[task] += 1
        
        if task_counts:
            print(f"  Tasks found: {dict(task_counts)}")
            
        # Store task info for this release
        if release_name not in self.task_info:
            self.task_info[release_name] = task_counts
    
    def analyze_competition_targets(self):
        """Analyze data relevant to competition challenges."""
        print("\n=== Competition Challenge Analysis ===\n")
        
        # Challenge 1: Cross-Task Transfer Learning
        print("Challenge 1: Cross-Task Transfer Learning")
        print("- Input: Passive task (Surround Suppression) + demographics")
        print("- Output: Active task performance (Contrast Change Detection)")
        print("- Tasks needed: 'surroundSupp' (input) -> 'contrastChangeDetection' (target)")
        
        # Challenge 2: Psychopathology Prediction  
        print("\nChallenge 2: Psychopathology Factor Prediction")
        print("- Input: EEG from multiple paradigms")
        print("- Output: 4 psychopathology scores (p_factor, attention, internalizing, externalizing)")
        
        # Check data availability across releases
        self._check_challenge_data_availability()
    
    def _check_challenge_data_availability(self):
        """Check data availability for competition challenges."""
        print("\n--- Data Availability Check ---")
        
        total_subjects = 0
        subjects_with_psycho = 0
        subjects_with_surround_supp = 0
        subjects_with_contrast_change = 0
        
        for release_name, df in self.participants_data.items():
            total_subjects += len(df)
            
            # Check psychopathology data
            psycho_cols = ['p_factor', 'attention', 'internalizing', 'externalizing'] 
            if all(col in df.columns for col in psycho_cols):
                # Count subjects with non-null psycho data
                psycho_available = df[psycho_cols].notna().all(axis=1).sum()
                subjects_with_psycho += psycho_available
            
            # Check task availability
            if 'surroundSupp_1' in df.columns:
                surround_available = (df['surroundSupp_1'] == 'available').sum()
                subjects_with_surround_supp += surround_available
                
            if 'contrastChangeDetection_1' in df.columns:
                contrast_available = (df['contrastChangeDetection_1'] == 'available').sum()
                subjects_with_contrast_change += contrast_available
        
        print(f"Total subjects across all releases: {total_subjects}")
        print(f"Subjects with psychopathology data: {subjects_with_psycho}")
        print(f"Subjects with Surround Suppression task: {subjects_with_surround_supp}")
        print(f"Subjects with Contrast Change Detection: {subjects_with_contrast_change}")
        
    def get_task_mapping(self):
        """Get mapping of task names to competition relevance."""
        task_mapping = {
            # Passive tasks (for input)
            'RestingState': 'passive',
            'DespicableMe': 'passive', 
            'FunwithFractals': 'passive',
            'ThePresent': 'passive',
            'DiaryOfAWimpyKid': 'passive',
            'surroundSupp': 'passive_key',  # Key for Challenge 1
            
            # Active tasks (for targets)
            'contrastChangeDetection': 'active_key',  # Key for Challenge 1  
            'seqLearning6target': 'active',
            'seqLearning8target': 'active',
            'symbolSearch': 'active'
        }
        
        print("\n=== Task Classification ===")
        for task, category in task_mapping.items():
            print(f"{task}: {category}")
            
        return task_mapping
    
    def recommend_data_splits(self):
        """Recommend data splits for the competition."""
        print("\n=== Recommended Data Approach ===")
        
        print("1. Data Selection Priority:")
        print("   - Focus on releases with both SuS and CCD tasks")
        print("   - Prioritize subjects with complete psychopathology scores")
        print("   - Include subjects with multiple task recordings")
        
        print("\n2. Cross-Task Transfer (Challenge 1):")
        print("   - Training: Use SuS (passive) -> CCD performance prediction")
        print("   - Validation: Hold out subjects, not just trials")
        print("   - Features: Extract from SuS EEG + demographics")
        
        print("\n3. Subject Invariant Representation (Challenge 2):")
        print("   - Use all available tasks for richer representations")
        print("   - Pre-train on all tasks, fine-tune for psychopathology")
        print("   - Cross-validation: Ensure subject-level splits")
        
        print("\n4. Foundation Model Strategy:")
        print("   - Pre-train on large corpus (all tasks, all subjects)")
        print("   - Self-supervised objectives: masking, contrastive learning")
        print("   - Fine-tune for specific competition objectives")

def main():
    """Main exploration function."""
    # Path to HBN dataset
    dataset_path = "src/data/raw/HBN_BIDS_EEG"
    
    if not Path(dataset_path).exists():
        print(f"Dataset path {dataset_path} not found!")
        print("Please check the path or create symbolic link to dataset.")
        return
    
    # Initialize explorer
    explorer = HBNDatasetExplorer(dataset_path)
    
    # Run exploration
    explorer.explore_dataset_structure()
    explorer.analyze_competition_targets()
    explorer.get_task_mapping()
    explorer.recommend_data_splits()
    
    print("\n=== Next Steps ===")
    print("1. Create data loading pipelines for each challenge")
    print("2. Implement baseline models (BENDR, EEGPT)")
    print("3. Design cross-task and cross-subject evaluation protocols")
    print("4. Develop foundation model pre-training strategy")

if __name__ == "__main__":
    main() 