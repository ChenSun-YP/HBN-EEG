"""
EEG Foundation Challenge Data Loader
Supports both Challenge 1 (Cross-Task) and Challenge 2 (Psychopathology)

CHALLENGE 1 OFFICIAL CONSTRAINTS:
- Training Input: ONLY SuS (Surround Suppression) EEG epochs
- Prediction Targets: CCD behavioral outcomes per trial:
  * Response time (regression)
  * Hit/miss accuracy (binary classification)
  * Age (auxiliary regression)
  * Sex (auxiliary classification)
- Constraint: CCD EEG data is NOT provided and NOT allowed for training

Per-trial approach: Extract per-trial samples (not subject-level aggregation)
Match SuS pre-trial EEG epochs (2 seconds before contrast change) with CCD behavioral outcomes
"""

import os
import re
import pandas as pd
import numpy as np
import mne
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import yaml
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import partial
import time

# We'll handle boundary events properly instead of ignoring warnings

logging.basicConfig(
    level=logging.INFO,       # Minimum level to log
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('challenge1_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Challenge1Dataset(Dataset):
    """
    Per-trial dataset for EEG Foundation Challenge 1
    
    Creates samples where:
    - X (Input): SuS EEG pre-trial epoch (2 seconds before CCD contrast change) + demographics
    - Y (Target): CCD behavioral outcome (response_time, hit_miss, age, sex) for that specific trial
    
    Each sample corresponds to one CCD trial with matching SuS pre-trial EEG data.
    """
    
    def __init__(self, data_dir: str, config: Dict[str, Any], split: str = 'train', 
                 subject_split: Optional[Dict[str, List[str]]] = None, transforms=None):
        """
        Initialize per-trial dataset
        
        Args:
            data_dir: Path to HBN BIDS EEG dataset
            config: Configuration dictionary
            split: Dataset split ('train', 'val', 'test')
            subject_split: Optional pre-defined subject splits
            transforms: Optional transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.subject_split = subject_split
        self.transforms = transforms
        
        # Get dataset configuration
        self.channel_count = config['data']['channel_count']
        self.sampling_rate = config['data']['sampling_rate']
        self.pre_trial_duration = 2.0  # 2 seconds as specified in paper
        
        # Parallel processing configuration
        self.use_parallel = config.get('parallel', {}).get('enabled', True)
        self.max_workers = config.get('parallel', {}).get('max_workers', min(64, multiprocessing.cpu_count()))
        self.batch_size = config.get('parallel', {}).get('batch_size', 10)  # Process subjects in batches
        
        logger.info(f"Parallel processing: {'enabled' if self.use_parallel else 'disabled'}")
        if self.use_parallel:
            logger.info(f"Using {self.max_workers} workers with batch size {self.batch_size}")
        
        # Setup preprocessing
        self._setup_preprocessing()
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Create per-trial samples
        self.samples = []
        self._create_per_trial_samples()
        
        logger.info(f"Created {len(self.samples)} per-trial samples for {split} split")
    
    def _setup_preprocessing(self):
        """Setup preprocessing parameters"""
        preprocess_config = self.config.get('preprocessing', {})
        
        self.filter_params = {
            'l_freq': preprocess_config.get('filter_low', 0.5),
            'h_freq': preprocess_config.get('filter_high', 70.0)
        }
        
        self.resample_freq = preprocess_config.get('resample_freq', 256)
    
    def _load_metadata(self) -> Dict[str, pd.DataFrame]:
        """Load BIDS metadata for all releases"""
        metadata = {}
        
        # Check if quick test mode is enabled
        quick_test = self.config.get('quick_test', {})
        quick_test_enabled = quick_test.get('enabled', False)
        allowed_bids_dirs = quick_test.get('bids_dirs', None) if quick_test_enabled else None
        
        for release_dir in self.data_dir.iterdir():
            if not release_dir.is_dir() or not release_dir.name.startswith('cmi_bids_'):
                logger.warning(f"Skipping non-BIDS directory: {release_dir}")
                continue
            
            # In quick test mode, only load specified BIDS directories
            if quick_test_enabled and allowed_bids_dirs and release_dir.name not in allowed_bids_dirs:
                logger.warning(f"Skipping BIDS directory not in quick test mode: {release_dir}")
                continue
                
            participants_file = release_dir / 'participants.tsv'
            if participants_file.exists():
                df = pd.read_csv(participants_file, sep='\t')
                metadata[release_dir.name] = df
                logger.info(f"Loaded metadata for {release_dir.name}: {len(df)} participants")
        
        return metadata
    
    def _create_per_trial_samples(self):
        """Create per-trial samples by matching SuS pre-trial EEG with CCD behavioral outcomes"""
        
        # Check if quick test mode is enabled
        quick_test = self.config.get('quick_test', {})
        quick_test_enabled = quick_test.get('enabled', False)
        max_subjects = quick_test.get('max_subjects', None) if quick_test_enabled else None
        subjects_processed = 0
        
        # Collect all subjects to process
        all_subjects = []
        for release, metadata_df in self.metadata.items():
            release_path = self.data_dir / release
            
            for _, participant in metadata_df.iterrows():
                # Check if we've reached the subject limit in quick test mode
                if max_subjects and subjects_processed >= max_subjects:
                    logger.info(f"Quick test mode: stopping after {subjects_processed} subjects")
                    break
                
                subject_id = participant['participant_id']
                
                # Apply subject split if provided
                if self.subject_split:
                    if subject_id not in self.subject_split[self.split]:
                        logger.info(f"Subject {subject_id} not in {self.split} split")
                        continue
                
                subject_dir = release_path / subject_id / 'eeg'
                
                if not subject_dir.exists():
                    logger.warning(f"Subject directory not found: {subject_dir}")
                    continue
                
                # Find SuS and CCD files for this subject
                sus_files = self._find_sus_files(subject_dir)
                ccd_files = self._find_ccd_files(subject_dir)
                
                if not sus_files or not ccd_files:
                    logger.warning(f"Missing SuS or CCD files for {subject_id}")
                    continue
                
                all_subjects.append({
                    'subject_id': subject_id,
                    'sus_files': sus_files,
                    'ccd_files': ccd_files,
                    'participant': participant,
                    'release': release
                })
                subjects_processed += 1
                
            if max_subjects and subjects_processed >= max_subjects:
                logger.info(f"Quick test mode: stopping after {subjects_processed} subjects")
                break
        
        logger.info(f"Processing {len(all_subjects)} subjects...")
        
        # Process subjects in parallel or sequentially
        if self.use_parallel and len(all_subjects) > 1:
            self._process_subjects_parallel(all_subjects)
        else:
            self._process_subjects_sequential(all_subjects)
    
    def _process_subjects_parallel(self, all_subjects: List[Dict]):
        """Process subjects in parallel for much faster data loading"""
        start_time = time.time()
        
        # Process subjects in batches to avoid memory issues
        for i in range(0, len(all_subjects), self.batch_size):
            batch = all_subjects[i:i + self.batch_size]
            batch_start = time.time()
            
            # Use ThreadPoolExecutor for I/O-bound operations
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create partial function with preprocessing parameters
                process_func = partial(
                    self._process_single_subject,
                    filter_params=self.filter_params,
                    resample_freq=self.resample_freq,
                    pre_trial_duration=self.pre_trial_duration
                )
                
                # Process subjects in parallel
                futures = [executor.submit(process_func, subject) for subject in batch]
                
                # Collect results
                for future in futures:
                    try:
                        subject_samples = future.result()
                        if subject_samples:
                            self.samples.extend(subject_samples)
                    except Exception as e:
                        logger.error(f"Error processing subject: {e}")
            
            batch_time = time.time() - batch_start
            logger.info(f"Processed batch {i//self.batch_size + 1}/{(len(all_subjects) + self.batch_size - 1)//self.batch_size} "
                       f"({len(batch)} subjects) in {batch_time:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Parallel processing completed in {total_time:.2f}s")
    
    def _process_subjects_sequential(self, all_subjects: List[Dict]):
        """Process subjects sequentially (fallback)"""
        start_time = time.time()
        
        for i, subject in enumerate(all_subjects):
            try:
                subject_samples = self._process_single_subject(
                    subject, self.filter_params, self.resample_freq, self.pre_trial_duration
                )
                if subject_samples:
                    self.samples.extend(subject_samples)
                    logger.info(f"Created {len(subject_samples)} per-trial samples for {subject['subject_id']}")
            except Exception as e:
                logger.error(f"Error processing subject {subject['subject_id']}: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"Sequential processing completed in {total_time:.2f}s")
    
    @staticmethod
    def _process_single_subject(subject_info: Dict, filter_params: Dict, resample_freq: int, pre_trial_duration: float):
        """Process a single subject - static method for parallel processing"""
        try:
            # Extract subject information
            subject_id = subject_info['subject_id']
            sus_files = subject_info['sus_files']
            ccd_files = subject_info['ccd_files']
            participant = subject_info['participant']
            release = subject_info['release']
            
            # Create per-trial samples for this subject
            subject_samples = []
            
            try:
                # Get demographics
                age = participant.get('age')
                sex = 1 if participant.get('sex') == 'F' else 0
            except Exception as e:
                logger.error(f"Error getting demographics for {subject_id}: {e}")

            # Load and concatenate all SuS EEG data for this subject
            sus_eeg_data = []
            sus_events = []
            
            for sus_file in sus_files:
                try:
                    # Load SuS EEG data
                    raw = mne.io.read_raw_eeglab(str(sus_file), preload=True, verbose=False)
                    
                    # Apply preprocessing
                    raw.filter(**filter_params, verbose=False)
                    if raw.info['sfreq'] != resample_freq:
                        raw.resample(resample_freq, verbose=False)
                    
                    # Get events
                    events_file = sus_file.parent / sus_file.name.replace('_eeg.set', '_events.tsv')
                    if events_file.exists():
                        events_df = pd.read_csv(events_file, sep='\t')
                        sus_events.append((raw, events_df))
                        
                except Exception as e:
                    logger.warning(f"Error loading SuS file {sus_file}: {e}")
                    continue
            
            # Process each CCD file
            for ccd_file in ccd_files:
                try:
                    # Load CCD trials
                    ccd_trials = Challenge1Dataset._load_ccd_trials(ccd_file)
                    
                    # Create samples for each CCD trial
                    for trial in ccd_trials:
                        # Find corresponding SuS pre-trial epoch
                        sus_epoch = Challenge1Dataset._find_sus_pre_trial_epoch(
                            sus_events, trial['onset_time'], pre_trial_duration, resample_freq
                        )
                        
                        if sus_epoch is not None:
                            # Create sample
                            sample = {
                                'eeg_data': sus_epoch,
                                'targets': {
                                    'response_time': trial['response_time'],
                                    'hit_miss': trial['hit_miss'],
                                    'age': age,
                                    'sex': sex
                                },
                                'subject_id': subject_id,
                                'trial_id': trial['trial_id'],
                                'release': release
                            }
                            subject_samples.append(sample)
                            
                except Exception as e:
                    logger.warning(f"Error processing CCD file {ccd_file}: {e}")
                    continue
            
            return subject_samples
            
        except Exception as e:
            logger.error(f"Error processing subject {subject_info['subject_id']}: {e}")
            return []
    
    def _find_sus_files(self, subject_dir: Path) -> List[Path]:
        """Find SuS (Surround Suppression) EEG files"""
        sus_files = []
        for file_path in subject_dir.glob("*task-surroundSupp*_eeg.set"):
            sus_files.append(file_path)
        return sorted(sus_files)
    
    def _find_ccd_files(self, subject_dir: Path) -> List[Path]:
        """Find CCD (Contrast Change Detection) EEG files"""
        ccd_files = []
        for file_path in subject_dir.glob("*task-contrastChangeDetection*_eeg.set"):
            ccd_files.append(file_path)
        return sorted(ccd_files)
    
    @staticmethod
    def _load_ccd_trials(ccd_file: Path) -> List[Dict[str, Any]]:
        """Load CCD trials from events file"""
        try:
            events_file = ccd_file.parent / ccd_file.name.replace('_eeg.set', '_events.tsv')
            
            if not events_file.exists():
                logger.warning(f"Events file not found: {events_file}")
                return []
            
            events_df = pd.read_csv(events_file, sep='\t')
            
            trials = []
            current_trial = None
            trial_id = 0
            
            for _, row in events_df.iterrows():
                event_value = str(row['value'])
                onset_time = float(row['onset'])
                feedback = row.get('feedback', 'n/a')
                
                # Detect trial start
                if 'contrastTrial_start' in event_value:
                    current_trial = {
                        'trial_id': trial_id,
                        'start_time': onset_time,
                        'onset_time': onset_time,  # For compatibility
                        'contrast_change_time': None,
                        'button_press_time': None,
                        'response_time': None,
                        'hit_miss': None
                    }
                    trial_id += 1
                
                # Detect target (contrast change)
                elif 'target' in event_value and current_trial is not None:
                    current_trial['contrast_change_time'] = onset_time
                    current_trial['onset_time'] = onset_time  # Update with actual contrast change time
                
                # Detect button press
                elif 'buttonPress' in event_value and current_trial is not None:
                    current_trial['button_press_time'] = onset_time
                    
                    # Calculate response time
                    if current_trial['contrast_change_time'] is not None:
                        current_trial['response_time'] = onset_time - current_trial['contrast_change_time']
                    
                    # Determine hit/miss from feedback
                    if feedback == 'smiley_face':
                        current_trial['hit_miss'] = 1  # Hit
                    elif feedback == 'sad_face':
                        current_trial['hit_miss'] = 0  # Miss
                    else:
                        current_trial['hit_miss'] = 0  # Default to miss
                    
                    # Trial complete
                    if current_trial['contrast_change_time'] is not None:
                        trials.append(current_trial)
                    
                    current_trial = None
            
            logger.info(f"Extracted {len(trials)} CCD trials from {ccd_file}")
            return trials
            
        except Exception as e:
            logger.error(f"Error loading CCD trials from {ccd_file}: {e}")
            return []
    
    @staticmethod
    def _find_sus_pre_trial_epoch(sus_events: List[Tuple], contrast_change_time: float, 
                                 pre_trial_duration: float, sampling_rate: int) -> Optional[np.ndarray]:
        """
        Find SuS pre-trial epoch (2 seconds before CCD contrast change)
        
        Args:
            sus_events: List of (raw, events_df) tuples from SuS files
            contrast_change_time: Time of contrast change in CCD task
            pre_trial_duration: Duration of pre-trial epoch (2.0 seconds)
            sampling_rate: Sampling rate
            
        Returns:
            Pre-trial EEG epoch or None if not found
        """
        try:
            # Calculate pre-trial epoch window
            epoch_start_time = contrast_change_time - pre_trial_duration  # 2 seconds before
            epoch_end_time = contrast_change_time
            
            # Try to find epoch in SuS data
            for raw, events_df in sus_events:
                # Check if epoch time is within this SuS recording
                recording_duration = raw.n_times / raw.info['sfreq']
                
                if epoch_start_time >= 0 and epoch_end_time <= recording_duration:
                    # Convert to samples
                    start_sample = int(epoch_start_time * sampling_rate)
                    end_sample = int(epoch_end_time * sampling_rate)
                    
                    # Check if epoch is within data bounds
                    if start_sample < 0 or end_sample >= raw.n_times:
                        continue
                    
                    # Extract epoch
                    epoch_data = raw.get_data()[:, start_sample:end_sample]
                    
                    # Validate epoch shape
                    expected_samples = int(pre_trial_duration * sampling_rate)
                    if epoch_data.shape[1] != expected_samples:
                        logger.warning(f"Epoch data shape mismatch: {epoch_data.shape[1]} != {expected_samples}")
                        continue
                    
                    # Apply z-score normalization
                    normalized_epoch = np.zeros_like(epoch_data)
                    for ch in range(epoch_data.shape[0]):
                        ch_data = epoch_data[ch, :]
                        if np.std(ch_data) > 1e-8:
                            normalized_epoch[ch, :] = (ch_data - np.mean(ch_data)) / np.std(ch_data)
                        else:
                            normalized_epoch[ch, :] = ch_data
                    
                    return normalized_epoch
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting SuS pre-trial epoch: {e}")
            return None

    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a per-trial sample.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (SuS pre-trial EEG tensor, targets dictionary)
            
        Raises:
            IndexError: If idx is out of bounds
            ValueError: If sample data is corrupted or invalid
        """
        if idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of bounds for dataset with {len(self.samples)} samples")
        
        try:
            sample = self.samples[idx]
            
            # Get SuS pre-trial EEG epoch
            sus_epoch = sample['eeg_data']  # Shape: (n_channels, n_timepoints)
            
            # Validate EEG data
            if sus_epoch is None:
                raise ValueError(f"EEG data is None for sample {idx}")
            
            if not isinstance(sus_epoch, np.ndarray):
                raise ValueError(f"EEG data must be numpy array, got {type(sus_epoch)}")
            
            if sus_epoch.shape[0] != self.channel_count:
                raise ValueError(f"Expected {self.channel_count} channels, got {sus_epoch.shape[0]}")
            
            # Create tensors
            sus_eeg_tensor = torch.FloatTensor(sus_epoch)
            
            # Create target tensors with validation
            targets = {}
            
            # Response time (regression)
            rt = sample['targets']['response_time']
            if rt is None or np.isnan(rt) or rt < 0:
                logger.error(f"Invalid response time {rt} for sample {idx}, using default")
                rt = 0  # Default response time
            targets['response_time'] = torch.FloatTensor([rt])
            
            # Hit/miss (classification)
            hm = sample['targets']['hit_miss']
            if hm not in [0, 1]:
                logger.error(f"Invalid hit/miss {hm} for sample {idx}, using default")
                hm = 0  # Default to miss
            targets['hit_miss'] = torch.LongTensor([hm])
            
            # Age (regression)
            age = sample['targets']['age']
            if age is None or np.isnan(age) or age < 0:
                logger.error(f"Invalid age {age} for sample {idx}, using default")
                age = 0  # Default age
            targets['age'] = torch.FloatTensor([age])
            
            # Sex (classification)
            sex = sample['targets']['sex']
            if sex not in [0, 1]:
                logger.error(f"Invalid sex {sex} for sample {idx}, using default")
                sex = 0  # Default to male
            targets['sex'] = torch.LongTensor([sex])
            
            # Apply transforms if provided
            if self.transforms:
                sus_eeg_tensor = self.transforms(sus_eeg_tensor)
            
            return sus_eeg_tensor, targets
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            raise ValueError(f"Failed to load sample {idx}: {e}")
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample"""
        return {
            'subject_id': self.samples[idx]['subject_id'],
            'trial_id': self.samples[idx]['trial_id'],
            'release': self.samples[idx]['release']
        }

def create_challenge1_dataloaders(
    data_dir: str,
    config: Dict[str, Any],
    batch_size: int = 32,
    num_workers: int = 4,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for Challenge 1 using per-trial approach
    
    Args:
        data_dir: Path to HBN BIDS EEG dataset
        config: Configuration dictionary
        batch_size: Batch size for training
        num_workers: Number of data loader workers
        test_size: Test set size (fraction)
        val_size: Validation set size (fraction)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset to get all subjects
    full_dataset = Challenge1Dataset(
        data_dir=data_dir,
        config=config,
        split='train'  # Use train split to get all data
    )
    
    # Get unique subjects
    subjects = list(set(s['subject_id'] for s in full_dataset.samples))
    
    # Split subjects (not samples) for proper cross-validation
    train_subjects, test_subjects = train_test_split(
        subjects, test_size=test_size, random_state=random_state
    )
    
    # Split train subjects into train and val subjects
    # val_size is the fraction of train subjects that will be used for validation
    train_subjects, val_subjects = train_test_split(
        train_subjects, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    subject_split = {
        'train': train_subjects,
        'val': val_subjects,
        'test': test_subjects
    }
    
    # Create datasets for each split
    train_dataset = Challenge1Dataset(
        data_dir=data_dir,
        config=config,
        split='train',
        subject_split=subject_split
    )
    
    val_dataset = Challenge1Dataset(
        data_dir=data_dir,
        config=config,
        split='val',
        subject_split=subject_split
    )
    
    test_dataset = Challenge1Dataset(
        data_dir=data_dir,
        config=config,
        split='test',
        subject_split=subject_split
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    logger.info(f"Created Challenge 1 data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_subjects)} subjects")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_subjects)} subjects")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_subjects)} subjects")
    
    return train_loader, val_loader, test_loader

# Keep the original EEGFoundationDataset for Challenge 2
class EEGFoundationDataset(Dataset):
    """
    Legacy dataset for Challenge 2 (Psychopathology prediction)
    """
    
    def __init__(
        self,
        data_dir: str,
        task_filter: List[str] = None,
        preprocessing: str = 'full',
        split: str = 'train',
        subject_split: Dict[str, List[str]] = None,
        cache_dir: str = None,
        transforms: Optional[callable] = None
    ):
        """
        Initialize EEG Foundation Dataset for Challenge 2
        
        Args:
            data_dir: Path to HBN_BIDS_EEG directory
            task_filter: List of task names to include
            preprocessing: Preprocessing level ('minimal', 'bandpass', 'full')
            split: Data split ('train', 'val', 'test')
            subject_split: Pre-defined subject splits
            cache_dir: Directory to cache processed data
            transforms: Additional transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.task_filter = task_filter or [
            'surroundSupp', 'contrastChangeDetection', 'naturalistic',
            'restingState', 'auditoryOddball', 'visualOddball'
        ]
        self.preprocessing = preprocessing
        self.split = split
        self.subject_split = subject_split
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transforms = transforms
        
        # Get releases
        self.releases = [d.name for d in self.data_dir.iterdir() if d.is_dir() and 'cmi_bids' in d.name]
        
        # Initialize data structures
        self.metadata = {}
        self.samples = []
        
        # Load and process data
        self._load_metadata()
        self._setup_preprocessing()
        self._create_samples()
        
        logger.info(f"Challenge 2 Dataset ({split}): {len(self.samples)} samples")
    
    def _load_metadata(self):
        """Load participant metadata from all releases"""
        for release in self.releases:
            release_path = self.data_dir / release
            participants_file = release_path / 'participants.tsv'
            
            if participants_file.exists():
                df = pd.read_csv(participants_file, sep='\t')
                self.metadata[release] = df
                logger.info(f"Loaded {len(df)} participants from {release}")
            else:
                logger.warning(f"No participants.tsv found in {release}")
    
    def _create_samples(self):
        """Create list of all valid samples for Challenge 2"""
        for release, metadata_df in self.metadata.items():
            release_path = self.data_dir / release
            
            for _, participant in metadata_df.iterrows():
                subject_id = participant['participant_id']
                subject_dir = release_path / subject_id / 'eeg'
                
                if not subject_dir.exists():
                    continue
                
                # Filter subjects based on split if provided
                if self.subject_split:
                    if not any(subject_id in split for split in self.subject_split.values()):
                        continue
                
                # Get EEG files for this subject
                eeg_files = list(subject_dir.glob('*.set'))
                
                for eeg_file in eeg_files:
                    # Extract task name from filename
                    task_name = self._extract_task_name(eeg_file.name)
                    
                    if task_name in self.task_filter:
                        sample = {
                            'subject_id': subject_id,
                            'release': release,
                            'task': task_name,
                            'file_path': str(eeg_file),
                            'metadata': participant.to_dict()
                        }
                        self.samples.append(sample)
        
        logger.info(f"Created {len(self.samples)} samples across {len(self.task_filter)} tasks")
    
    def _extract_task_name(self, filename: str) -> str:
        """Extract task name from EEG filename"""
        # Implementation remains the same as before
        for task in self.task_filter:
            if task in filename:
                return task
        return 'unknown'
    
    def _setup_preprocessing(self):
        """Setup preprocessing pipelines"""
        if self.preprocessing == 'minimal':
            self.filter_params = None
        elif self.preprocessing == 'bandpass':
            self.filter_params = {'l_freq': 0.5, 'h_freq': 70.0}
        elif self.preprocessing == 'full':
            self.filter_params = {'l_freq': 0.5, 'h_freq': 70.0}
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get a sample for Challenge 2"""
        # Implementation for Challenge 2 - placeholder
        sample = self.samples[idx]
        
        # Return dummy data for now
        dummy_eeg = torch.randn(129, 256)
        dummy_targets = {
            'p_factor': torch.FloatTensor([0.5]),
            'attention': torch.FloatTensor([0.5]),
            'internalizing': torch.FloatTensor([0.5]),
            'externalizing': torch.FloatTensor([0.5])
        }
        
        return dummy_eeg, dummy_targets

# Main function for testing
if __name__ == '__main__':
    # Test Challenge 1 dataset
    config = {
        'data': {
            'channel_count': 129,
            'sampling_rate': 256
        },
        'preprocessing': {
            'filter_low': 0.5,
            'filter_high': 70.0,
            'resample_freq': 256,
            'epoch_length': 1.0
        },
        'parallel': {
            'enabled': True,
            'max_workers': 4,
            'batch_size': 10
        }
    }
    
    print("Testing Challenge 1 Dataset (Per-Trial)...")
    dataset = Challenge1Dataset(
        data_dir='./data/raw/HBN_BIDS_EEG',
        config=config,
        split='train'
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        sample_eeg, sample_targets = dataset[0]
        print(f"SuS EEG shape: {sample_eeg.shape}")
        print(f"Targets: {list(sample_targets.keys())}")
        print("Challenge 1 dataset test completed successfully!")
    else:
        print("No samples found - check data directory path")