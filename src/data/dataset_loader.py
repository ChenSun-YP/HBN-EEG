"""
EEG Foundation Challenge Data Loader
Supports both Challenge 1 (Cross-Task) and Challenge 2 (Psychopathology)

CHALLENGE 1 OFFICIAL CONSTRAINTS:
- Training Input: CCD EEG data (X1) as primary input
- Optional Additional Features: SuS EEG data (X2) and demographics/psychopathology (P)
- Prediction Targets: CCD behavioral outcomes per trial:
  * Response time (regression)
  * Hit/miss accuracy (binary classification)

Official Challenge 1 Data Structure:
- X1 ∈ ℝ^(c×n×t1): CCD EEG recording (c=128 channels, n≈70 epochs, t1=2 seconds)
- X2 ∈ ℝ^(c×t2): SuS EEG recording (c=128 channels, t2=total number of SuS trials)
- P ∈ ℝ^7: Subject traits (3 demographics + 4 psychopathology factors)
- Y: CCD behavioral outcomes (response time, hit/miss) for each trial

EEG Foundation Challenge Requirements:
- Use 128 channels (standard EEG layout)
- Include psychopathology factors (p_factor, attention, internalizing, externalizing)
- Include handedness (ehq_total), age, and sex
- Release-based splits: R1-11 excluding R5 for train, R5 for val
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
from bids import BIDSLayout
import mne_bids

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
    Challenge 1 dataset compliant with official EEG Foundation Challenge requirements
    
    Creates samples where:
    - X1 (Primary Input): CCD EEG epochs (2 seconds per trial)
    - X2 (Optional): SuS EEG data as additional features
    - P (Optional): Demographics and psychopathology factors
    - Y (Target): CCD behavioral outcomes (response_time, hit_miss) for each trial
    
    Each sample corresponds to one CCD trial with its associated EEG data and CCD behavioral outcome.
    
    EEG Foundation Challenge Requirements:
    - Use 128 channels (standard EEG layout)
    - Include psychopathology factors (p_factor, attention, internalizing, externalizing)
    - Include handedness (ehq_total), age, and sex
    - Release-based splits: R1-11 excluding R5 for train, R5 for val
    """
    
    def __init__(self, data_dir: str, config: Dict[str, Any], split: str = 'train', 
                 subject_split: Optional[Dict[str, List[str]]] = None, transforms=None, 
                 samples: Optional[List] = None, releases: Optional[List[str]] = None,
                 use_demographics: bool = True, use_sus_eeg: bool = False,
                 shared_layouts: Optional[Dict[str, Any]] = None):
        """
        Initialize Challenge 1 dataset
        
        Args:
            data_dir: Path to HBN BIDS EEG dataset
            config: Configuration dictionary
            split: Dataset split ('train', 'val', 'test')
            subject_split: Optional pre-defined subject splits
            transforms: Optional transforms to apply
            samples: Pre-processed samples (for reuse)
            releases: Specific releases for this split
            use_demographics: Whether to include demographics and psychopathology as additional features
            use_sus_eeg: Whether to include SuS EEG data as additional features
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.subject_split = subject_split
        self.transforms = transforms
        self.releases = releases  # Specific releases for this split
        self.use_demographics = use_demographics
        self.use_sus_eeg = use_sus_eeg
        self.shared_layouts = shared_layouts
        
        # Get dataset configuration
        self.channel_count = config['data']['channel_count']  # Should be 128
        self.sampling_rate = config['data']['sampling_rate']
        self.epoch_duration = 2.0  # 2 seconds per CCD epoch as specified
        
        # Validate channel count
        if self.channel_count != 128:
            logger.warning(f"Channel count should be 128 for EEG Foundation Challenge, got {self.channel_count}")
        
        # If samples are provided, use them directly (for reuse)
        if samples is not None:
            self.samples = samples
            logger.info(f"Reusing {len(self.samples)} pre-processed samples for {split} split")
            return
        
        # Try to load from cache first (if enabled)
        cache_config = config.get('caching', {})
        cache_enabled = cache_config.get('enabled', True)
        
        if cache_enabled:
            cache_file = self._get_cache_file_path()
            if self._load_from_cache(cache_file):
                logger.info(f"Loaded {len(self.samples)} samples from cache for {split} split")
                return
        
        # Parallel processing configuration
        self.use_parallel = config.get('parallel', {}).get('enabled', True)
        self.max_workers = config.get('parallel', {}).get('max_workers', multiprocessing.cpu_count())
        self.batch_size = config.get('parallel', {}).get('batch_size', 100)  # Process subjects in batches
        
        logger.info(f"Parallel processing: {'enabled' if self.use_parallel else 'disabled'}")
        if self.use_parallel:
            logger.info(f"Using {self.max_workers} workers with batch size {self.batch_size}")
        
        # Setup preprocessing
        self._setup_preprocessing()
        
        # Initialize BIDS layout for file discovery (use shared layouts if provided)
        if self.shared_layouts is not None:
            logger.info(f"Using shared BIDS layouts for {split} split")
            self.layouts = self.shared_layouts
            # Keep a reference to the first layout for backward compatibility
            if self.layouts:
                self.layout = list(self.layouts.values())[0]
            else:
                self.layout = None
        else:
            self._setup_bids_layout()
        
        # Load metadata and create quality control whitelist
        self.metadata, self.qc_whitelist = self._load_metadata_and_qc()
        
        # Create samples
        self.samples = []
        self._create_samples()
        
        # Save to cache (if enabled)
        if cache_enabled:
            self._save_to_cache(cache_file)
        
        logger.info(f"Created {len(self.samples)} samples for {split} split")
    
    def _setup_preprocessing(self):
        """Setup preprocessing parameters"""
        preprocess_config = self.config.get('preprocessing', {})
        
        self.filter_params = {
            'l_freq': preprocess_config.get('filter_low', 0.5),
            'h_freq': preprocess_config.get('filter_high', 70.0)
        }
        
        self.resample_freq = preprocess_config.get('resample_freq', 256)
    
    def _setup_bids_layout(self):
        """Initialize BIDS layouts for all release directories"""
        try:
            # Discover all release directories
            release_dirs = []
            for d in self.data_dir.iterdir():
                if d.is_dir() and d.name.startswith('cmi_bids_R'):
                    release_dirs.append(str(d))
            
            logger.info(f"Found {len(release_dirs)} release directories: {[Path(d).name for d in release_dirs]}")
            
            # Initialize individual BIDS layouts for each release
            # This approach works reliably and gives access to all subjects
            self.layouts = {}
            total_subjects = 0
            
            for release_dir in release_dirs:
                try:
                    layout = BIDSLayout(release_dir)
                    subjects = layout.get_subjects()
                    self.layouts[release_dir] = layout
                    total_subjects += len(subjects)
                    logger.info(f"✅ {Path(release_dir).name}: {len(subjects)} subjects")
                except Exception as e:
                    logger.warning(f"Failed to initialize layout for {release_dir}: {e}")
            
            logger.info(f"BIDS layouts initialized successfully: {len(self.layouts)} releases, {total_subjects} total subjects")
            
            # Keep a reference to the first layout for backward compatibility
            if self.layouts:
                self.layout = list(self.layouts.values())[0]
            else:
                logger.warning("No BIDS layouts initialized")
                self.layout = None
            
        except Exception as e:
            logger.error(f"Failed to initialize BIDS layouts: {e}")
            self.layouts = {}
            self.layout = None
    
    def _load_metadata_and_qc(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Load BIDS metadata and quality control information
        
        Returns:
            Tuple of (metadata_dict, qc_whitelist_dataframe)
        """
        metadata = {}
        qc_dataframes = []
        
        # Check if quick test mode is enabled
        quick_test = self.config.get('quick_test', {})
        quick_test_enabled = quick_test.get('enabled', False)
        allowed_bids_dirs = quick_test.get('bids_dirs', None) if quick_test_enabled else None
        
        # Determine which releases to load
        if self.releases is not None:
            # Use specified releases for this split
            releases_to_load = self.releases
            logger.info(f"Loading metadata for specified releases: {releases_to_load}")
        elif quick_test_enabled and allowed_bids_dirs:
            # Quick test mode
            releases_to_load = allowed_bids_dirs
            logger.info(f"Quick test mode: Loading metadata for {releases_to_load}")
        else:
            # Load all available releases
            releases_to_load = [d.name for d in self.data_dir.iterdir() 
                              if d.is_dir() and d.name.startswith('cmi_bids_R')]
            logger.info(f"Loading metadata for all available releases: {releases_to_load}")
        
        for release_name in releases_to_load:
            release_dir = self.data_dir / release_name
            
            if not release_dir.exists():
                logger.warning(f"Release directory not found: {release_dir}")
                continue
            
            # Load participants.tsv
            participants_file = release_dir / 'participants.tsv'
            if participants_file.exists():
                df = pd.read_csv(participants_file, sep='\t')
                metadata[release_name] = df
                logger.info(f"Loaded metadata for {release_name}: {len(df)} participants")
            else:
                logger.warning(f"No participants.tsv found in {release_name}")
            
            # Load quality control files
            code_dir = release_dir / 'code'
            if code_dir.exists():
                qc_files = list(code_dir.glob('*_quality_table.tsv'))
                for qc_file in qc_files:
                    try:
                        # Read QC file as CSV (files are comma-separated despite .tsv extension)
                        qc_df = pd.read_csv(qc_file, sep=',')
                        qc_df['release'] = release_name
                        qc_df['qc_file'] = qc_file.name
                        qc_dataframes.append(qc_df)
                        logger.info(f"Loaded QC data from {qc_file.name}: {len(qc_df)} entries")
                    except Exception as e:
                        logger.warning(f"Failed to load QC file {qc_file}: {e}")
        
        # Combine all QC dataframes
        if qc_dataframes:
            qc_combined = pd.concat(qc_dataframes, ignore_index=True)
            # Filter for valid recordings (key_events_exist == 1)
            qc_whitelist = qc_combined[qc_combined['key_events_exist'] == 1].copy()
            logger.info(f"QC whitelist created: {len(qc_whitelist)} valid recordings from {len(qc_combined)} total")
        else:
            qc_whitelist = pd.DataFrame()
            logger.warning("No QC files found, proceeding without quality control filtering")
        
        return metadata, qc_whitelist
    
    def _create_samples(self):
        """Create samples using CCD EEG data as primary input"""
        
        # Check if quick test mode is enabled
        quick_test = self.config.get('quick_test', {})
        quick_test_enabled = quick_test.get('enabled', False)
        max_subjects = quick_test.get('max_subjects', None) if quick_test_enabled else None
        
        # Use BIDS layout to discover files if available
        if self.layout is not None:
            self._create_samples_with_bids()
        else:
            self._create_samples_manual()
    
    def _create_samples_with_bids(self):
        """Create samples using BIDS layouts for file discovery for current split releases"""
        logger.info(f"Using BIDS layouts for file discovery for {self.split} split releases")
        
        # Get all CCD files from BIDS layouts for current split releases only
        ccd_files = []
        sus_files = []
        
        # Determine which releases to process for this split
        if self.releases is not None:
            # Use specified releases for this split
            releases_to_process = self.releases
            logger.info(f"Processing specified releases for {self.split} split: {releases_to_process}")
        else:
            # Fallback: process all available releases
            releases_to_process = [Path(release_dir).name for release_dir in self.layouts.keys()]
            logger.info(f"Processing all available releases for {self.split} split: {releases_to_process}")
        
        # Only process releases needed for this split
        for release_dir, layout in self.layouts.items():
            release_name = Path(release_dir).name
            
            # Skip if this release is not needed for the current split
            if release_name not in releases_to_process:
                logger.debug(f"Skipping {release_name} (not needed for {self.split} split)")
                continue
            
            try:
                # Get CCD files from this release
                release_ccd_files = layout.get(
                    task='contrastChangeDetection',
                    suffix='eeg',
                    extension='.set',
                    return_type='filename'
                )
                ccd_files.extend(release_ccd_files)
                
                # Get SuS files if using as additional features (X2)
                if self.use_sus_eeg:
                    release_sus_files = layout.get(
                        task='surroundSupp',
                        suffix='eeg',
                        extension='.set',
                        return_type='filename'
                    )
                    sus_files.extend(release_sus_files)
                
                logger.info(f"✅ {release_name}: {len(release_ccd_files)} CCD files, {len(release_sus_files) if self.use_sus_eeg else 0} SuS files")
                
            except Exception as e:
                logger.warning(f"Failed to get files from {release_dir}: {e}")
        
        logger.info(f"Found {len(ccd_files)} total CCD files and {len(sus_files)} total SuS files for {self.split} split")
        
        # Create subject mapping
        subject_files = {}
        for file_path in ccd_files + sus_files:
            # Extract BIDS entities from file path manually
            try:
                path = Path(file_path)
                parts = path.parts
                
                # Extract release (e.g., "cmi_bids_R1" from path)
                release = parts[-4] if len(parts) >= 4 else 'unknown'
                
                # Extract subject from filename
                filename = path.name
                subject_id = filename.split('_')[0] if filename.startswith('sub-') else 'unknown'
                
                # Extract task and run from filename
                task = 'unknown'
                run = None
                
                if '_task-' in filename:
                    task_start = filename.find('_task-') + 6
                    task_end = filename.find('_', task_start)
                    if task_end == -1:
                        task_end = filename.find('.', task_start)
                    task = filename[task_start:task_end] if task_end != -1 else filename[task_start:]
                
                if '_run-' in filename:
                    run_start = filename.find('_run-') + 5
                    run_end = filename.find('_', run_start)
                    if run_end == -1:
                        run_end = filename.find('.', run_start)
                    run = filename[run_start:run_end] if run_end != -1 else filename[run_start:]
                
            except Exception as e:
                logger.warning(f"Failed to parse BIDS entities from {file_path}: {e}")
                continue
            
            if subject_id not in subject_files:
                subject_files[subject_id] = {'ccd_files': [], 'sus_files': []}
            
            if task == 'contrastChangeDetection':
                subject_files[subject_id]['ccd_files'].append(file_path)
            elif task == 'surroundSupp':
                subject_files[subject_id]['sus_files'].append(file_path)
        
        # Process subjects
        all_subjects = []
        for subject_id, files in subject_files.items():
            if not files['ccd_files']:  # Must have CCD data
                continue
            
            # Get participant metadata
            participant = self._get_participant_metadata(subject_id)
            if participant is None:
                continue
            
            # Check QC whitelist if available
            if not self.qc_whitelist.empty:
                if not self._is_subject_in_qc_whitelist(subject_id):
                    logger.debug(f"Subject {subject_id} not in QC whitelist, skipping")
                    continue
            
            # Determine release from BIDS path
            release = None
            if files['ccd_files']:
                try:
                    # Extract release from file path manually
                    file_path = Path(files['ccd_files'][0])
                    parts = file_path.parts
                    release = parts[-4] if len(parts) >= 4 else 'unknown'
                except Exception as e:
                    logger.warning(f"Failed to extract release from {files['ccd_files'][0]}: {e}")
                    release = 'unknown'
            
            all_subjects.append({
                'subject_id': subject_id,
                'ccd_files': files['ccd_files'],
                'sus_files': files['sus_files'],
                'participant': participant,
                'release': release
            })
        
        logger.info(f"Processing {len(all_subjects)} subjects with valid data...")
        
        # Process subjects in parallel or sequentially
        if self.use_parallel and len(all_subjects) > 1:
            self._process_subjects_parallel(all_subjects)
        else:
            self._process_subjects_sequential(all_subjects)
        
        # Quick test mode check
        if self.config.get('quick_test', {}).get('enabled', False):
            max_subjects = self.config['quick_test'].get('max_subjects', 5)
            if len(all_subjects) > max_subjects:
                logger.info(f"Quick test mode: limiting to {max_subjects} subjects")
                # Limit the samples to only the first max_subjects
                total_samples_before = len(self.samples)
                self.samples = self.samples[:max_subjects * 50]  # Approximate samples per subject
                logger.info(f"Quick test mode: reduced samples from {total_samples_before} to {len(self.samples)}")
    
    def _create_samples_manual(self):
        """Create samples using manual file discovery (fallback)"""
        logger.info("Using manual file discovery")
        
        # Collect all subjects to process
        all_subjects = []
        for release, metadata_df in self.metadata.items():
            release_path = self.data_dir / release
            
            for _, participant in metadata_df.iterrows():
                subject_id = participant['participant_id']
                
                # Apply subject split if provided
                if self.subject_split:
                    if subject_id not in self.subject_split[self.split]:
                        continue
                
                subject_dir = release_path / subject_id / 'eeg'
                
                if not subject_dir.exists():
                    continue
                
                # Find CCD files (primary input)
                ccd_files = self._find_ccd_files(subject_dir)
                
                # Find SuS files (optional additional features)
                sus_files = []
                if self.use_sus_eeg:
                    sus_files = self._find_sus_files(subject_dir)
                
                if not ccd_files:  # Must have CCD data
                    continue
                
                all_subjects.append({
                    'subject_id': subject_id,
                    'ccd_files': ccd_files,
                    'sus_files': sus_files,
                    'participant': participant,
                    'release': release
                })
        
        logger.info(f"Processing {len(all_subjects)} subjects...")
        
        # Process subjects in parallel or sequentially
        if self.use_parallel and len(all_subjects) > 1:
            self._process_subjects_parallel(all_subjects)
        else:
            self._process_subjects_sequential(all_subjects)
        
        # Quick test mode check
        if self.config.get('quick_test', {}).get('enabled', False):
            max_subjects = self.config['quick_test'].get('max_subjects', 5)
            if len(all_subjects) > max_subjects:
                logger.info(f"Quick test mode: limiting to {max_subjects} subjects")
                # Limit the samples to only the first max_subjects
                total_samples_before = len(self.samples)
                self.samples = self.samples[:max_subjects * 50]  # Approximate samples per subject
                logger.info(f"Quick test mode: reduced samples from {total_samples_before} to {len(self.samples)}")
    
    def _get_participant_metadata(self, subject_id: str) -> Optional[Dict]:
        """Get participant metadata from all releases"""
        for release, metadata_df in self.metadata.items():
            participant_data = metadata_df[metadata_df['participant_id'] == subject_id]
            if not participant_data.empty:
                return participant_data.iloc[0].to_dict()
        return None
    
    def _is_subject_in_qc_whitelist(self, subject_id: str) -> bool:
        """Check if subject is in QC whitelist"""
        if self.qc_whitelist.empty:
            return True
        
        # Remove 'sub-' prefix for QC comparison
        subject_id_clean = subject_id.replace('sub-', '')
        return subject_id_clean in self.qc_whitelist['Row'].values
    
    def _process_subjects_parallel(self, all_subjects: List[Dict]):
        """Process subjects in parallel for maximum CPU utilization"""
        start_time = time.time()
        
        # Use ProcessPoolExecutor for true parallel processing across all CPU cores
        # This will bypass the GIL and utilize all cores effectively
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Create partial function with preprocessing parameters
            process_func = partial(
                self._process_single_subject,
                filter_params=self.filter_params,
                resample_freq=self.resample_freq,
                epoch_duration=self.epoch_duration,
                use_demographics=self.use_demographics,
                use_sus_eeg=self.use_sus_eeg
            )
            
            # Process all subjects in parallel with optimal chunking
            chunk_size = max(1, len(all_subjects) // (self.max_workers * 4))  # Dynamic chunking
            
            logger.info(f"Processing {len(all_subjects)} subjects with {self.max_workers} workers, chunk_size={chunk_size}")
            
            # Use map with chunksize for optimal load balancing
            results = list(executor.map(process_func, all_subjects, chunksize=chunk_size))
            
            # Collect all results
            for subject_samples in results:
                if subject_samples:
                    self.samples.extend(subject_samples)
        
        total_time = time.time() - start_time
        logger.info(f"Parallel processing completed in {total_time:.2f}s")
    
    def _process_subjects_sequential(self, all_subjects: List[Dict]):
        """Process subjects sequentially (fallback)"""
        start_time = time.time()
        
        for i, subject in enumerate(all_subjects):
            try:
                subject_samples = self._process_single_subject(
                    subject, self.filter_params, self.resample_freq, self.epoch_duration, 
                    self.use_demographics, self.use_sus_eeg
                )
                if subject_samples:
                    self.samples.extend(subject_samples)
                    logger.info(f"Created {len(subject_samples)} samples for {subject['subject_id']}")
            except Exception as e:
                logger.error(f"Error processing subject {subject['subject_id']}: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"Sequential processing completed in {total_time:.2f}s")
    
    @staticmethod
    def _process_single_subject(subject_info: Dict, filter_params: Dict, resample_freq: int, 
                               epoch_duration: float, use_demographics: bool, use_sus_eeg: bool):
        """Process a single subject - static method for parallel processing"""
        try:
            # Extract subject information
            subject_id = subject_info['subject_id']
            ccd_files = subject_info['ccd_files']
            sus_files = subject_info['sus_files']
            participant = subject_info['participant']
            release = subject_info.get('release')
            
            # Create samples for this subject
            subject_samples = []
            
            try:
                # Get demographics and psychopathology factors
                age = participant.get('age')
                sex = 1 if participant.get('sex') == 'F' else 0
                handedness = participant.get('ehq_total')
                
                # Get psychopathology factors
                p_factor = participant.get('p_factor')
                attention = participant.get('attention')
                internalizing = participant.get('internalizing')
                externalizing = participant.get('externalizing')
                
                # Handle missing values
                if handedness == 'n/a' or pd.isna(handedness):
                    handedness = 0.0  # Default to neutral handedness
                else:
                    handedness = float(handedness)
                
                if p_factor == 'n/a' or pd.isna(p_factor):
                    p_factor = 0.0
                else:
                    p_factor = float(p_factor)
                
                if attention == 'n/a' or pd.isna(attention):
                    attention = 0.0
                else:
                    attention = float(attention)
                
                if internalizing == 'n/a' or pd.isna(internalizing):
                    internalizing = 0.0
                else:
                    internalizing = float(internalizing)
                
                if externalizing == 'n/a' or pd.isna(externalizing):
                    externalizing = 0.0
                else:
                    externalizing = float(externalizing)
                
            except Exception as e:
                logger.error(f"Error getting demographics for {subject_id}: {e}")
                return []

            # Load SuS EEG data if requested (additional features X2)
            sus_eeg_data = None
            if use_sus_eeg and sus_files:
                sus_eeg_data = Challenge1Dataset._load_sus_eeg_data(sus_files, filter_params, resample_freq)
            
            # Process each CCD file
            for ccd_file in ccd_files:
                try:
                    # Load CCD EEG data (primary input X1)
                    ccd_eeg_data = Challenge1Dataset._load_ccd_eeg_data(ccd_file, filter_params, resample_freq)
                    if ccd_eeg_data is None:
                        continue
                    
                    # Load CCD trials and behavioral outcomes
                    ccd_trials = Challenge1Dataset._load_ccd_trials(ccd_file)
                    
                    # Create samples for each CCD trial
                    for trial in ccd_trials:
                        # Extract CCD EEG epoch for this trial
                        ccd_epoch = Challenge1Dataset._extract_ccd_epoch(
                            ccd_eeg_data, trial, epoch_duration, resample_freq
                        )
                        
                        if ccd_epoch is not None:
                            # Create sample with CCD EEG as primary input
                            sample = {
                                'ccd_eeg_data': ccd_epoch,  # Primary input X1
                                'demographics': {
                                    'age': age,
                                    'sex': sex,
                                    'handedness': handedness,
                                    'p_factor': p_factor,
                                    'attention': attention,
                                    'internalizing': internalizing,
                                    'externalizing': externalizing
                                } if use_demographics else None,
                                'targets': {
                                    'response_time': trial['response_time'],
                                    'hit_miss': trial['hit_miss']
                                },
                                'subject_id': subject_id,
                                'trial_id': trial['trial_id'],
                                'release': release
                            }
                            
                            # Only add SuS EEG data if it's actually being used
                            if use_sus_eeg and sus_eeg_data is not None:
                                sample['sus_eeg_data'] = sus_eeg_data
                            
                            subject_samples.append(sample)
                            
                except Exception as e:
                    logger.warning(f"Error processing CCD file {ccd_file}: {e}")
                    continue
            
            return subject_samples
            
        except Exception as e:
            logger.error(f"Error processing subject {subject_info['subject_id']}: {e}")
            return []
    
    @staticmethod
    def _load_ccd_eeg_data(ccd_file: str, filter_params: Dict, resample_freq: int) -> Optional[np.ndarray]:
        """Load CCD EEG data from .set file"""
        try:
            # Load EEG data using MNE directly
            raw = mne.io.read_raw_eeglab(ccd_file, preload=True, verbose=False)
            
            # Apply preprocessing
            if filter_params:
                raw.filter(l_freq=filter_params['l_freq'], h_freq=filter_params['h_freq'], verbose=False)
            
            if resample_freq:
                raw.resample(resample_freq, verbose=False)
            
            # Ensure 128 channels (EEG Foundation Challenge requirement)
            if raw.info['nchan'] != 128:
                # More efficient channel selection - only log once per subject
                if raw.info['nchan'] > 128:
                    # Take first 128 channels (most common case)
                    raw.pick(raw.ch_names[:128])
                elif raw.info['nchan'] < 128:
                    logger.warning(f"Not enough EEG channels ({raw.info['nchan']}), skipping")
                    return None
                else:
                    # Exactly 128 channels, no action needed
                    pass
            
            # Get data and apply z-score normalization
            data = raw.get_data()
            
            # Check for NaN or Inf in raw data
            if np.isnan(data).any():
                logger.warning(f"NaN detected in raw EEG data from {ccd_file}")
                return None
            if np.isinf(data).any():
                logger.warning(f"Inf detected in raw EEG data from {ccd_file}")
                return None
            
            normalized_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                ch_data = data[ch, :]
                if np.std(ch_data) > 1e-8:
                    normalized_data[ch, :] = (ch_data - np.mean(ch_data)) / np.std(ch_data)
                else:
                    normalized_data[ch, :] = ch_data
            
            # Check for NaN or Inf in normalized data
            if np.isnan(normalized_data).any():
                logger.warning(f"NaN detected in normalized EEG data from {ccd_file}")
                return None
            if np.isinf(normalized_data).any():
                logger.warning(f"Inf detected in normalized EEG data from {ccd_file}")
                return None
            
            return normalized_data
                
        except Exception as e:
            logger.warning(f"Error loading CCD EEG data from {ccd_file}: {e}")
            return None
    
    @staticmethod
    def _load_sus_eeg_data(sus_files: List[str], filter_params: Dict, resample_freq: int) -> Optional[np.ndarray]:
        """Load SuS EEG data from .set files"""
        try:
            sus_data = []
            
            for sus_file in sus_files:
                # Load EEG data using MNE directly
                raw = mne.io.read_raw_eeglab(sus_file, preload=True, verbose=False)
                
                # Apply preprocessing
                if filter_params:
                    raw.filter(l_freq=filter_params['l_freq'], h_freq=filter_params['h_freq'], verbose=False)
                
                if resample_freq:
                    raw.resample(resample_freq, verbose=False)
                
                # Ensure 128 channels (EEG Foundation Challenge requirement)
                if raw.info['nchan'] != 128:
                    # More efficient channel selection - only log once per subject
                    if raw.info['nchan'] > 128:
                        # Take first 128 channels (most common case)
                        raw.pick(raw.ch_names[:128])
                    elif raw.info['nchan'] < 128:
                        logger.warning(f"Not enough EEG channels ({raw.info['nchan']}), skipping")
                        continue
                    else:
                        # Exactly 128 channels, no action needed
                        pass
                
                # Get data and apply z-score normalization
                data = raw.get_data()
                
                # Check for NaN or Inf in raw data
                if np.isnan(data).any():
                    logger.warning(f"NaN detected in raw SuS EEG data from {sus_file}")
                    continue
                if np.isinf(data).any():
                    logger.warning(f"Inf detected in raw SuS EEG data from {sus_file}")
                    continue
                
                normalized_data = np.zeros_like(data)
                for ch in range(data.shape[0]):
                    ch_data = data[ch, :]
                    if np.std(ch_data) > 1e-8:
                        normalized_data[ch, :] = (ch_data - np.mean(ch_data)) / np.std(ch_data)
                    else:
                        normalized_data[ch, :] = ch_data
                
                # Check for NaN or Inf in normalized data
                if np.isnan(normalized_data).any():
                    logger.warning(f"NaN detected in normalized SuS EEG data from {sus_file}")
                    continue
                if np.isinf(normalized_data).any():
                    logger.warning(f"Inf detected in normalized SuS EEG data from {sus_file}")
                    continue
                
                sus_data.append(normalized_data)
            
            # Concatenate all SuS data
            if sus_data:
                return np.concatenate(sus_data, axis=1)  # Concatenate along time dimension
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error loading SuS EEG data: {e}")
            return None
    
    @staticmethod
    def _load_ccd_trials(ccd_file: str) -> List[Dict[str, Any]]:
        """Load CCD trials from events file"""
        try:
            # Load events file for CCD task
            events_file = Path(ccd_file).parent / Path(ccd_file).name.replace('_eeg.set', '_events.tsv')
            
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
                        'contrast_change_time': None,
                        'button_press_time': None,
                        'response_time': None,
                        'hit_miss': None
                    }
                    trial_id += 1
                
                # Detect target (contrast change)
                elif 'target' in event_value and current_trial is not None:
                    current_trial['contrast_change_time'] = onset_time
                
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
            
            return trials
            
        except Exception as e:
            logger.error(f"Error loading CCD trials from {ccd_file}: {e}")
            return []
    
    @staticmethod
    def _extract_ccd_epoch(eeg_data: np.ndarray, trial: Dict, epoch_duration: float, sampling_rate: int) -> Optional[np.ndarray]:
        """Extract CCD EEG epoch for a specific trial"""
        try:
            # Calculate start and end samples based on contrast change time
            start_time = trial['contrast_change_time']
            start_sample = int(start_time * sampling_rate)
            end_sample = start_sample + int(epoch_duration * sampling_rate)
            
            # Check bounds
            if start_sample < 0 or end_sample > eeg_data.shape[1]:
                logger.warning(f"Trial {trial['trial_id']} outside EEG data bounds")
                return None
            
            # Extract epoch (data is already normalized from _load_ccd_eeg_data)
            epoch = eeg_data[:, start_sample:end_sample]
            
            # Validate epoch shape
            if epoch.shape[0] != 128:
                logger.warning(f"Expected 128 channels, got {epoch.shape[0]} for trial {trial['trial_id']}")
                return None
            
            # Return epoch directly (no need for additional normalization)
            return epoch
            
        except Exception as e:
            logger.warning(f"Error extracting CCD epoch for trial {trial['trial_id']}: {e}")
            return None
    
    def _find_ccd_files(self, subject_dir: Path) -> List[Path]:
        """Find CCD (Contrast Change Detection) EEG files"""
        ccd_files = []
        for file_path in subject_dir.glob("*task-contrastChangeDetection*_eeg.set"):
            ccd_files.append(file_path)
        return sorted(ccd_files)
    
    def _find_sus_files(self, subject_dir: Path) -> List[Path]:
        """Find SuS (Surround Suppression) EEG files"""
        sus_files = []
        for file_path in subject_dir.glob("*task-surroundSupp*_eeg.set"):
            sus_files.append(file_path)
        return sorted(sus_files)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """
        Get a sample.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (input_features_dict, targets_dict)
            Input features include CCD EEG data and optional SuS EEG data and demographics
        """
        if idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of bounds for dataset with {len(self.samples)} samples")
        
        try:
            sample = self.samples[idx]
            
            # Get CCD EEG data (primary input X1)
            ccd_eeg = sample['ccd_eeg_data']
            
            # Validate EEG data
            if ccd_eeg is None:
                raise ValueError(f"CCD EEG data is None for sample {idx}")
            
            if not isinstance(ccd_eeg, np.ndarray):
                raise ValueError(f"CCD EEG data must be numpy array, got {type(ccd_eeg)}")
            
            if ccd_eeg.shape[0] != 128:
                raise ValueError(f"Expected 128 channels, got {ccd_eeg.shape[0]}")
            
            # Validate EEG data for NaN/Inf before creating tensor
            if np.isnan(ccd_eeg).any():
                logger.error(f"NaN detected in CCD EEG data for sample {idx}")
                raise ValueError(f"NaN in CCD EEG data for sample {idx}")
            if np.isinf(ccd_eeg).any():
                logger.error(f"Inf detected in CCD EEG data for sample {idx}")
                raise ValueError(f"Inf in CCD EEG data for sample {idx}")
            
            # Create input features dictionary
            input_features = {
                'ccd_eeg': torch.FloatTensor(ccd_eeg)  # Primary input X1
            }
            
            # Add SuS EEG data if available (X2)
            if self.use_sus_eeg and sample.get('sus_eeg_data') is not None:
                sus_eeg = sample['sus_eeg_data']
                if isinstance(sus_eeg, np.ndarray) and sus_eeg.shape[0] == 128:
                    input_features['sus_eeg'] = torch.FloatTensor(sus_eeg)
            
            # Add demographics if available (P)
            if self.use_demographics and sample.get('demographics') is not None:
                demo = sample['demographics']
                demographics_tensor = torch.FloatTensor([
                    demo.get('age', 0.0),
                    demo.get('sex', 0.0),
                    demo.get('handedness', 0.0),
                    demo.get('p_factor', 0.0),
                    demo.get('attention', 0.0),
                    demo.get('internalizing', 0.0),
                    demo.get('externalizing', 0.0)
                ])
                input_features['demographics'] = demographics_tensor
            
            # Apply transforms to all EEG data if provided
            if self.transforms:
                if 'ccd_eeg' in input_features:
                    input_features['ccd_eeg'] = self.transforms(input_features['ccd_eeg'])
                if 'sus_eeg' in input_features:
                    input_features['sus_eeg'] = self.transforms(input_features['sus_eeg'])
            
            # Create target tensors
            targets = {}
            
            # Response time (regression)
            rt = sample['targets']['response_time']
            if rt is None or np.isnan(rt) or rt < 0:
                rt = 0  # Default response time
            
            # Use raw response time values (no log transform)
            # This matches the model's output scaling
            targets['response_time'] = torch.FloatTensor([rt])
            
            # Hit/miss (classification)
            hm = sample['targets']['hit_miss']
            if hm not in [0, 1]:
                hm = 0  # Default to miss
            targets['hit_miss'] = torch.LongTensor([hm])
            
            return input_features, targets
            
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
    
    def _get_cache_file_path(self) -> Path:
        """Generate cache file path based on configuration and split"""
        import hashlib
        
        # Create a hash of the configuration to ensure cache invalidation when config changes
        config_str = str(sorted(self.config.items()))
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Include feature flags in cache filename to avoid conflicts
        features_str = f"d{int(self.use_demographics)}_s{int(self.use_sus_eeg)}"
        
        # Include split in filename
        cache_config = self.config.get('caching', {})
        cache_dir = Path(cache_config.get('cache_dir', './cache'))
        cache_dir.mkdir(exist_ok=True)
        
        cache_file = cache_dir / f"challenge1_samples_{self.split}_{features_str}_{config_hash}.pkl"
        return cache_file
    
    def _load_from_cache(self, cache_file: Path) -> bool:
        """Load samples from cache file"""
        try:
            if cache_file.exists():
                import pickle
                with open(cache_file, 'rb') as f:
                    self.samples = pickle.load(f)
                logger.info(f"Cache loaded from {cache_file}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_file}: {e}")
        return False
    
    def _save_to_cache(self, cache_file: Path) -> None:
        """Save samples to cache file"""
        try:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(self.samples, f)
            logger.info(f"Cache saved to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_file}: {e}")

# Global cache for BIDS layouts to avoid duplicate initialization
_BIDS_LAYOUTS_CACHE = {}

def create_challenge1_dataloaders(
    data_dir: str,
    config: Dict[str, Any],
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    clear_cache: bool = False,
    use_demographics: bool = True,
    use_sus_eeg: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for Challenge 1
    
    EEG Foundation Challenge Split Strategy:
    - Training Set: Releases 1-11, excluding Release 5
    - Validation Set: Release 5 only
    - Test Set: Release 12 (held-out, not publicly released)
    
    Args:
        data_dir: Path to HBN BIDS EEG dataset
        config: Configuration dictionary
        batch_size: Batch size for training
        num_workers: Number of data loader workers
        clear_cache: Whether to clear existing cache and reprocess data
        use_demographics: Whether to include demographics and psychopathology as additional features
        use_sus_eeg: Whether to include SuS EEG data as additional features
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    global _BIDS_LAYOUTS_CACHE
    
    # Get num_workers from config if not provided
    if num_workers is None:
        num_workers = config.get('hardware', {}).get('num_workers', 4)
        logger.info(f"Using num_workers from config: {num_workers}")
    
    # Validate that required releases are available
    data_path = Path(data_dir)
    available_releases = [d.name for d in data_path.iterdir() 
                         if d.is_dir() and d.name.startswith('cmi_bids')]
    logger.info(f"Available releases: {available_releases}")
    
    # Clear cache if requested
    if clear_cache:
        cache_config = config.get('caching', {})
        cache_dir = Path(cache_config.get('cache_dir', './cache'))
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            logger.info(f"Cache cleared from {cache_dir}")
        # Also clear BIDS layouts cache
        _BIDS_LAYOUTS_CACHE.clear()
    
    # EEG Foundation Challenge Release-based Split Strategy
    logger.info("Creating EEG Foundation Challenge release-based splits...")
    
    # Define release assignments according to challenge specification
    # Note: Only using available releases (R1-R9), excluding R10, R11, and NC
    # train_releases = ['cmi_bids_R1', 'cmi_bids_R2', 'cmi_bids_R3', 'cmi_bids_R4',
    #                  'cmi_bids_R6', 'cmi_bids_R7', 'cmi_bids_R8', 'cmi_bids_R9']
    # val_releases = ['cmi_bids_R5']
    train_releases = ['cmi_bids_R1_mini']
    val_releases = ['cmi_bids_R1_mini']
    
    
    logger.info(f"Training releases: {train_releases}")
    logger.info(f"Validation releases: {val_releases}")
    
    # Use cached BIDS layouts or initialize if not cached
    if not _BIDS_LAYOUTS_CACHE:
        logger.info("Initializing shared BIDS layouts...")
        total_subjects = 0
        
        for release_dir in [str(data_path / release) for release in available_releases]:
            try:
                layout = BIDSLayout(release_dir)
                subjects = layout.get_subjects()
                _BIDS_LAYOUTS_CACHE[release_dir] = layout
                total_subjects += len(subjects)
                logger.info(f"✅ {Path(release_dir).name}: {len(subjects)} subjects")
            except Exception as e:
                logger.warning(f"Failed to initialize layout for {release_dir}: {e}")
        
        logger.info(f"Shared BIDS layouts initialized: {len(_BIDS_LAYOUTS_CACHE)} releases, {total_subjects} total subjects")
    else:
        logger.info(f"Using cached BIDS layouts: {len(_BIDS_LAYOUTS_CACHE)} releases")
    
    # Create datasets for each split with shared layouts
    train_dataset = Challenge1Dataset(
        data_dir=data_dir,
        config=config,
        split='train',
        releases=train_releases,
        use_demographics=use_demographics,
        use_sus_eeg=use_sus_eeg,
        shared_layouts=_BIDS_LAYOUTS_CACHE  # Pass cached layouts
    )
    
    val_dataset = Challenge1Dataset(
        data_dir=data_dir,
        config=config,
        split='val',
        releases=val_releases,
        use_demographics=use_demographics,
        use_sus_eeg=use_sus_eeg,
        shared_layouts=_BIDS_LAYOUTS_CACHE  # Pass cached layouts
    )
    
    # Create data loaders
    logger.info(f"Creating DataLoaders with num_workers={num_workers}")
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
    
    logger.info(f"Created Challenge 1 data loaders (EEG Foundation Challenge splits):")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    
    return train_loader, val_loader

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
        data_dir='src/data/raw/HBN_BIDS_EEG',
        config=config,
        split='train'
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        sample_eeg, sample_targets = dataset[0]
        print(f"SuS EEG shape: {sample_eeg['ccd_eeg'].shape}")
        print(f"Targets: {list(sample_targets.keys())}")
        print("Challenge 1 dataset test completed successfully!")
    else:
        print("No samples found - check data directory path")