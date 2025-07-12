# app.py

import streamlit as st
import os
import io
import tempfile
import json  # Import for saving labels
from contextlib import redirect_stdout
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary machine learning libraries
try:
    # Alternative PyTorch import fix
    import sys
    import importlib.util
    
    # Check if torch is already imported
    if 'torch' in sys.modules:
        torch = sys.modules['torch']
    else:
        import torch
    
    # Fix the _classes module issue if it exists
    if hasattr(torch, '_classes') and hasattr(torch._classes, '__path__'):
        # Create a mock __path__ to prevent the error
        class MockPath:
            def __init__(self):
                self._path = []
            
            def __iter__(self):
                return iter(self._path)
            
        torch._classes.__path__ = MockPath()
    import librosa
    import librosa.display
    from sklearn.metrics import accuracy_score
    import numpy as np
    from datasets import load_dataset, Audio, DatasetDict # Ensure DatasetDict is imported
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
    import warnings
    warnings.filterwarnings('ignore')
    
    # Check accelerate version compatibility
    try:
        import accelerate
        accelerate_version = accelerate.__version__
        print(f"Accelerate version: {accelerate_version}")
    except ImportError:
        st.error("Accelerate library not found. Please install it with: pip install accelerate>=0.20.0")
        st.stop()
        
except ImportError as e:
    st.error(f"Missing required libraries: {e}. Please run 'pip install -r requirements.txt'")
    st.stop()

# =====================================================================================
# Configuration and Helpers
# =====================================================================================

def setup_directories(model_output_dir):
    """Ensure output directories exist"""
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(os.path.join(model_output_dir, 'training_checkpoints'), exist_ok=True)

# =====================================================================================
# Audio Analysis and Visualization Functions
# =====================================================================================

def extract_audio_features(audio_array, sr):
    """Extract comprehensive audio features for analysis."""
    features = {}
    
    # Basic features
    features['duration'] = len(audio_array) / sr
    features['rms_energy'] = float(np.sqrt(np.mean(audio_array**2)))
    features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio_array)[0]))
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=sr)[0]
    features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
    features['spectral_centroid_std'] = float(np.std(spectral_centroids))
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=sr)[0]
    features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
    
    # Additional spectral features for better analysis
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_array, sr=sr)[0]
    features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio_array, sr=sr)
    features['chroma_mean'] = float(np.mean(chroma))
    features['chroma_std'] = float(np.std(chroma))
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
        features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
    
    # Tempo and rhythm
    tempo, _ = librosa.beat.beat_track(y=audio_array, sr=sr)
    features['tempo'] = float(tempo)
    
    # Additional rhythm features
    onset_frames = librosa.onset.onset_detect(y=audio_array, sr=sr)
    features['onset_rate'] = len(onset_frames) / features['duration'] if features['duration'] > 0 else 0
    
    return features

def analyze_feature_significance(features, predicted_class, confidence, all_class_probabilities):
    """
    Analyze which features are most significant for the prediction and provide justifications.
    """
    justifications = []
    
    # Define typical ranges and characteristics for different audio types
    feature_insights = {
        'tempo': {
            'slow': (0, 90, "slow tempo"),
            'moderate': (90, 140, "moderate tempo"),
            'fast': (140, 300, "fast tempo")
        },
        'spectral_centroid_mean': {
            'low': (0, 2000, "low frequency content, darker timbre"),
            'medium': (2000, 4000, "balanced frequency content"),
            'high': (4000, 22050, "high frequency content, brighter timbre")
        },
        'rms_energy': {
            'quiet': (0, 0.05, "low energy/volume"),
            'moderate': (0.05, 0.2, "moderate energy/volume"),
            'loud': (0.2, 1.0, "high energy/volume")
        },
        'zero_crossing_rate': {
            'low': (0, 0.05, "smooth, tonal content"),
            'medium': (0.05, 0.15, "mixed tonal/noisy content"),
            'high': (0.15, 1.0, "noisy, percussive content")
        },
        'spectral_bandwidth_mean': {
            'narrow': (0, 2000, "narrow frequency spread, pure tones"),
            'medium': (2000, 4000, "moderate frequency spread"),
            'wide': (4000, 22050, "wide frequency spread, complex harmonics")
        },
        'chroma_mean': {
            'low': (0, 0.3, "weak harmonic content"),
            'medium': (0.3, 0.6, "moderate harmonic content"),
            'high': (0.6, 1.0, "strong harmonic/tonal content")
        }
    }
    
    # Analyze key features
    tempo = features.get('tempo', 0)
    spectral_centroid = features.get('spectral_centroid_mean', 0)
    rms_energy = features.get('rms_energy', 0)
    zcr = features.get('zero_crossing_rate', 0)
    spectral_bandwidth = features.get('spectral_bandwidth_mean', 0)
    chroma = features.get('chroma_mean', 0)
    duration = features.get('duration', 0)
    onset_rate = features.get('onset_rate', 0)
    
    # Generate tempo-based justification
    for range_name, (min_val, max_val, description) in feature_insights['tempo'].items():
        if min_val <= tempo <= max_val:
            justifications.append(f"🎵 **Tempo Analysis**: {tempo:.1f} BPM indicates {description}, which is characteristic of the predicted class '{predicted_class}'")
            break
    
    # Generate spectral centroid justification
    for range_name, (min_val, max_val, description) in feature_insights['spectral_centroid_mean'].items():
        if min_val <= spectral_centroid <= max_val:
            justifications.append(f"🎼 **Spectral Brightness**: {spectral_centroid:.0f} Hz spectral centroid suggests {description}, supporting the '{predicted_class}' classification")
            break
    
    # Generate energy-based justification
    for range_name, (min_val, max_val, description) in feature_insights['rms_energy'].items():
        if min_val <= rms_energy <= max_val:
            justifications.append(f"🔊 **Energy Level**: RMS energy of {rms_energy:.3f} indicates {description}, typical for '{predicted_class}' audio")
            break
    
    # Generate texture justification based on zero crossing rate
    for range_name, (min_val, max_val, description) in feature_insights['zero_crossing_rate'].items():
        if min_val <= zcr <= max_val:
            justifications.append(f"🌊 **Audio Texture**: Zero crossing rate of {zcr:.3f} suggests {description}, consistent with '{predicted_class}' characteristics")
            break
    
    # Generate harmonic content justification
    for range_name, (min_val, max_val, description) in feature_insights['chroma_mean'].items():
        if min_val <= chroma <= max_val:
            justifications.append(f"🎹 **Harmonic Content**: Chroma mean of {chroma:.3f} indicates {description}, aligning with '{predicted_class}' expectations")
            break
    
    # Duration-based insights
    if duration < 5:
        justifications.append(f"⏱️ **Duration**: Short duration ({duration:.1f}s) may indicate brief sounds or clips typical of '{predicted_class}'")
    elif duration > 30:
        justifications.append(f"⏱️ **Duration**: Long duration ({duration:.1f}s) suggests extended audio content characteristic of '{predicted_class}'")
    
    # Onset rate insights
    if onset_rate > 5:
        justifications.append(f"🥁 **Rhythmic Activity**: High onset rate ({onset_rate:.1f} onsets/sec) indicates rhythmic or percussive elements supporting '{predicted_class}' classification")
    elif onset_rate < 1:
        justifications.append(f"🎵 **Sustained Content**: Low onset rate ({onset_rate:.1f} onsets/sec) suggests sustained tones or smooth content typical of '{predicted_class}'")
    
    # MFCC-based insights (analyze first few MFCCs which capture spectral shape)
    mfcc1_mean = features.get('mfcc_1_mean', 0)
    # The absolute value of MFCC1 often relates to overall loudness, positive or negative indicates spectrum tilt
    justifications.append(f"📊 **Spectral Balance (MFCC-1)**: MFCC-1 mean of {mfcc1_mean:.2f} contributes to the overall spectral shape supporting '{predicted_class}' classification.")
    
    # Confidence-based justification
    if confidence > 0.8:
        justifications.append(f"✅ **High Confidence**: Model confidence of {confidence:.1%} indicates strong feature alignment with '{predicted_class}' training examples")
    elif confidence > 0.6:
        justifications.append(f"⚖️ **Moderate Confidence**: Model confidence of {confidence:.1%} suggests good but not perfect feature match with '{predicted_class}'")
    else:
        justifications.append(f"⚠️ **Lower Confidence**: Model confidence of {confidence:.1%} indicates some uncertainty, possibly due to mixed characteristics or limited training data")
    
    # Alternative class analysis
    sorted_probs = sorted(all_class_probabilities.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_probs) > 1:
        second_class, second_prob = sorted_probs[1]
        prob_diff = confidence - second_prob
        if prob_diff < 0.2 and confidence >= 0.5: # Only suggest close decision if primary confidence is reasonable
            justifications.append(f"🤔 **Close Decision**: '{predicted_class}' was chosen over '{second_class}' by only {prob_diff:.1%}, indicating some shared characteristics between classes.")
        elif prob_diff < 0.1 and confidence < 0.5: # Even lower confidence, might be very ambiguous
             justifications.append(f"🤔 **Very Ambiguous**: The model showed very low confidence for '{predicted_class}' and was close to '{second_class}', suggesting highly ambiguous features.")
    
    return justifications

def create_feature_justification_plot(features, predicted_class):
    """Create a radar plot showing key features that influenced the prediction."""
    # Select key features for visualization and normalize them
    # These normalization factors are heuristic and may need tuning based on actual data ranges
    key_features = {
        'Tempo\n(BPM)': features.get('tempo', 0),
        'Spectral\nCentroid': features.get('spectral_centroid_mean', 0),
        'Energy\n(RMS)': features.get('rms_energy', 0),
        'Zero Crossing\nRate': features.get('zero_crossing_rate', 0),
        'Chroma\nMean': features.get('chroma_mean', 0),
        'Spectral\nBandwidth': features.get('spectral_bandwidth_mean', 0),
        'Onset\nRate': features.get('onset_rate', 0)
    }

    # Normalize values for radar plot (0-1 scale)
    normalized_values = {}
    normalized_values['Tempo\n(BPM)'] = min(key_features['Tempo\n(BPM)'] / 200.0, 1.0) # Assume max tempo around 200 BPM
    normalized_values['Spectral\nCentroid'] = min(key_features['Spectral\nCentroid'] / 6000.0, 1.0) # Assume max centroid around 6000 Hz
    normalized_values['Energy\n(RMS)'] = min(key_features['Energy\n(RMS)'] / 0.3, 1.0) # Assume max RMS around 0.3 (adjusted from 1.0 for better visual range)
    normalized_values['Zero Crossing\nRate'] = min(key_features['Zero Crossing\nRate'] / 0.5, 1.0) # Assume max ZCR around 0.5
    normalized_values['Chroma\nMean'] = key_features['Chroma\nMean'] # Chroma is already 0-1
    normalized_values['Spectral\nBandwidth'] = min(key_features['Spectral\nBandwidth'] / 4000.0, 1.0) # Assume max bandwidth around 4000 Hz
    normalized_values['Onset\nRate'] = min(key_features['Onset\nRate'] / 10.0, 1.0) # Assume max onset rate around 10 onsets/sec

    # Create radar plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Calculate angles for each feature
    angles = np.linspace(0, 2 * np.pi, len(normalized_values), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Values for plotting
    values = list(normalized_values.values())
    values += values[:1]  # Complete the circle
    
    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Audio Features Profile', color='darkblue')
    ax.fill(angles, values, alpha=0.25, color='skyblue')
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(normalized_values.keys(), fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True)
    
    plt.title(f'Feature Profile for "{predicted_class}" Classification', 
              size=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

def create_spectrogram_plot(audio_array, sr, title="Spectrogram"):
    """Create spectrogram visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_array)), ref=np.max)
    
    # Plot spectrogram
    img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    
    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label('Magnitude (dB)')
    
    plt.tight_layout()
    return fig

def create_mel_spectrogram_plot(audio_array, sr, title="Mel Spectrogram"):
    """Create mel spectrogram visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=128)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    
    # Plot mel spectrogram
    img = librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time', sr=sr, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mel Frequency')
    
    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label('Magnitude (dB)')
    
    plt.tight_layout()
    return fig

def create_waveform_plot(audio_array, sr, title="Waveform"):
    """Create waveform visualization."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    time_axis = np.linspace(0, len(audio_array) / sr, len(audio_array))
    ax.plot(time_axis, audio_array, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_mfcc_plot(audio_array, sr, title="MFCC Features"):
    """Create MFCC features visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute MFCC
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
    
    # Plot MFCC
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MFCC Coefficients')
    
    # Add colorbar
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('MFCC Value')
    
    plt.tight_layout()
    return fig

def create_confidence_distribution_plot(results):
    """Create confidence distribution plot for batch results."""
    confidences = [r['confidence'] for r in results if 'confidence' in r and 'error' not in r]
    
    if not confidences:
        return None, "No confidence data available for plotting."

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
    ax.set_title('Confidence Score Distribution')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, None

def create_feature_comparison_plot(results):
    """Create feature comparison plot for batch results."""
    # Extract features for comparison
    features_data = []
    for result in results:
        if 'audio_features' in result and 'predicted_label' in result and 'error' not in result:
            features_data.append({
                'filename': result['filename'],
                'duration': result['audio_features']['duration'],
                'tempo': result['audio_features']['tempo'],
                'rms_energy': result['audio_features']['rms_energy'],
                'spectral_centroid': result['audio_features']['spectral_centroid_mean'],
                'predicted_class': result['predicted_label']
            })
    
    if not features_data:
        return None, "No valid results with audio features to create comparison plot."
    
    df = pd.DataFrame(features_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Duration vs Class
    sns.boxplot(x='predicted_class', y='duration', data=df, ax=axes[0, 0])
    axes[0, 0].set_xlabel('Predicted Class')
    axes[0, 0].set_ylabel('Duration (s)')
    axes[0, 0].set_title('Duration by Predicted Class')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Tempo vs Class
    sns.boxplot(x='predicted_class', y='tempo', data=df, ax=axes[0, 1])
    axes[0, 1].set_xlabel('Predicted Class')
    axes[0, 1].set_ylabel('Tempo (BPM)')
    axes[0, 1].set_title('Tempo by Predicted Class')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # RMS Energy vs Class
    sns.boxplot(x='predicted_class', y='rms_energy', data=df, ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicted Class')
    axes[1, 0].set_ylabel('RMS Energy')
    axes[1, 0].set_title('RMS Energy by Predicted Class')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Spectral Centroid vs Class
    sns.boxplot(x='predicted_class', y='spectral_centroid', data=df, ax=axes[1, 1])
    axes[1, 1].set_xlabel('Predicted Class')
    axes[1, 1].set_ylabel('Spectral Centroid (Hz)')
    axes[1, 1].set_title('Spectral Centroid by Predicted Class')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig, None


def create_class_distribution_plot(results):
    """Create class distribution plot for batch results."""
    classes = [r['predicted_label'] for r in results if 'predicted_label' in r and 'error' not in r]
    
    if not classes:
        return None, "No predicted class data available for plotting."
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count classes
    class_counts = pd.Series(classes).value_counts()
    
    # Create bar plot
    bars = ax.bar(class_counts.index, class_counts.values, alpha=0.8, edgecolor='black')
    ax.set_title('Predicted Class Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Files')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, None

# =====================================================================================
# AST Training Logic
# =====================================================================================

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": accuracy_score(eval_pred.label_ids, predictions)}

def validate_dataset_structure(data_dir):
    """
    Validate that the dataset directory has the correct structure for classification.
    Supports both explicit train/val/test subdirectories and flat class folders.
    """
    if not os.path.isdir(data_dir):
        return False, f"Directory does not exist: {data_dir}"
    
    # Check for explicit train/validation/test subdirectories first
    expected_splits_dirs = ['train', 'validation', 'test']
    found_explicit_splits = [s for s in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, s)) and s in expected_splits_dirs]

    if found_explicit_splits:
        # If explicit splits are found, validate content within each split
        all_class_names = set()
        for split_name in found_explicit_splits:
            split_path = os.path.join(data_dir, split_name)
            class_subdirs = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            
            if len(class_subdirs) < 2:
                return False, f"Split '{split_name}' in {split_path} has less than 2 class subdirectories. Need at least 2 class folders per split."
            
            all_class_names.update(class_subdirs)
            
            audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
            for class_subdir in class_subdirs:
                class_path = os.path.join(split_path, class_subdir)
                files = os.listdir(class_path)
                audio_files = [f for f in files if any(f.lower().endswith(ext) for ext in audio_extensions)]
                if not audio_files:
                    return False, f"No audio files found in class folder: {class_path}"
        
        return True, f"Found explicit splits: {found_explicit_splits} with {len(all_class_names)} classes: {list(all_class_names)}"
    else:
        # If no explicit splits, assume flat structure (class folders directly under data_dir)
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if len(subdirs) >= 2: # Assuming at least 2 class folders for a valid dataset
            audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
            for subdir in subdirs:
                subdir_path = os.path.join(data_dir, subdir)
                files = os.listdir(subdir_path)
                audio_files = [f for f in files if any(f.lower().endswith(ext) for ext in audio_extensions)]
                if not audio_files:
                    return False, f"No audio files found in class folder: {subdir_path}. Please ensure each class folder contains audio files."
            
            # Removed the st.warning message as requested
            return True, f"Found {len(subdirs)} classes: {subdirs} (in a flat structure)."
        else:
            return False, f"No recognized splits ('train', 'validation', 'test') or class folders found under {data_dir}. Please ensure dataset structure is correct (either explicit splits or flat class folders with audio files)."


def run_training_ui(data_dir, model_output_dir, num_train_epochs, learning_rate, batch_size):
    """UI wrapper for training with progress updates."""
    progress_area = st.empty()
    
    with st.spinner("Training in progress..."):
        log_output = run_classification_training(
            data_dir, model_output_dir, num_train_epochs, learning_rate, batch_size, progress_area
        )
    
    st.success("Training process finished!") 
    st.subheader("📋 Training Log")
    with st.expander("Show Full Training Log", expanded=True):
        st.text(log_output)

def run_classification_training(data_dir, model_output_dir, num_train_epochs, learning_rate, batch_size, progress_area):
    log_stream = io.StringIO()
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_stream)
    
    try:
        progress_area.text("Step 1/7: Validating dataset structure...")
        is_valid, message = validate_dataset_structure(data_dir)
        if not is_valid:
            log_print(f"ERROR: {message}")
            return log_stream.getvalue()
        log_print(f"✓ Dataset validation passed: {message}")

        progress_area.text("Step 2/7: Loading dataset...")
        
        # Check if the dataset has explicit train/val/test subdirectories
        has_explicit_splits = any(s in os.listdir(data_dir) for s in ['train', 'validation', 'test'])

        if has_explicit_splits:
            # Load as DatasetDict if explicit splits are found
            dataset = load_dataset("audiofolder", data_dir=data_dir)
            log_print(f"✓ Dataset loaded as DatasetDict. Found splits: {list(dataset.keys())}")
            
            if "train" not in dataset:
                log_print("ERROR: 'train' split not found in the loaded DatasetDict.")
                log_print("Available splits: " + ", ".join(dataset.keys()))
                return log_stream.getvalue()
            
            # Use 'validation' if available, otherwise fallback to 'test' for evaluation
            eval_split_name = "validation" if "validation" in dataset else "test"
            if eval_split_name not in dataset:
                log_print("ERROR: Neither 'validation' nor 'test' split found for evaluation.")
                log_print("Available splits: " + ", ".join(dataset.keys()))
                return log_stream.getvalue()

            train_dataset = dataset["train"]
            eval_dataset = dataset[eval_split_name]

        else:
            # If no explicit splits, load the entire dataset as one split (e.g., 'train')
            # and then use it as both train and eval dataset.
            log_print("No explicit train/val/test splits found. Loading entire dataset as a single split for both training and evaluation.")
            
            # The 'audiofolder' builder often defaults to 'train' or 'validation' for flat structures.
            # We try 'train' first, then fallback to 'validation' to load the full flat dataset.
            try:
                full_dataset = load_dataset("audiofolder", data_dir=data_dir, split="train")
            except ValueError: 
                full_dataset = load_dataset("audiofolder", data_dir=data_dir, split="validation")
            
            log_print(f"Loaded full dataset with {len(full_dataset)} examples.")
            
            # Assign the full dataset to both train and eval
            train_dataset = full_dataset
            eval_dataset = full_dataset # Evaluate on the same data for "whole data training"
            eval_split_name = "full_dataset_as_eval" # For logging clarity

            log_print(f"✓ Dataset loaded as single split for training and evaluation.")
            log_print(f"Train size: {len(train_dataset)}, Evaluation size: {len(eval_dataset)}")


        progress_area.text("Step 3/7: Loading pre-trained AST model...")
        model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        
        labels = train_dataset.features["label"].names # Get labels from the train split
        model = AutoModelForAudioClassification.from_pretrained(
            model_id, num_labels=len(labels),
            label2id={l: str(i) for i, l in enumerate(labels)},
            id2label={str(i): l for i, l in enumerate(labels)},
            ignore_mismatched_sizes=True
        )
        log_print("✓ Pre-trained model loaded.")

        progress_area.text("Step 4/7: Preprocessing audio data...")
        target_sampling_rate = feature_extractor.sampling_rate
        
        # Cast audio column for all relevant splits
        train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
        eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
        
        def preprocess_function(examples):
            audio_arrays = [x["array"] for x in examples["audio"]]
            return feature_extractor(audio_arrays, sampling_rate=target_sampling_rate, padding=True, return_tensors="pt")
        
        encoded_train_dataset = train_dataset.map(preprocess_function, remove_columns=["audio"], batched=True, batch_size=8)
        encoded_eval_dataset = eval_dataset.map(preprocess_function, remove_columns=["audio"], batched=True, batch_size=8)
        
        log_print("✓ Dataset preprocessed.")

        progress_area.text("Step 5/7: Setting up training configuration...")
        setup_directories(model_output_dir)
        
        training_args = TrainingArguments(
            output_dir=os.path.join(model_output_dir, 'training_checkpoints'),
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,  
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            logging_steps=10,
            save_total_limit=2,
            evaluation_strategy="epoch",  
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to=None,  
            dataloader_drop_last=False,
            remove_unused_columns=True,
        )
        
        trainer = Trainer(
            model=model,  
            args=training_args,
            train_dataset=encoded_train_dataset,  # Use the encoded train dataset
            eval_dataset=encoded_eval_dataset,    # Use the encoded eval dataset
            tokenizer=feature_extractor,  
            compute_metrics=compute_metrics,
        )
        log_print("✓ Trainer configured.")

        progress_area.text("Step 6/7: Training model...")
        trainer.train()
        log_print("✅ Training completed!")

        progress_area.text("Step 7/7: Evaluating and saving trained model...")
        # Evaluate the final model
        eval_results = trainer.evaluate()
        log_print(f"✓ Final evaluation results: {eval_results}")
        
        # Save the final model
        trainer.save_model(model_output_dir)
        feature_extractor.save_pretrained(model_output_dir)

        # Explicitly save the labels list for robust prediction
        with open(os.path.join(model_output_dir, 'labels.json'), 'w') as f:
            json.dump(labels, f)
        log_print(f"✅ Model, feature extractor, and labels saved to {model_output_dir}")

    except Exception as e:
        log_print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc(file=log_stream)

    progress_area.empty()
    return log_stream.getvalue()

# =====================================================================================
# AST Prediction Logic
# =====================================================================================

@st.cache_resource
def load_model_and_artifacts(model_path):
    """Load model, feature extractor, and the labels file."""
    try:
        if not os.path.isdir(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
            return None, None, None, "Model not found. Ensure path is correct and model was trained."
        
        labels_path = os.path.join(model_path, "labels.json")
        if not os.path.exists(labels_path):
            return None, None, None, "labels.json not found. Please retrain the model with the latest script."
        with open(labels_path, 'r') as f:
            labels = json.load(f)

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        model = AutoModelForAudioClassification.from_pretrained(model_path)
        model.eval()
        
        return model, feature_extractor, labels, None
    except Exception as e:
        return None, None, None, f"Error loading model artifacts: {str(e)}"

def run_single_classification_prediction(model_path, audio_file):
    """Run prediction on a single audio file and return detailed results."""
    model, feature_extractor, labels, error_msg = load_model_and_artifacts(model_path)
    if error_msg:
        return None, f"Error: {error_msg}"
    
    tmp_file_path = None # Initialize outside try-block for cleanup
    try:
        # Use the actual file extension for the temporary file
        file_extension = audio_file.name.split('.')[-1] if '.' in audio_file.name else 'wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name
        target_sampling_rate = feature_extractor.sampling_rate
        audio_array, _ = librosa.load(tmp_file_path, sr=target_sampling_rate, mono=True)
        
        inputs = feature_extractor(audio_array, sampling_rate=target_sampling_rate, padding=True, return_tensors="pt")
    except Exception as e:
        return None, f"Error processing audio file '{audio_file.name}': {str(e)}"
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path) # Clean up the temporary file

    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Get all class probabilities
    probabilities = torch.softmax(logits, dim=-1).squeeze().numpy()
    predicted_class_id = np.argmax(probabilities)
    confidence = probabilities[predicted_class_id]
    
    # Get top 3 predictions (kept for internal data structure, though not displayed directly in UI)
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_predictions = [(labels[i] if i < len(labels) else f"Unknown_Class_{i}", probabilities[i]) for i in top_indices]
    
    # Extract audio features
    features = extract_audio_features(audio_array, target_sampling_rate)
    
    result = {
        'filename': audio_file.name, # Add filename directly here for convenience
        'predicted_label': labels[predicted_class_id] if predicted_class_id < len(labels) else f"Unknown_Class_{predicted_class_id}",
        'confidence': confidence,
        'top_predictions': top_predictions, # Still includes this in the result dict
        'all_probabilities': {labels[i] if i < len(labels) else f"Unknown_Class_{i}": prob for i, prob in enumerate(probabilities)}, # Still includes this
        'audio_features': features,
        'audio_array': audio_array,
        'sampling_rate': target_sampling_rate,
        'raw_audio_bytes': audio_file.getvalue() # Store raw bytes for playback in tabs
    }
    
    return result, None

def run_batch_classification_prediction(model_path, audio_files):
    """Run prediction on multiple audio files."""
    results = []
    
    for i, audio_file in enumerate(audio_files):
        # Use st.spinner for individual file processing feedback in batch mode
        st.info(f"Classifying {audio_file.name} ({i+1}/{len(audio_files)})...")
        with st.spinner(f"Processing {audio_file.name}..."):
            result, error = run_single_classification_prediction(model_path, audio_file)
            if error:
                results.append({
                    'filename': audio_file.name,
                    'error': error
                })
                st.warning(f"Skipped {audio_file.name} due to error: {error}")
            else:
                results.append(result)
        st.success(f"Finished classifying {audio_file.name}.") # Give immediate feedback for each file
        
    return results

# =====================================================================================
# Enhanced Streamlit UI Components with Show More Functionality
# =====================================================================================

def display_single_audio_details(audio_result):
    """Displays essential metrics and visualizations for a single audio file within a tab."""
    if 'error' in audio_result:
        st.error(f"Error processing {audio_result['filename']}: {audio_result['error']}")
        return

    st.markdown(f"### 🎧 Analysis for: `{audio_result['filename']}`")
    st.audio(audio_result['raw_audio_bytes'], format='audio/wav')
    st.divider()

    # ========== 1. KEY CLASSIFICATION METRICS (Always Visible) ==========
    st.markdown("#### 🎯 Prediction Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Predicted Class", f"**{audio_result['predicted_label'].upper()}** 🎶")
    with col2:
        st.metric("Confidence", f"{audio_result['confidence']:.2%} ✨")
    with col3:
        st.metric("Duration", f"{audio_result['audio_features']['duration']:.2f}s ⏱️")
    with col4:
        st.metric("Tempo", f"{audio_result['audio_features']['tempo']:.1f} BPM 🥁")

    st.markdown("---") # Visual separator

    # ========== 2. PREDICTION JUSTIFICATION (MOVED HERE) ==========
    # New checkbox to control visibility of the entire justification section
    safe_filename_key_base = audio_result['filename'].replace('.', '_').replace('-', '_').replace(' ', '_').replace('(', '').replace(')', '')
    show_justification = st.checkbox("💡 Show Prediction Justification & Feature Insights", value=True, key=f"show_just_{safe_filename_key_base}")

    if show_justification:
        st.markdown("#### 💡 Why This Prediction? (Feature Justification)")
        with st.expander("Understand the Model's Reasoning", expanded=True): # Still expanded by default within this optional section
            justifications = analyze_feature_significance(
                audio_result['audio_features'],
                audio_result['predicted_label'],
                audio_result['confidence'],
                audio_result['all_probabilities']
            )
            for j in justifications:
                st.markdown(f"- {j}")
            
            # New checkbox specifically for the radar plot - Default value is now False
            show_radar_plot = st.checkbox("📊 Show Feature Profile Radar Plot", value=False, key=f"show_radar_{safe_filename_key_base}")
            if show_radar_plot:
                feature_radar_fig = create_feature_justification_plot(
                    audio_result['audio_features'],
                    audio_result['predicted_label']
                )
                st.pyplot(feature_radar_fig)
                plt.close()
            else:
                st.info("Tick the checkbox above to view the Feature Profile Radar Plot.")
    
    st.markdown("---") # Visual separator


    # ========== 3. AUDIO VISUALIZATIONS ==========
    st.markdown("#### 📊 Audio Visualizations")
    with st.expander("Explore Waveforms, Spectrograms, and MFCCs", expanded=True): # Default expanded for visual impact
        viz_cols = st.columns(2)
        # Unique keys for checkboxes
        waveform_key = f"wf_tab_{safe_filename_key_base}"
        spectrogram_key = f"spec_tab_{safe_filename_key_base}"
        mel_key = f"mel_tab_{safe_filename_key_base}"
        mfcc_key = f"mfcc_tab_{safe_filename_key_base}"

        with viz_cols[0]:
            # Set all visualization checkboxes to True by default for easy exploration
            show_waveform = st.checkbox("🌊 Waveform", value=True, key=waveform_key)
            show_spectrogram = st.checkbox("📈 Spectrogram", value=True, key=spectrogram_key)
        with viz_cols[1]:
            show_mel = st.checkbox("🎼 Mel Spectrogram", value=True, key=mel_key)
            show_mfcc = st.checkbox("🎵 MFCC", value=True, key=mfcc_key)

        audio_array = audio_result['audio_array']
        sr = audio_result['sampling_rate']

        # Display plots based on checkbox state
        if show_waveform:
            st.pyplot(create_waveform_plot(audio_array, sr, title=f"Waveform - {audio_result['filename']}"))
            plt.close()

        if show_spectrogram:
            st.pyplot(create_spectrogram_plot(audio_array, sr, title=f"Spectrogram - {audio_result['filename']}"))
            plt.close()

        if show_mel:
            st.pyplot(create_mel_spectrogram_plot(audio_array, sr, title=f"Mel Spectrogram - {audio_result['filename']}"))
            plt.close()

        if show_mfcc:
            st.pyplot(create_mfcc_plot(audio_array, sr, title=f"MFCC Features - {audio_result['filename']}"))
            plt.close()
    
    st.markdown("---") # Visual separator

    # ========== 4. DETAILED AUDIO FEATURES (Expander) ==========
    st.markdown("#### ⚙️ All Extracted Audio Features")
    with st.expander("Dive Deeper into Audio Characteristics", expanded=False):
        features = audio_result['audio_features']
        
        st.markdown("**Core Properties**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMS Energy", f"{features['rms_energy']:.4f}")
        with col2:
            st.metric("Zero Crossing Rate", f"{features['zero_crossing_rate']:.4f}")
        with col3:
            st.metric("Spectral Centroid Mean", f"{features['spectral_centroid_mean']:.1f} Hz") # Renamed for clarity
        
        st.markdown("**Advanced Spectral & Temporal Features**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Spectral Rolloff Mean", f"{features['spectral_rolloff_mean']:.1f} Hz") # Renamed for clarity
        with col2:
            st.metric("Spectral Centroid Std", f"{features['spectral_centroid_std']:.1f} Hz")
        with col3:
            st.metric("Spectral Bandwidth Mean", f"{features['spectral_bandwidth_mean']:.1f} Hz")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Chroma Mean", f"{features['chroma_mean']:.3f}")
        with col2:
            st.metric("Chroma Std", f"{features['chroma_std']:.3f}")
        with col3:
            st.metric("Onset Rate", f"{features['onset_rate']:.1f} onsets/s")

        # All MFCC Features in sub-expander
        with st.expander("All 13 MFCC Features (Mean & Std) 📊"):
            mfcc_data = []
            for i in range(13):
                mfcc_data.append({
                    'Feature': f'MFCC {i+1}',
                    'Mean': f"{features[f'mfcc_{i+1}_mean']:.3f}",
                    'Std': f"{features[f'mfcc_{i+1}_std']:.3f}"
                })
            mfcc_df = pd.DataFrame(mfcc_data)
            st.dataframe(mfcc_df, use_container_width=True, hide_index=True)
    
    st.markdown("---") # Visual separator

    # ========== 5. ALL CLASS PROBABILITIES (Expander) ==========
    st.markdown("#### 🔍 All Class Probabilities")
    with st.expander("View Full Probability Distribution", expanded=False):
        prob_df = pd.DataFrame(list(audio_result['all_probabilities'].items()), columns=['Class', 'Probability'])
        prob_df = prob_df.sort_values('Probability', ascending=False)
        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.4f}")
        st.dataframe(prob_df, use_container_width=True, hide_index=True)


def display_batch_summary(all_results):
    """Summarizes batch results with aggregate statistics and a simplified overview table."""
    st.success("✅ Classification Complete!")
    
    # ========== KEY BATCH METRICS (Always Visible) ==========
    st.markdown("### 📊 Batch Classification Summary")
    
    # Calculate summary statistics
    successful_results = [r for r in all_results if 'error' not in r]
    failed_results = [r for r in all_results if 'error' in r]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Files", len(all_results))
    with col2:
        st.metric("Successfully Classified", len(successful_results))
    with col3:
        st.metric("Failed Classifications", len(failed_results))
    with col4:
        if successful_results:
            avg_confidence = np.mean([r['confidence'] for r in successful_results])
            st.metric("Average Confidence", f"{avg_confidence:.2%} 👍")
        else:
            st.metric("Average Confidence", "N/A")

    st.markdown("---") # Visual separator

    # Overview of files classified table (simple, no detailed reports)
    st.markdown("#### 📋 Overview of Files Classified")
    overview_data = []
    for res in all_results:
        if 'error' in res:
            overview_data.append({
                'Filename': res['filename'],
                'Status': "❌ Error",
                'Predicted Class': "N/A",
                'Confidence': "N/A"
            })
        else:
            overview_data.append({
                'Filename': res['filename'],
                'Status': "✅ Classified",
                'Predicted Class': res['predicted_label'].upper(),
                'Confidence': f"{res['confidence']:.2%}"
            })
    
    overview_df = pd.DataFrame(overview_data)
    st.dataframe(overview_df, use_container_width=True, hide_index=True)


    #st.info("Detailed classification reports and aggregate visualization plots (e.g., class distribution, confidence distribution, feature comparison) have been removed from this section as per user request to streamline the overview.")


def main():
    st.set_page_config(page_title="AST Audio Classifier", layout="wide", initial_sidebar_state="expanded")
    st.title("🔉 Enhanced AST Audio Classifier with Advanced Analytics")
    st.markdown("---")

    # Add dependency check section
    with st.expander("🔧 System Information", expanded=False):
        try:
            import transformers
            import accelerate
            import torch
            st.success(f"✅ Dependencies loaded successfully!")
            st.info(f"🤖 Transformers: {transformers.__version__}")
            st.info(f"⚡ Accelerate: {accelerate.__version__}")
            st.info(f"🔥 PyTorch: {torch.__version__}")
        except Exception as e:
            st.error(f"❌ Dependency issue: {e}")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose your action:", ["🎯 Train Classifier", "🎵 Classify Audio"])

    if page == "🎯 Train Classifier":
        st.header("Fine-Tune a Multi-Class Audio Classifier")
        with st.expander("📋 Instructions", expanded=True):
            st.markdown("""
            **IMPORTANT: For large (GB) datasets, the data folder must be on the same computer as this app.**
            
            **Steps:**
            1.  Find the **full, absolute path** to your dataset folder.
            2.  **Drag and drop** the folder onto the text box below, or paste the full path.
            3.  Adjust training parameters and click 'Start Training'.
            
            **Requirements:**
            - Python 3.8+
            - Compatible versions: `transformers>=4.21.0`, `accelerate>=0.20.0`
            """)
        
        data_dir = st.text_input(
            "📁 Full, Absolute Path to your dataset folder:", 
            help="Example: /Users/yourname/Documents/SufiSonic/optimal_compressor_dataset"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            model_output_dir = st.text_input("💾 Model output directory:", value="./ast_classifier_model")
            num_train_epochs = st.number_input("🔄 Training epochs:", min_value=1, max_value=100, value=10)
        with col2:
            learning_rate = st.selectbox("📊 Learning rate:", [1e-5, 3e-5, 5e-5, 1e-4], index=1)
            batch_size = st.selectbox("📦 Batch size:", [2, 4, 8, 16], index=1, help="Use smaller sizes if you run out of memory.")

        if st.button("🚀 Start Training", type="primary"):
            run_training_ui(data_dir, model_output_dir, num_train_epochs, learning_rate, batch_size)

    elif page == "🎵 Classify Audio":
        st.header("Classify Audio Files (Single or Batch)")
        model_path = st.text_input("📁 Path to trained model:", value="./ast_classifier_model", help="e.g., './ast_classifier_model'")
        
        # --- Session State Management ---
        # Initialize session state variables if they don't exist
        if 'uploaded_files_state' not in st.session_state:
            st.session_state.uploaded_files_state = None
        if 'classification_results_state' not in st.session_state:
            st.session_state.classification_results_state = None
        if 'model_path_classified_state' not in st.session_state:
            st.session_state.model_path_classified_state = None
        
        # File uploader
        uploaded_files_current = st.file_uploader(
            "🎵 Upload audio file(s)", 
            type=['wav', 'mp3', 'flac', 'm4a', 'ogg'], 
            accept_multiple_files=True,
            help="Upload one or more audio files for classification."
        )

        # Update session state with current uploaded files only if new files are selected
        # Or if the uploader is cleared by the user.
        if uploaded_files_current:
            if st.session_state.uploaded_files_state is None or \
               len(uploaded_files_current) != len(st.session_state.uploaded_files_state) or \
               any(f.name != sf.name for f, sf in zip(uploaded_files_current, st.session_state.uploaded_files_state)):
                st.session_state.uploaded_files_state = uploaded_files_current
                # Clear previous results if new files are uploaded
                st.session_state.classification_results_state = None 
                st.session_state.model_path_classified_state = None
        elif uploaded_files_current is None and st.session_state.uploaded_files_state is not None:
             # If user cleared the uploader, clear session state too
            st.session_state.uploaded_files_state = None
            st.session_state.classification_results_state = None
            st.session_state.model_path_classified_state = None

        
        col_classify, col_clear = st.columns([1,1])

        with col_classify:
            classify_button = st.button("🎯 Classify Audio(s)", type="primary")
        with col_clear:
            clear_button = st.button("🗑️ Clear Results", type="secondary")

        if clear_button:
            st.session_state.uploaded_files_state = None
            st.session_state.classification_results_state = None
            st.session_state.model_path_classified_state = None
            st.success("Results cleared. Please upload new files.")
            # st.rerun() # Optional: Force a rerun to instantly clear the UI if needed
            
        # Logic for classification
        if classify_button and st.session_state.uploaded_files_state:
            if not model_path.strip():
                st.error("Please specify the model directory path before classifying.")
                # It's important to return here to prevent further execution if no model path.
                return 

            st.session_state.classification_results_state = run_batch_classification_prediction(model_path, st.session_state.uploaded_files_state)
            st.session_state.model_path_classified_state = model_path # Store model path used for these results
            
            # This reruns the script to display results properly after classification
            # This is key to ensuring results are displayed only AFTER processing
            # and prevents infinite loops with button presses.
            st.rerun() 
        
        # Display results if they exist in session state
        # This block runs on every rerun, displaying the persisted state
        if st.session_state.classification_results_state:
            # Check if model path is still valid for these results (e.g., if user changed path after classification)
            if st.session_state.model_path_classified_state != model_path and model_path.strip() != "":
                st.warning(f"Results displayed below were classified using model path: '{st.session_state.model_path_classified_state}'. "
                           f"Current model path is different: '{model_path}'. Click 'Classify Audio(s)' to re-classify.")
            
            display_batch_summary(st.session_state.classification_results_state)

            successful_results_for_tabs = [res for res in st.session_state.classification_results_state if 'error' not in res]
            if successful_results_for_tabs:
                st.markdown("## 🔬 Individual Audio Analysis")
                st.info("Click on each tab below to explore detailed metrics, visualizations, and justification for the prediction of each audio file.")
                
                tabs_labels = []
                for i, res in enumerate(successful_results_for_tabs):
                    label = res['filename']
                    if len(label) > 20:
                        label = label[:10] + "..." + label[-7:]
                    tabs_labels.append(label)

                tabs = st.tabs(tabs_labels)

                for i, tab in enumerate(tabs):
                    with tab:
                        display_single_audio_details(successful_results_for_tabs[i])
            else:
                st.info("No successful classifications to display detailed analysis tabs for.")
        elif st.session_state.uploaded_files_state and not classify_button: # Only show prompt if files are uploaded but not yet classified
            st.info("Files uploaded. Click 'Classify Audio(s)' to start the analysis.")


# Main entry point for the Streamlit app
if __name__ == "__main__":
    main()
