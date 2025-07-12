import numpy as np
import os
import json
import pandas as pd
import random
from scipy.io.wavfile import write as write_wav, read as read_wav
from scipy.signal import butter, filtfilt, lfilter
from tqdm import tqdm
import hashlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================================
# OPTIMAL CONFIGURATION PARAMETERS
# ================================

class OptimalConfig:
    """Optimal configuration for realistic air compressor audio generation"""

    # Core Audio Parameters
    SAMPLE_RATE = 16000
    DURATION = 10.0
    NUM_CHANNELS = 8

    # Machine Parameters (Based on real compressor analysis)
    MACHINE_BASE_FREQ = 50  # Hz (European standard, use 60 for US)
    COMPRESSOR_CYCLE_FREQ_RANGE = (1.5, 2.2)  # Hz, realistic range

    # Load and Operational Parameters
    LOAD_SCENARIOS = {
        'light': (0.3, 0.6),
        'medium': (0.6, 0.9),
        'high': (0.9, 1.2),
        'overload': (1.1, 1.4)
    }

    # SNR Scenarios (dB) - Factory realistic
    SNR_SCENARIOS = {
        'noisy_factory': (5, 12),
        'moderate_industrial': (12, 20),
        'controlled_environment': (20, 30)
    }

    # Dataset Composition (Realistic industrial distribution)
    CLASS_DISTRIBUTION = {
        'normal': 0.45,  # 45% - Most common
        'leak': 0.25,    # 25% - Very common
        'bearing_fault': 0.15,  # 15% - Critical but less frequent
        'valve_knock': 0.10,    # 10% - Serious but rare
        'belt_squeal': 0.05     # 5% - Occasional
    }

    # Anomaly Intensity Ranges (Realistic severity distribution)
    ANOMALY_INTENSITIES = {
        'leak': (0.1, 0.4),
        'bearing_fault': (0.15, 0.6),
        'valve_knock': (0.2, 0.5),
        'belt_squeal': (0.08, 0.25)
    }

    # Detailed configurations moved here for central access
    ENVIRONMENT_CONFIGS = {
        'factory': {'noise_type': 'pink', 'freq_emphasis': (100, 2000)},
        'workshop': {'noise_type': 'brown', 'freq_emphasis': (200, 1500)},
        'outdoor': {'noise_type': 'white', 'freq_emphasis': (300, 3000)}
    }

    LEAK_CONFIGS = {
        'small': {'freq_range': (2000, 6000), 'pressure_factor': 0.5},
        'medium': {'freq_range': (1500, 5000), 'pressure_factor': 0.7},
        'large': {'freq_range': (800, 4000), 'pressure_factor': 1.0}
    }

    BEARING_FAULT_MULTIPLIERS = { # Characteristic frequency multipliers
        'outer_race': 3.5,
        'inner_race': 5.4,
        'ball': 4.7,
        'cage': 0.4
    }

# ================================
# ENHANCED AUDIO GENERATION FUNCTIONS
# ================================

def generate_pink_noise(duration, sample_rate, amplitude=1.0):
    """Generate pink noise using more accurate spectral shaping"""
    num_samples = int(sample_rate * duration)
    # Generate white noise
    white = np.random.randn(num_samples)
    
    # Apply pink noise filter (1/f characteristics)
    # Coefficients from a popular pink noise generation algorithm
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    a = np.array([1, -2.494956002, 2.017265875, -0.522189496])
    pink = lfilter(b, a, white)
    
    # Normalize and apply amplitude
    if np.max(np.abs(pink)) > 0: # Avoid division by zero for silent pink noise
        pink = pink / np.max(np.abs(pink)) * amplitude
    return pink

def generate_enhanced_motor_hum(duration, sample_rate, base_freq, load_factor=1.0, 
                               amplitude=0.4, num_harmonics=5):
    """Enhanced motor hum with load-dependent harmonics and modulation"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Load-dependent frequency shift
    effective_freq = base_freq * (0.95 + 0.1 * load_factor)
    
    hum_signal = np.zeros_like(t)
    
    # Generate harmonics with realistic decay
    for i in range(1, num_harmonics + 1):
        harmonic_freq = effective_freq * i
        # Amplitude decay for higher harmonics, also influenced by load
        harmonic_amp = amplitude * (0.7 ** (i - 1)) * load_factor 
        
        # Add slight frequency modulation for realism
        fm_depth = 0.02 * harmonic_freq # Modulation depth as a percentage of harmonic frequency
        fm_rate = 0.5 + 0.3 * np.random.random() # Random modulation rate (e.g., 0.5 to 0.8 Hz)
        freq_mod = fm_depth * np.sin(2 * np.pi * fm_rate * t)
        
        # Phase accumulation for FM
        instantaneous_freq = harmonic_freq + freq_mod
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate # Integrate frequency to get phase
        
        hum_signal += harmonic_amp * np.sin(phase)
    
    return hum_signal

def generate_realistic_intake_exhaust(duration, sample_rate, load_factor=1.0, amplitude=0.15):
    """Generate realistic air intake/exhaust sound with turbulence"""
    num_samples = int(sample_rate * duration)
    
    # Base turbulent flow noise
    base_noise = generate_pink_noise(duration, sample_rate, 1.0) # Start with full amplitude pink noise
    
    # Apply band-pass filter for airflow characteristics
    nyquist = 0.5 * sample_rate
    # Dynamic cutoff frequencies based on load_factor
    low_cutoff = (150 + 50 * load_factor) / nyquist
    high_cutoff = (2500 + 500 * load_factor) / nyquist
    
    # Ensure cutoffs are valid
    low_cutoff = max(0.01, min(low_cutoff, 0.98)) # Keep within (0, 1) and ensure low < high
    high_cutoff = max(low_cutoff + 0.01, min(high_cutoff, 0.99)) # Ensure high > low and within (0,1)

    b, a = butter(3, [low_cutoff, high_cutoff], btype='band') # 3rd order Butterworth filter
    filtered_noise = filtfilt(b, a, base_noise) # Zero-phase filtering
    
    # Add pulsation due to valve operation or cyclical intake
    t = np.linspace(0, duration, num_samples, endpoint=False)
    pulse_freq = 2 * load_factor + np.random.uniform(-0.3, 0.3) # Pulsation frequency also tied to load
    pulsation = 0.7 + 0.3 * np.sin(2 * np.pi * pulse_freq * t) # Modulating envelope
    
    # Final sound scaled by amplitude and load-dependent factor
    intake_sound = filtered_noise * pulsation * amplitude * (0.8 + 0.4 * load_factor)
    
    return intake_sound

def generate_advanced_piston_cycle(duration, sample_rate, cycle_freq, amplitude=0.2):
    """Generate realistic piston cycle with compression/expansion phases"""
    num_samples = int(sample_rate * duration)
    cycle_sound = np.zeros(num_samples)
    
    if cycle_freq <= 0: # Prevent division by zero or infinite loop
        return cycle_sound 
        
    samples_per_cycle = int(sample_rate / cycle_freq)
    if samples_per_cycle == 0: # Prevent issues if cycle_freq is too high for sample_rate
        return cycle_sound

    current_sample = 0
    while current_sample < num_samples:
        # Compression phase (sharp onset) - 30% of cycle
        compression_samples = int(samples_per_cycle * 0.3)
        if compression_samples > 0 and current_sample + compression_samples <= num_samples:
            compression_env = np.exp(-np.linspace(0, 5, compression_samples)) # Exponential decay envelope
            compression_noise = np.random.randn(compression_samples) * compression_env # Modulated white noise
            cycle_sound[current_sample : current_sample + compression_samples] += compression_noise * amplitude
        
        # Expansion phase (gradual decay) - remaining 70% of cycle
        expansion_start = current_sample + compression_samples
        expansion_samples = samples_per_cycle - compression_samples
        expansion_samples = min(expansion_samples, num_samples - expansion_start) # Ensure not to overrun buffer

        if expansion_samples > 0 and expansion_start < num_samples:
            expansion_env = np.exp(-np.linspace(0, 2, expansion_samples)) # Slower decay
            
            # Generate pink noise for expansion phase
            raw_noise = np.random.randn(expansion_samples)
            b_pink = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
            a_pink = np.array([1, -2.494956002, 2.017265875, -0.522189496])
            expansion_noise_filtered = lfilter(b_pink, a_pink, raw_noise)
            
            if np.max(np.abs(expansion_noise_filtered)) > 0: # Normalize
                expansion_noise_filtered = expansion_noise_filtered / np.max(np.abs(expansion_noise_filtered))
            
            cycle_sound[expansion_start : expansion_start + expansion_samples] += \
                expansion_noise_filtered * expansion_env * amplitude * 0.3 # Softer than compression
        
        # Add slight randomness to cycle timing for naturalness
        current_sample += samples_per_cycle + random.randint(-int(samples_per_cycle * 0.02), int(samples_per_cycle * 0.02))
    
    return cycle_sound

# ================================
# ADVANCED ANOMALY GENERATION
# ================================

def generate_realistic_leak(duration, sample_rate, intensity=0.3, leak_type='small',
                            leak_configs=None):
    """Generate realistic leak sound based on leak size and pressure"""
    if leak_configs is None: 
        leak_configs = OptimalConfig.LEAK_CONFIGS # Use central config
    
    config = leak_configs.get(leak_type, leak_configs['medium']) # Default to medium if type is unknown
    
    base_noise = generate_pink_noise(duration, sample_rate, 1.0)
    
    nyquist = 0.5 * sample_rate
    low_cutoff = config['freq_range'][0] / nyquist
    high_cutoff = config['freq_range'][1] / nyquist

    # Ensure valid cutoffs
    low_cutoff = max(0.01, min(low_cutoff, 0.98))
    high_cutoff = max(low_cutoff + 0.01, min(high_cutoff, 0.99))

    b, a = butter(4, [low_cutoff, high_cutoff], btype='band') # 4th order filter for sharper cutoff
    leak_sound_filtered = filtfilt(b, a, base_noise)
    
    # Add pressure-dependent amplitude modulation
    t = np.linspace(0, duration, len(leak_sound_filtered), endpoint=False)
    pressure_variation = 0.9 + 0.1 * np.sin(2 * np.pi * 0.1 * t)  # Slow pressure variation (0.1 Hz)
    
    leak_sound = leak_sound_filtered * pressure_variation * config['pressure_factor'] * intensity
    
    return leak_sound

def generate_progressive_bearing_fault(duration, sample_rate, fundamental_freq, 
                                     severity=0.3, fault_type='outer_race',
                                     bearing_fault_multipliers=None):
    """Generate bearing fault with realistic fault frequencies"""
    if bearing_fault_multipliers is None:
        bearing_fault_multipliers = OptimalConfig.BEARING_FAULT_MULTIPLIERS # Use central config

    fault_multiplier = bearing_fault_multipliers.get(fault_type, bearing_fault_multipliers['outer_race'])
    
    if fundamental_freq <=0: # Avoid issues with zero or negative fundamental frequency
        return np.zeros(int(sample_rate * duration))

    fault_freq = fundamental_freq * fault_multiplier
    if fault_freq <= 0: # If multiplier is zero or negative, no fault impacts
        return np.zeros(int(sample_rate * duration))

    num_samples = int(sample_rate * duration)
    fault_signal = np.zeros(num_samples)
    
    impact_interval = int(sample_rate / fault_freq)
    if impact_interval == 0: # Fault freq too high for sample rate to produce distinct impacts
        return fault_signal # Or generate continuous noise

    impact_duration = int(sample_rate * 0.01)  # 10ms impact duration
    if impact_duration == 0: impact_duration = 1 # Ensure at least 1 sample

    current_sample = 0
    impact_count = 0
    max_possible_impacts = (duration * fault_freq) # For progressive factor normalization
    
    while current_sample < num_samples - impact_duration:
        # Progressive severity - impacts get stronger over time
        progressive_factor = 1.0
        if max_possible_impacts > 0: # Avoid division by zero
            progressive_factor = 1.0 + 0.5 * (impact_count / max_possible_impacts)
        
        impact_amplitude = severity * progressive_factor
        
        # Generate impact with exponential decay
        impact_env = np.exp(-np.linspace(0, 8, impact_duration))
        impact_noise_segment = np.random.randn(impact_duration) * impact_env * impact_amplitude
        
        # Add to signal, ensuring no overrun
        end_idx = min(current_sample + impact_duration, num_samples)
        actual_segment_len = end_idx - current_sample
        fault_signal[current_sample:end_idx] += impact_noise_segment[:actual_segment_len]
        
        # Add slight randomness to timing (jitter)
        jitter = random.randint(-int(impact_interval * 0.1), int(impact_interval * 0.1))
        current_sample += impact_interval + jitter
        impact_count += 1
    
    # Add continuous background grinding for severe faults
    if severity > 0.4:
        grinding = generate_pink_noise(duration, sample_rate, severity * 0.3)
        # Filter grinding to a typical bearing noise frequency range
        ny_grind = 0.5 * sample_rate
        grind_low = 200 / ny_grind
        grind_high = 2000 / ny_grind
        if grind_low < grind_high and grind_low > 0 and grind_high < 1:
            b_g, a_g = butter(2, [grind_low, grind_high], btype='band')
            grinding = filtfilt(b_g, a_g, grinding)
            fault_signal += grinding
    
    return fault_signal

def generate_valve_knock_with_timing(duration, sample_rate, cycle_freq, intensity=0.3):
    """Generate valve knock synchronized with compressor cycle"""
    num_samples = int(sample_rate * duration)
    knock_signal = np.zeros(num_samples)
    
    if cycle_freq <= 0: return knock_signal # Avoid division by zero
    cycle_interval = int(sample_rate / cycle_freq)
    if cycle_interval == 0: return knock_signal

    knock_duration = int(sample_rate * 0.003)  # 3ms sharp knock
    if knock_duration == 0: knock_duration = 1 # Ensure at least 1 sample

    current_sample = 0
    while current_sample < num_samples - knock_duration: # Ensure there's room for a knock
        # Knock occurs at specific phase of compression cycle
        knock_phase_delay = int(cycle_interval * 0.2)  # 20% into cycle (relative to start of this conceptual cycle)
        knock_position = current_sample + knock_phase_delay
        
        if knock_position < num_samples - knock_duration: # Check again after adding delay
            end_idx = knock_position + knock_duration # Already checked this won't exceed num_samples
            segment_length = knock_duration # Should be knock_duration
            
            # Create a sharp, transient pulse
            amp = intensity + random.uniform(-0.1, 0.1) * intensity # Randomize intensity slightly
            amp = np.clip(amp, 0.05, 1.0) # Clip amplitude
            
            impulse = np.exp(-np.linspace(0, 5, segment_length)) * amp * (np.random.rand(segment_length) * 2 - 1)
            
            # Add metallic resonance
            resonance_freq = random.uniform(2000, 3000) # Vary resonance freq
            t_knock = np.linspace(0, segment_length / sample_rate, segment_length, endpoint=False)
            metallic_ring = np.sin(2 * np.pi * resonance_freq * t_knock) * np.exp(-np.linspace(0, 10, segment_length)) # Decay faster
            
            combined_knock = (impulse + metallic_ring * 0.5)
            knock_signal[knock_position:end_idx] += combined_knock
        
        # Add timing variation to the start of the next cycle
        timing_variation = random.randint(-int(cycle_interval * 0.05), int(cycle_interval * 0.05))
        current_sample += cycle_interval + timing_variation
    
    return knock_signal

def generate_intermittent_belt_squeal(duration, sample_rate, intensity=0.2):
    """Generate realistic intermittent belt squeal"""
    num_samples = int(sample_rate * duration)
    squeal_signal = np.zeros(num_samples)
    
    current_time_seconds = 0.0
    
    while current_time_seconds < duration:
        # Random chance of squeal event starting
        if random.random() < 0.3:  # 30% chance of squeal event
            squeal_duration_seconds = random.uniform(0.1, 0.8)  # 0.1-0.8 second squeals
            
            # Ensure squeal doesn't exceed total duration
            if current_time_seconds + squeal_duration_seconds > duration:
                squeal_duration_seconds = duration - current_time_seconds
            
            if squeal_duration_seconds <= 0: # No time left for squeal
                break

            squeal_start_sample = int(current_time_seconds * sample_rate)
            squeal_end_sample = int((current_time_seconds + squeal_duration_seconds) * sample_rate)
            squeal_num_samples = squeal_end_sample - squeal_start_sample
            
            if squeal_num_samples > 0:
                # Generate frequency-modulated squeal
                squeal_base_freq = random.uniform(2000, 4000) # Hz
                freq_drift_hz = random.uniform(-200, 200)  # Hz drift over squeal duration
                
                squeal_time_vector = np.linspace(0, squeal_duration_seconds, squeal_num_samples, endpoint=False)
                
                # Linear frequency drift
                instantaneous_freq_vector = squeal_base_freq + (freq_drift_hz * (squeal_time_vector / squeal_duration_seconds))
                
                # Generate squeal with amplitude envelope
                # Phase accumulation for frequency modulation
                phase_vector = 2 * np.pi * np.cumsum(instantaneous_freq_vector) / sample_rate
                envelope = np.sin(np.pi * squeal_time_vector / squeal_duration_seconds) ** 2  # Smooth attack/decay (sine squared)
                
                squeal_tone = np.sin(phase_vector) * envelope * intensity
                
                squeal_signal[squeal_start_sample:squeal_end_sample] += squeal_tone
        
        # Wait before next potential squeal
        current_time_seconds += random.uniform(0.5, 2.0) 
    
    return squeal_signal

# ================================
# MULTI-CHANNEL AND ENVIRONMENTAL EFFECTS
# ================================

def simulate_multichannel_recording(mono_signal, num_channels=8, room_size='medium', 
                                    sample_rate_config=OptimalConfig.SAMPLE_RATE):
    """Simulate realistic multi-channel recording with spatial effects"""
    room_configs_local = { # Renamed to avoid conflict if OptimalConfig has a similar name
        'small': {'reverb_time': 0.2, 'max_delay_s': 0.005}, # max_delay in seconds
        'medium': {'reverb_time': 0.5, 'max_delay_s': 0.010},
        'large': {'reverb_time': 1.0, 'max_delay_s': 0.020}
    }
    selected_room_config = room_configs_local.get(room_size, room_configs_local['medium'])
    multichannel_signal = np.zeros((num_channels, len(mono_signal)))
    
    for ch in range(num_channels):
        # Simulate microphone position effects
        distance_factor = 0.8 + 0.4 * (ch / (num_channels -1 if num_channels > 1 else 1) ) # Varying distances, avoid div by zero
        attenuated_signal = mono_signal * distance_factor
        
        # Time delay for different positions
        delay_samples = int(random.uniform(0, selected_room_config['max_delay_s']) * sample_rate_config)
        
        delayed_signal = np.copy(attenuated_signal) # Start with a copy
        if delay_samples > 0:
            delayed_signal = np.roll(attenuated_signal, delay_samples)
            delayed_signal[:delay_samples] = 0  # Zero-pad beginning
        
        # Simple reverb simulation
        if selected_room_config['reverb_time'] > 0:
            reverb_delay_samples = int(selected_room_config['reverb_time'] * sample_rate_config)
            if reverb_delay_samples < len(delayed_signal) and reverb_delay_samples > 0: # Ensure valid delay for reverb
                reverb_component = np.zeros_like(delayed_signal)
                reverb_component[reverb_delay_samples:] = delayed_signal[:-reverb_delay_samples] * 0.3 # Simple single reflection
                delayed_signal += reverb_component # Add reverb
        
        # Add channel-specific noise
        channel_noise_duration_s = len(mono_signal) / sample_rate_config
        if channel_noise_duration_s > 0:
            channel_noise = generate_pink_noise(channel_noise_duration_s, sample_rate_config, 0.02) # Small amplitude noise
            # Ensure channel_noise length matches delayed_signal if there were rounding issues
            if len(channel_noise) == len(delayed_signal):
                 multichannel_signal[ch, :] = delayed_signal + channel_noise
            else: # Fallback if lengths mismatch (should be rare)
                 multichannel_signal[ch, :] = delayed_signal[:len(channel_noise)] + channel_noise if len(channel_noise) < len(delayed_signal) else delayed_signal + channel_noise[:len(delayed_signal)]

        else:
            multichannel_signal[ch, :] = delayed_signal


    return multichannel_signal

def add_environmental_noise(signal, environment='factory', snr_db=15,
                            sample_rate_config=OptimalConfig.SAMPLE_RATE,
                            env_configs_passed=None): # Renamed for clarity
    """Add realistic environmental noise based on setting"""
    if env_configs_passed is None:
        env_configs_passed = OptimalConfig.ENVIRONMENT_CONFIGS # Use central config

    selected_env_config = env_configs_passed.get(environment, env_configs_passed['factory']) # Default to factory
    
    signal_duration_s = len(signal) / sample_rate_config
    if signal_duration_s <= 0: return signal # No signal to add noise to

    # Generate base environmental noise
    if selected_env_config['noise_type'] == 'pink':
        env_noise_base = generate_pink_noise(signal_duration_s, sample_rate_config, 1.0)
    elif selected_env_config['noise_type'] == 'brown':
        # Brownian noise: cumulative sum of white noise
        env_noise_base = np.cumsum(np.random.randn(len(signal))) # Match signal length
        if np.max(np.abs(env_noise_base)) > 0: # Normalize
            env_noise_base = env_noise_base / np.max(np.abs(env_noise_base))
    else:  # 'white' noise
        env_noise_base = np.random.randn(len(signal)) # Match signal length
    
    # Apply frequency emphasis
    nyquist = 0.5 * sample_rate_config
    low_cutoff = selected_env_config['freq_emphasis'][0] / nyquist
    high_cutoff = selected_env_config['freq_emphasis'][1] / nyquist

    # Robust cutoff validation
    low_cutoff = max(0.001, min(low_cutoff, 0.998)) # Ensure >0 and <1
    high_cutoff = max(low_cutoff + 0.001, min(high_cutoff, 0.999)) # Ensure high > low and <1
    if low_cutoff >= high_cutoff : # Final fallback
        low_cutoff = 0.01
        high_cutoff = 0.99

    b, a = butter(2, [low_cutoff, high_cutoff], btype='band') # 2nd order filter
    env_noise_filtered = filtfilt(b, a, env_noise_base)
    
    # Scale noise according to SNR
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(env_noise_filtered ** 2)
    
    scaled_noise_component = np.zeros_like(env_noise_filtered) # Initialize
    if noise_power > 1e-12: # Avoid division by zero or issues with silent noise
        snr_linear = 10 ** (snr_db / 10)
        # Ensure signal_power is positive and the denominator term for sqrt is positive
        if signal_power > 0 and snr_linear * noise_power > 0:
            required_noise_power = signal_power / snr_linear
            noise_scale_factor = np.sqrt(required_noise_power / noise_power)
            scaled_noise_component = env_noise_filtered * noise_scale_factor
        # If signal_power is zero, for any finite SNR, noise should also be zero.
        # If SNR is infinite (very large snr_db), noise should be zero.
        # The current formula correctly makes noise_scale_factor very small for large snr_linear.
    
    return signal + scaled_noise_component

# <<< NEW FUNCTION to add global Gaussian noise >>>
def add_global_gaussian_noise(signal_data, noise_std_dev):
    """
    Adds Gaussian noise to the input signal data.
    Works for both mono and multichannel (expects channels x samples for multichannel).

    Args:
        signal_data (np.ndarray): The input signal. Can be 1D (mono) or 2D (multichannel).
        noise_std_dev (float): Standard deviation of the Gaussian noise.
                               If 0 or negative, no noise is added.

    Returns:
        np.ndarray: Signal with added Gaussian noise.
    """
    if noise_std_dev <= 0:
        return signal_data
    
    # Generate Gaussian noise with the same shape as the input signal_data
    noise = np.random.normal(loc=0.0, scale=noise_std_dev, size=signal_data.shape)
    return signal_data + noise
# <<< END NEW FUNCTION >>>

# ================================
# DATASET GENERATION PIPELINE
# ================================

class OptimalDatasetGenerator:
    """Comprehensive dataset generator with optimal configurations"""
    
    def __init__(self, output_dir="optimal_compressor_dataset"):
        self.output_dir = output_dir
        self.metadata = []
        self.config = OptimalConfig() # Instance of OptimalConfig
        
        os.makedirs(output_dir, exist_ok=True)
        for anomaly_type_key in self.config.CLASS_DISTRIBUTION.keys():
            os.makedirs(os.path.join(output_dir, anomaly_type_key), exist_ok=True)
    
    # <<< MODIFIED: Added new parameter 'additional_gaussian_noise_std' with default 0.0 >>>
    def generate_single_sample(self, anomaly_type, sample_id, load_scenario='medium', 
                             snr_scenario='moderate_industrial', environment='factory',
                             additional_gaussian_noise_std=0.0):
        """Generate a single optimized audio sample"""
        
        load_range = self.config.LOAD_SCENARIOS[load_scenario]
        snr_range = self.config.SNR_SCENARIOS[snr_scenario]
        
        load_factor = random.uniform(*load_range)
        snr_db = random.uniform(*snr_range)

        base_cycle_freq_val = random.uniform(*self.config.COMPRESSOR_CYCLE_FREQ_RANGE)
        cycle_freq = base_cycle_freq_val * load_factor
        if cycle_freq <= 0: cycle_freq = 0.1 # Ensure positive

        # Generate base compressor sounds
        motor_hum = generate_enhanced_motor_hum(
            self.config.DURATION, self.config.SAMPLE_RATE, 
            self.config.MACHINE_BASE_FREQ, load_factor, amplitude=0.4
        )
        intake_exhaust = generate_realistic_intake_exhaust(
            self.config.DURATION, self.config.SAMPLE_RATE, 
            load_factor, amplitude=0.2
        )
        piston_cycle = generate_advanced_piston_cycle(
            self.config.DURATION, self.config.SAMPLE_RATE, 
            cycle_freq, amplitude=0.15
        )
        base_signal = motor_hum + intake_exhaust + piston_cycle
        
        # Initialize anomaly & detailed metadata fields
        anomaly_signal = np.zeros_like(base_signal)
        anomaly_intensity_val = 0.0
        anomaly_subtype_val = 'none'
        
        leak_config_freq_low_hz = np.nan
        leak_config_freq_high_hz = np.nan
        leak_config_pressure_factor = np.nan
        bearing_fault_target_freq_hz = np.nan
        
        if anomaly_type != 'normal':
            intensity_range = self.config.ANOMALY_INTENSITIES[anomaly_type]
            anomaly_intensity_val = random.uniform(*intensity_range)
            
            if anomaly_type == 'leak':
                leak_types = list(self.config.LEAK_CONFIGS.keys())
                anomaly_subtype_val = random.choice(leak_types)
                anomaly_signal = generate_realistic_leak(
                    self.config.DURATION, self.config.SAMPLE_RATE, 
                    anomaly_intensity_val, anomaly_subtype_val,
                    leak_configs=self.config.LEAK_CONFIGS # Pass central config
                )
                selected_leak_config = self.config.LEAK_CONFIGS.get(anomaly_subtype_val, {})
                leak_config_freq_low_hz = selected_leak_config.get('freq_range', (np.nan, np.nan))[0]
                leak_config_freq_high_hz = selected_leak_config.get('freq_range', (np.nan, np.nan))[1]
                leak_config_pressure_factor = selected_leak_config.get('pressure_factor', np.nan)
            
            elif anomaly_type == 'bearing_fault':
                fault_types = list(self.config.BEARING_FAULT_MULTIPLIERS.keys())
                anomaly_subtype_val = random.choice(fault_types)
                anomaly_signal = generate_progressive_bearing_fault(
                    self.config.DURATION, self.config.SAMPLE_RATE,
                    self.config.MACHINE_BASE_FREQ, anomaly_intensity_val, anomaly_subtype_val,
                    bearing_fault_multipliers=self.config.BEARING_FAULT_MULTIPLIERS # Pass central
                )
                multiplier = self.config.BEARING_FAULT_MULTIPLIERS.get(anomaly_subtype_val, 0)
                if self.config.MACHINE_BASE_FREQ > 0 and multiplier > 0 :
                     bearing_fault_target_freq_hz = round(self.config.MACHINE_BASE_FREQ * multiplier, 2)
                else:
                     bearing_fault_target_freq_hz = 0.0 # Or np.nan

            elif anomaly_type == 'valve_knock':
                anomaly_subtype_val = 'compression_knock'
                anomaly_signal = generate_valve_knock_with_timing(
                    self.config.DURATION, self.config.SAMPLE_RATE, 
                    cycle_freq, anomaly_intensity_val
                )
            
            elif anomaly_type == 'belt_squeal':
                anomaly_subtype_val = 'intermittent'
                anomaly_signal = generate_intermittent_belt_squeal(
                    self.config.DURATION, self.config.SAMPLE_RATE, 
                    anomaly_intensity_val
                )
        
        combined_signal = base_signal + anomaly_signal
        
        if np.max(np.abs(combined_signal)) > 0: # Normalize before adding noise
            combined_signal = combined_signal / np.max(np.abs(combined_signal)) * 0.8
        
        # Environment details for metadata
        env_details = self.config.ENVIRONMENT_CONFIGS.get(environment, self.config.ENVIRONMENT_CONFIGS['factory'])
        env_noise_type_applied = env_details['noise_type']
        env_noise_freq_low_hz = env_details['freq_emphasis'][0]
        env_noise_freq_high_hz = env_details['freq_emphasis'][1]

        noisy_signal_env = add_environmental_noise(combined_signal, environment, snr_db,
                                               sample_rate_config=self.config.SAMPLE_RATE,
                                               env_configs_passed=self.config.ENVIRONMENT_CONFIGS)
        
        multichannel_signal = simulate_multichannel_recording(
            noisy_signal_env, self.config.NUM_CHANNELS, room_size='medium', # room_size currently fixed
            sample_rate_config=self.config.SAMPLE_RATE
        )

        # <<< MODIFIED: Apply additional global Gaussian noise if specified >>>
        if additional_gaussian_noise_std > 0:
            multichannel_signal = add_global_gaussian_noise(
                multichannel_signal,
                additional_gaussian_noise_std
            )
        # <<< END MODIFICATION >>>
        
        max_val = np.max(np.abs(multichannel_signal))
        if max_val > 0: # Final normalization for 16-bit audio
            multichannel_signal = multichannel_signal / max_val * 0.9
        
        audio_int16 = (multichannel_signal * 32767).astype(np.int16)
        
        filename = f"{anomaly_type}_{load_scenario}_{snr_scenario}_{environment}_{sample_id:04d}.wav"
        filepath = os.path.join(self.output_dir, anomaly_type, filename)
        
        file_saved_successfully = False
        try:
            write_wav(filepath, self.config.SAMPLE_RATE, audio_int16.T) # Transpose for scipy.io.wavfile
            file_saved_successfully = True
        except Exception as e:
            print(f"Error saving {filepath}: {e}")
            # return None # Decide if an error here should halt metadata entry
        
        # Create metadata entry
        metadata_entry = {
            'filename': filename,
            'filepath': filepath,
            'anomaly_type': anomaly_type,
            'anomaly_subtype': anomaly_subtype_val,
            'anomaly_intensity': round(anomaly_intensity_val, 3),
            'load_factor': round(load_factor, 3),
            'load_scenario': load_scenario,
            'snr_db': round(snr_db, 1),
            'snr_scenario': snr_scenario,
            'environment': environment,
            'cycle_frequency_hz': round(cycle_freq, 2),
            'duration_s': self.config.DURATION,
            'sample_rate_hz': self.config.SAMPLE_RATE,
            'num_channels': self.config.NUM_CHANNELS,
            'machine_base_freq_hz': self.config.MACHINE_BASE_FREQ,
            'env_noise_type_applied': env_noise_type_applied,
            'env_noise_freq_low_hz': env_noise_freq_low_hz,
            'env_noise_freq_high_hz': env_noise_freq_high_hz,
            'leak_config_freq_low_hz': leak_config_freq_low_hz,
            'leak_config_freq_high_hz': leak_config_freq_high_hz,
            'leak_config_pressure_factor': leak_config_pressure_factor,
            'bearing_fault_target_freq_hz': bearing_fault_target_freq_hz,
            'additional_gaussian_noise_std': round(additional_gaussian_noise_std, 5), # <<< MODIFIED: Added metadata field >>>
            'file_hash': "NotCalculated", # Default if file not saved
            'generation_timestamp': datetime.now().isoformat()
        }
        
        if file_saved_successfully:
            try:
                with open(filepath, 'rb') as f_hash:
                    metadata_entry['file_hash'] = hashlib.md5(f_hash.read()).hexdigest()[:8]
            except Exception as e_hash:
                print(f"Error calculating hash for {filepath}: {e_hash}")
                metadata_entry['file_hash'] = "HashError"
        
        return metadata_entry
    
    def generate_balanced_dataset(self, total_samples=1000, stratify_by_scenarios=True): # stratify_by_scenarios kept for API, simplified internally
        """Generate a balanced, realistic dataset with simplified scenario distribution"""
        print(f"Generating optimal dataset with {total_samples} samples...")
        print(f"Class distribution: {self.config.CLASS_DISTRIBUTION}")
        
        samples_per_class = {}
        for anomaly_type_key, proportion in self.config.CLASS_DISTRIBUTION.items():
            samples_per_class[anomaly_type_key] = int(total_samples * proportion)
        
        total_allocated = sum(samples_per_class.values())
        if total_allocated < total_samples and 'normal' in samples_per_class: # Ensure 'normal' exists
            samples_per_class['normal'] += (total_samples - total_allocated)
        elif total_allocated < total_samples and samples_per_class: # Add to the first class if 'normal' doesn't exist
             first_class_key = list(samples_per_class.keys())[0]
             samples_per_class[first_class_key] += (total_samples - total_allocated)


        print(f"Samples per class: {samples_per_class}")
        
        overall_sample_id = 0 # For unique file naming if desired across all classes

        for anomaly_type_key, num_class_samples in samples_per_class.items():
            print(f"\nGenerating {num_class_samples} samples for '{anomaly_type_key}'...")
            
            load_scenarios_available = list(self.config.LOAD_SCENARIOS.keys())
            snr_scenarios_available = list(self.config.SNR_SCENARIOS.keys())
            environments_available = list(self.config.ENVIRONMENT_CONFIGS.keys())
            
            # Use tqdm for the loop generating samples for the current class
            for i in tqdm(range(num_class_samples), desc=f"Class: {anomaly_type_key}"):
                # Simple round-robin for scenario distribution within the class
                current_load_scenario = load_scenarios_available[i % len(load_scenarios_available)]
                current_snr_scenario = snr_scenarios_available[i % len(snr_scenarios_available)]
                current_environment = environments_available[i % len(environments_available)]
                
                # sample_id for filename can be i (resets per class) or overall_sample_id
                # <<< MODIFIED: Calls generate_single_sample with default additional_gaussian_noise_std=0.0
                # User can modify this call if they want to apply this noise systematically during batch generation.
                # e.g., pass additional_gaussian_noise_std=0.005 if desired for all samples.
                metadata = self.generate_single_sample(
                    anomaly_type_key,
                    i, # Using per-class index for sample_id in filename
                    load_scenario=current_load_scenario,
                    snr_scenario=current_snr_scenario,
                    environment=current_environment,
                    additional_gaussian_noise_std=0.0 # Default, no additional noise here unless changed
                )
                if metadata:
                    self.metadata.append(metadata)
                overall_sample_id +=1
        
        self.save_metadata()
        self.generate_dataset_report()
        print(f"\nDataset generation complete!")
        print(f"Total samples generated: {len(self.metadata)}")
        print(f"Dataset saved to: {self.output_dir}")

    def save_metadata(self):
        """Save comprehensive metadata"""
        if not self.metadata:
            print("No metadata generated to save.")
            return

        df = pd.DataFrame(self.metadata)
        csv_path = os.path.join(self.output_dir, 'metadata.csv')
        try:
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error saving metadata to CSV {csv_path}: {e}")
            return # Stop if CSV can't be saved


        json_metadata = {
            'dataset_info': {
                'total_samples': len(self.metadata),
                'generation_date': datetime.now().isoformat(),
                'configuration': {
                    'sample_rate_hz': self.config.SAMPLE_RATE,
                    'duration_s': self.config.DURATION,
                    'num_channels': self.config.NUM_CHANNELS,
                    'class_distribution': self.config.CLASS_DISTRIBUTION,
                    'machine_base_freq_hz': self.config.MACHINE_BASE_FREQ,
                    'compressor_cycle_freq_range_hz': self.config.COMPRESSOR_CYCLE_FREQ_RANGE,
                    # Potentially add more config like ANOMALY_INTENSITIES if needed
                }
            },
            'samples': self.metadata # List of dicts
        }
        
        json_path = os.path.join(self.output_dir, 'metadata.json')
        try:
            with open(json_path, 'w') as f:
                def default_serializer(o): # Handles np types and NaN for JSON
                    if isinstance(o, (np.integer, np.int64)): return int(o)
                    if isinstance(o, (np.floating, np.float64)): return float(o) if pd.notna(o) else None
                    if isinstance(o, np.ndarray): return o.tolist()
                    if pd.isna(o): return None # Explicitly handle other NaNs as null
                    # Let it raise TypeError for other unhandled types to catch issues
                    return o.__dict__ # Fallback, may not always be suitable

                json.dump(json_metadata, f, indent=2, default=default_serializer)
            print(f"Metadata saved to: {csv_path} and {json_path}")
        except Exception as e:
            print(f"Error saving metadata to JSON {json_path}: {e}")


    def generate_dataset_report(self):
        """Generate comprehensive dataset analysis report"""
        if not self.metadata:
            print("No metadata available to generate a report.")
            return
            
        df = pd.DataFrame(self.metadata)
        if df.empty:
            print("Metadata DataFrame is empty. Cannot generate report.")
            return

        report = {
            'Dataset Summary': {
                'Total Samples': len(df),
                'Duration per Sample (s)': f"{self.config.DURATION}",
                'Total Dataset Duration (hours)': f"{len(df) * self.config.DURATION / 3600:.2f}",
                'Sample Rate (Hz)': f"{self.config.SAMPLE_RATE}",
                'Channels': self.config.NUM_CHANNELS
            },
            'Class Distribution': df['anomaly_type'].value_counts().to_dict(),
            'Load Scenario Distribution': df['load_scenario'].value_counts().to_dict(),
            'SNR Scenario Distribution': df['snr_scenario'].value_counts().to_dict(),
            'Environment Distribution': df['environment'].value_counts().to_dict(),
            'Environmental Noise Type Distribution': df['env_noise_type_applied'].value_counts().to_dict() if 'env_noise_type_applied' in df else 'N/A',
        }
        
        # Helper function for safe statistics
        def safe_stats(series, round_digits):
            if series.empty or series.isnull().all(): return {'Min': 'N/A', 'Max': 'N/A', 'Average': 'N/A'}
            return {
                'Min': round(series.min(), round_digits),
                'Max': round(series.max(), round_digits),
                'Average': round(series.mean(), round_digits)
            }

        if 'anomaly_intensity' in df:
            report['Anomaly Intensity Statistics'] = safe_stats(df[df['anomaly_type'] != 'normal']['anomaly_intensity'].dropna(), 3)
        if 'load_factor' in df:
            report['Load Factor Statistics'] = safe_stats(df['load_factor'].dropna(), 3)
        if 'snr_db' in df:
            report['SNR Statistics (dB)'] = safe_stats(df['snr_db'].dropna(), 1)
        if 'cycle_frequency_hz' in df:
            report['Cycle Frequency Statistics (Hz)'] = safe_stats(df['cycle_frequency_hz'].dropna(), 2)
        if 'bearing_fault_target_freq_hz' in df:
             report['Bearing Fault Target Frequency Statistics (Hz)'] = safe_stats(df['bearing_fault_target_freq_hz'].dropna(), 2)
        
        # <<< MODIFIED: Added report section for additional_gaussian_noise_std >>>
        if 'additional_gaussian_noise_std' in df.columns:
            # Only show stats if some non-zero values exist for this noise
            series_additional_noise = df['additional_gaussian_noise_std'].dropna()
            if not series_additional_noise.empty and (series_additional_noise > 1e-9).any(): # Check if any meaningful noise was added
                report['Additional Gaussian Noise Std Dev Statistics'] = safe_stats(series_additional_noise[series_additional_noise > 1e-9], 5)
            else:
                report['Additional Gaussian Noise Std Dev Statistics'] = "Not applied or all effectively zero."
        # <<< END MODIFICATION >>>

        if 'anomaly_subtype' in df.columns:
            anomaly_subtypes = df[df['anomaly_subtype'].notna() & (df['anomaly_subtype'] != 'none')]['anomaly_subtype'].value_counts().to_dict()
            if anomaly_subtypes:
                report['Anomaly Subtype Distribution'] = anomaly_subtypes
        
        report_path = os.path.join(self.output_dir, 'dataset_report.json')
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str) # Use str for simplicity if np types remain
            print(f"\nDetailed report saved to: {report_path}")
        except Exception as e:
            print(f"Error saving dataset report to {report_path}: {e}")

        # Print summary to console
        print("\n" + "="*50); print("DATASET GENERATION REPORT (Summary)"); print("="*50)
        for section, data in report.items():
            print(f"\n{section}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict): # For nested stats like Min/Max/Avg
                        print(f"  {key}:")
                        for subkey, subvalue in value.items(): print(f"    {subkey}: {subvalue}")
                    else: print(f"  {key}: {value}")
            else: print(f"  {data}")
        

# ================================
# DATASET VALIDATION AND QUALITY CHECKS
# ================================

def validate_dataset(dataset_dir):
    """Validate the generated dataset for quality and consistency"""
    print("Validating dataset...")
    metadata_path = os.path.join(dataset_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.csv not found at {metadata_path}!")
        return False
    
    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"Error reading metadata.csv: {e}")
        return False

    validation_results = {
        'total_files_expected': len(df),
        'files_found': 0,
        'files_missing': [],
        'corrupted_files': [],
        'file_size_issues': [],
        'validation_passed': True # Assume pass until a failure
    }

    if df.empty:
        print("Metadata is empty. No files to validate.")
        # Depending on requirements, an empty but valid metadata might pass or fail
        # validation_results['validation_passed'] = False 
        # Save validation report even if empty
        validation_report_path = os.path.join(dataset_dir, 'validation_report.json')
        with open(validation_report_path, 'w') as f: json.dump(validation_results, f, indent=2)
        print(f"Validation report for empty dataset saved to: {validation_report_path}")
        return validation_results['validation_passed'] # Or False

    print(f"Checking {len(df)} files based on metadata...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating files"):
        if 'filepath' not in row or pd.isna(row['filepath']):
            validation_results['files_missing'].append(f"Metadata row {idx}: Filepath missing or NaN.")
            validation_results['validation_passed'] = False
            continue
            
        filepath = str(row['filepath']) # Ensure it's a string

        if not os.path.exists(filepath):
            validation_results['files_missing'].append(filepath)
            validation_results['validation_passed'] = False
            continue
        
        validation_results['files_found'] += 1
        
        # File size check
        file_size = os.path.getsize(filepath)
        sr_val = int(row.get('sample_rate_hz', OptimalConfig.SAMPLE_RATE))
        dur_val = float(row.get('duration_s', OptimalConfig.DURATION))
        chan_val = int(row.get('num_channels', OptimalConfig.NUM_CHANNELS))
        bytes_per_sample = 2 # For int16

        expected_min_size = sr_val * dur_val * chan_val * bytes_per_sample * 0.8
        expected_max_size = sr_val * dur_val * chan_val * bytes_per_sample * 1.2
        
        if not (expected_min_size <= file_size <= expected_max_size):
            validation_results['file_size_issues'].append({
                'file': filepath, 'size_bytes': file_size,
                'expected_range_bytes': (int(expected_min_size), int(expected_max_size))
            })
            # This might not be a critical failure, depending on tolerance
            # validation_results['validation_passed'] = False 

        # Audio content check
        try:
            # read_wav is imported globally now
            sr_read, audio_data = read_wav(filepath) 
            
            if sr_read != sr_val:
                validation_results['corrupted_files'].append(f"{filepath} - wrong sample rate (expected {sr_val}, got {sr_read})")
                validation_results['validation_passed'] = False
            
            actual_channels = audio_data.shape[1] if audio_data.ndim == 2 else 1
            if actual_channels != chan_val:
                validation_results['corrupted_files'].append(f"{filepath} - wrong channel count (expected {chan_val}, got {actual_channels})")
                validation_results['validation_passed'] = False
        except Exception as e:
            validation_results['corrupted_files'].append(f"{filepath} - audio read error: {str(e)}")
            validation_results['validation_passed'] = False

    # Print validation summary
    print("\n" + "="*50); print("DATASET VALIDATION REPORT"); print("="*50)
    print(f"Total files expected (from metadata): {validation_results['total_files_expected']}")
    print(f"Files found on disk and checked: {validation_results['files_found']}")
    print(f"Files missing or with invalid paths: {len(validation_results['files_missing'])}")
    print(f"Corrupted/problematic audio files: {len(validation_results['corrupted_files'])}")
    print(f"Files with size issues: {len(validation_results['file_size_issues'])}")

    for category_name, items_list in [('Missing/Invalid Path Files', validation_results['files_missing']), 
                                     ('Corrupted/Problematic Audio Files', validation_results['corrupted_files']),
                                     ('File Size Issues', validation_results['file_size_issues'])]:
        if items_list:
            print(f"\n{category_name}:")
            for item_detail in items_list[:10]: # Show first 10 details
                if isinstance(item_detail, dict): # For size issues
                    print(f"  - File: {item_detail.get('file')}, Size: {item_detail.get('size_bytes')}, Expected: {item_detail.get('expected_range_bytes')}")
                else: # For missing/corrupted paths
                    print(f"  - {item_detail}")
            if len(items_list) > 10: print(f"  ... and {len(items_list) - 10} more.")
    
    print(f"\nOverall Validation Status: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
    
    # Save validation report to JSON
    validation_report_path = os.path.join(dataset_dir, 'validation_report.json')
    try:
        with open(validation_report_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"Validation report saved to: {validation_report_path}")
    except Exception as e:
        print(f"Error saving validation report to {validation_report_path}: {e}")
        
    return validation_results['validation_passed']

# ================================
# MAIN EXECUTION FUNCTION
# ================================
def main():
    """Main function to generate the optimal compressor dataset"""
    print("="*60); print("OPTIMAL AIR COMPRESSOR ANOMALY DETECTION DATASET GENERATOR"); print("="*60)
    config_instance = OptimalConfig()
    print(f"\nUsing Configuration:")
    print(f"  Sample Rate: {config_instance.SAMPLE_RATE} Hz")
    print(f"  Duration: {config_instance.DURATION} seconds")
    print(f"  Channels: {config_instance.NUM_CHANNELS}")
    print(f"  Anomaly Classes: {list(config_instance.CLASS_DISTRIBUTION.keys())}")
    print(f"  Target Class Distribution: {config_instance.CLASS_DISTRIBUTION}")

    try:
        total_samples_input = input(f"\nEnter total number of samples to generate (default: 100): ") # Reduced default for quick tests
        total_samples = int(total_samples_input) if total_samples_input.strip() else 100
        if total_samples <= 0: raise ValueError("Number of samples must be positive.")
    except ValueError as e:
        print(f"Invalid input for number of samples: {e}. Using default: 100 samples.")
        total_samples = 100

    output_dir_input = input(f"Enter output directory name (default: 'optimal_compressor_dataset'): ").strip()
    output_dir = output_dir_input if output_dir_input else "optimal_compressor_dataset"
    
    print(f"\nDataset Generation Parameters:")
    print(f"  Total samples to generate: {total_samples}")
    print(f"  Output directory: {os.path.abspath(output_dir)}")
    
    # Example of how you might use the new noise parameter:
    # You could ask the user for `additional_gaussian_noise_std_to_apply`
    # and then pass it to `generate_balanced_dataset` if you modify that function,
    # or pass it to `generate_single_sample` if calling it directly.
    # For this example, `generate_balanced_dataset` will use the default of 0.0.
    # To apply it, you would modify the call inside `generate_balanced_dataset`
    # e.g. additional_gaussian_noise_std=0.005 (for a small amount of noise)

    try:
        generator = OptimalDatasetGenerator(output_dir=output_dir)
        generator.generate_balanced_dataset(total_samples=total_samples) 
        
        print("\nDataset generation phase completed successfully!")
        
        validate_q_input = input("\nDo you want to validate the generated dataset? (y/n, default: y): ").strip().lower()
        if validate_q_input != 'n':
            print("\nStarting dataset validation...")
            validation_passed = validate_dataset(dataset_dir=output_dir)
            if validation_passed: 
                print("\n✓ Dataset validation PASSED - Dataset appears to be consistent and ready for use!")
            else: 
                print("\n✗ Dataset validation FAILED - Please check the validation report for details.")
        else:
            print("\nSkipping dataset validation.")
            
    except KeyboardInterrupt:
        print("\n\nDataset generation interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nAn unexpected error occurred during dataset generation or validation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nGenerator script finished.")

# ================================
# UTILITY FUNCTIONS
# ================================
def analyze_existing_dataset(dataset_dir):
    """Analyze an existing dataset's metadata.csv"""
    metadata_path = os.path.join(dataset_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found at {metadata_path}")
        return
    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"Error reading metadata file {metadata_path}: {e}")
        return

    print(f"\n--- Dataset Analysis for: {os.path.abspath(dataset_dir)} ---")
    if df.empty:
        print("Metadata is empty.")
        return
        
    print(f"Total samples in metadata: {len(df)}")
    
    # Use OptimalConfig for default duration if not in df, or make it a parameter
    duration_per_sample = df['duration_s'].iloc[0] if 'duration_s' in df.columns and not df.empty else OptimalConfig.DURATION
    total_duration_hours = len(df) * duration_per_sample / 3600
    print(f"Approx. Total Dataset Duration: {total_duration_hours:.2f} hours (assuming {duration_per_sample}s per sample)")

    # <<< MODIFIED: Added 'additional_gaussian_noise_std' to list of columns to analyze >>>
    for col in ['anomaly_type', 'load_scenario', 'snr_scenario', 'environment', 
                'env_noise_type_applied', 'anomaly_subtype', 'additional_gaussian_noise_std']:
        if col in df.columns:
            print(f"\nDistribution for '{col}':")
            if df[col].dtype == 'object' or df[col].nunique() < 20 : # Show value_counts for categorical or few unique values
                 print(df[col].value_counts(dropna=False).to_string())
            else: # For continuous numeric with many values, show describe
                 print(df[col].describe().to_string())
        else:
            print(f"\nColumn '{col}' not found in metadata.")
    print("\n--- End of Analysis ---")


def create_dataset_subset(source_dir, target_dir, subset_size=100, balanced=True):
    """Create a smaller subset from an existing generated dataset."""
    print(f"\nCreating a subset of size ~{subset_size} from '{source_dir}' into '{target_dir}' (balanced={balanced}).")
    metadata_path = os.path.join(source_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"Source metadata file not found at {metadata_path}. Cannot create subset.")
        return
    
    try:
        df_source = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"Error reading source metadata file {metadata_path}: {e}")
        return

    if df_source.empty:
        print("Source metadata is empty. Cannot create subset.")
        return
    
    actual_subset_size = min(subset_size, len(df_source)) # Don't try to sample more than available
    if actual_subset_size == 0:
        print("Subset size is zero or source is empty. No subset created.")
        return

    subset_df = pd.DataFrame()
    if balanced and 'anomaly_type' in df_source.columns:
        try:
            # Proportional sampling per anomaly_type group
            subset_df = df_source.groupby('anomaly_type', group_keys=False).apply(
                lambda x: x.sample(n=max(1, int(np.ceil(actual_subset_size * len(x) / len(df_source)))), 
                                   random_state=42, replace=False) if len(x) > 0 else x.head(0) # handle empty groups
            )
            # If due to max(1,...) the total is over subset_size, trim it down
            if len(subset_df) > actual_subset_size:
                subset_df = subset_df.sample(n=actual_subset_size, random_state=42, replace=False)
            elif len(subset_df) < actual_subset_size and len(subset_df) < len(df_source) : # If we got less than desired but more could be taken
                 pass # For now, accept the slightly smaller balanced set if it happens

        except Exception as e_balanced_sample: # Catch potential sampling errors (e.g. group too small)
            print(f"Warning: Error during balanced sampling ({e_balanced_sample}). Falling back to random sampling.")
            subset_df = df_source.sample(n=actual_subset_size, random_state=42, replace=False)
    else: # Random sampling
        subset_df = df_source.sample(n=actual_subset_size, random_state=42, replace=False)

    if subset_df.empty:
        print(f"Could not create a subset (resulting DataFrame is empty). Source size: {len(df_source)}.")
        return

    os.makedirs(target_dir, exist_ok=True)
    
    new_metadata_list = []
    import shutil # Ensure shutil is imported
    
    print(f"Copying {len(subset_df)} files for the subset...")
    for _, row_series in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Copying subset files"):
        row = row_series.to_dict() # Work with dicts for easier modification
        
        original_filepath = str(row.get('filepath', ''))
        if not original_filepath or not os.path.exists(original_filepath):
            print(f"Warning: Source file '{original_filepath}' for row not found or path invalid. Skipping.")
            continue
            
        anomaly_folder_name = str(row.get('anomaly_type', 'unknown_anomaly'))
        
        target_anomaly_class_dir = os.path.join(target_dir, anomaly_folder_name)
        try:
            os.makedirs(target_anomaly_class_dir, exist_ok=True)
        except OSError as e:
            print(f"Error: Could not create directory {target_anomaly_class_dir}: {e}. Skipping file.")
            continue
            
        filename = os.path.basename(original_filepath)
        new_filepath = os.path.join(target_anomaly_class_dir, filename)
        
        try:
            shutil.copy2(original_filepath, new_filepath)
            row['filepath'] = new_filepath # Update filepath to the new location
            new_metadata_list.append(row)
        except Exception as e_copy_file:
            print(f"Error copying file {original_filepath} to {new_filepath}: {e_copy_file}")
            
    if new_metadata_list:
        df_new_metadata = pd.DataFrame(new_metadata_list)
        df_new_metadata.to_csv(os.path.join(target_dir, 'metadata.csv'), index=False)
        print(f"Successfully created subset with {len(df_new_metadata)} samples in '{os.path.abspath(target_dir)}'.")
    else:
        print(f"No files were successfully copied to the subset in '{target_dir}'. Check source files and permissions.")

# ================================
# COMMAND LINE INTERFACE
# ================================
if __name__ == "__main__":
    import sys
    
    # Simple CLI parser
    if len(sys.argv) > 1:
        command = sys.argv[1].lower() # Make command case-insensitive
        
        if command == "generate":
            main() # Calls the main generation and validation workflow
        elif command == "validate":
            if len(sys.argv) > 2:
                dataset_directory_to_validate = sys.argv[2]
                print(f"Validating dataset in: {os.path.abspath(dataset_directory_to_validate)}")
                validate_dataset(dataset_directory_to_validate)
            else:
                print("Usage: python your_script_name.py validate <dataset_directory_path>")
        elif command == "analyze":
            if len(sys.argv) > 2:
                dataset_directory_to_analyze = sys.argv[2]
                analyze_existing_dataset(dataset_directory_to_analyze)
            else:
                print("Usage: python your_script_name.py analyze <dataset_directory_path>")
        elif command == "subset":
            if len(sys.argv) > 4:
                source_dataset_dir = sys.argv[2]
                target_subset_dir = sys.argv[3]
                try:
                    num_subset_samples = int(sys.argv[4])
                    if num_subset_samples <= 0:
                         print("Error: subset_size must be a positive integer.")
                    else:
                         create_dataset_subset(source_dataset_dir, target_subset_dir, num_subset_samples)
                except ValueError:
                    print("Error: subset_size (argument 4) must be an integer.")
                    print("Usage: python your_script_name.py subset <source_dir> <target_dir> <subset_size_integer>")
            else:
                print("Usage: python your_script_name.py subset <source_dir_path> <target_dir_path> <subset_size_integer>")
        else:
            print(f"Unknown command: '{command}'.")
            print("Available commands: generate, validate, analyze, subset")
    else:
        # No command provided, run interactive main generation
        main()