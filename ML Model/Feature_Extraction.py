# IMPORTING

import os
import librosa
import librosa.display
import numpy as np
import csv
import subprocess


#%%
# GLOBAL VARIABLES

# Directory with all training and testing files
audio_dir = "audios" 

# Frequency percentage of max at which to extract the power
freq_power = 0.8

# CSV files for storing features
csv_file = 'features_training.csv'
csv_file_test = 'features_test.csv'


#%%
# FUNCTIONS

def get_class_name(filename):
    '''
    Returns the class an audio file belongs to (guitar, ukulele, piano or no relation)
    
    Parameters
    ----------
    filename : str - Name of the audio file

    Returns
    -------
    str - class label

    '''
    if filename.startswith("guitarra"):
        return "guitar"
    elif filename.startswith("ukulele"):
        return "ukulele"
    elif filename.startswith("piano"):
        return "piano"
    elif filename.startswith("voz"):
        return "voice"
    elif filename.startswith("flauta"):
        return "flute"
    elif filename.startswith("rejeicao"):
        return "rejection"
    else:
        print("Error in class assignement")
    
    

def normalize_audio(data):
    '''
    Normalizes audio signal to be in the [-1,1] range.
    
    Parameters
    ----------
    data : np.ndarray - Audio time series.

    Returns
    -------
    data : np.ndarray - Normalized audio time series.

    '''
    max_abs_value = np.max(np.abs(data))
    if max_abs_value > 0: # To make sure no divisions by 0
        data = data / max_abs_value
    return data

    

def extract_features(audio_file):
    '''
    Extracts features from an audio file

    Parameters
    ----------
    audio_file : str - Audio file path.

    Returns
    -------
    features: list - List of exctracted features

    '''
    data, sr = librosa.load(audio_file, sr=None)
    print(sr)
    data = normalize_audio(data)
    
    features = []

    # Feature 1 - Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=data)
    zc_rate = len(zcr[0])
    features.append(zc_rate)

    # Feature 2 - Maximum Power at a Given Frequency
    mel = librosa.feature.melspectrogram(y=data, sr=sr)
    S_dB = librosa.power_to_db(mel, ref=np.max)
    max_pw = np.max(S_dB[int(freq_power * len(S_dB))])
    features.append(max_pw)

    # Feature 3 - Mean Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=data, sr=sr)
    bw_mean = np.mean(spec_bw)
    features.append(bw_mean)
    
    # Feature 4 - Mean of MFCCs
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)  
    features.extend(mfcc_mean)  
    
    # Feature 5 - Dominant Frequency 
    dominant_bin = np.argmax(np.mean(S_dB, axis=1))
    dominant_freq = librosa.mel_frequencies(n_mels=mel.shape[0])[dominant_bin]
    features.append(dominant_freq)
    
    # Feature 6 - Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=data, sr=sr)
    centroid_mean = np.mean(centroid)
    features.append(centroid_mean)

    # Feature 7 - Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=data)
    flatness_mean = np.mean(flatness)
    features.append(flatness_mean)

    return features


#%%
# MAIN 
''' 
Feature extraction for training files into features_training.csv file and
                   for testing files into features_test.csv file
'''

# Convert .m4a files to .wav
for filename in os.listdir(audio_dir):
    if filename.endswith(".m4a"): 
        input_file = os.path.join(audio_dir, filename)
        output_file = os.path.join(audio_dir, os.path.splitext(filename)[0] + ".wav")
        subprocess.run(["ffmpeg", "-i", input_file, output_file])
         
            
# Process .wav files
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):
        file_path = os.path.join(audio_dir, filename)
        class_name = get_class_name(filename)
        features = extract_features(file_path)
        
        # Determine the appropriate CSV file
        target_csv = csv_file_test if filename.endswith("test.wav") else csv_file
        
        # Append features to the selected CSV file
        with open(target_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([class_name] + features)
            


