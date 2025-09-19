
# IMPORTING AND LOADING

import paho.mqtt.client as mqtt
import threading as th
import json
import time
import numpy as np
import os
import joblib
import librosa
import librosa.display
import random


# Load trained model
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl') 



#%%
# GLOBAL VARIABLES

SAMPLING_RATE_L = 11025
SAMPLING_RATE_H = 44100
sr = SAMPLING_RATE_L #default

FREQ_POWER = 0.8

START_FILE = "a.txt"
DATA_FILE = "dados.txt"
TEST_FOLDER = "test_dataset"



#%%
# MQTT FUNCTIONS AND SETUP

def mqtt_setup(topic):
    '''
    Sets up an MQTT client, connects to a broker, and processes incoming 
    messages. It performs the following actions:
    - Establishes a connection to the MQTT broker 
    - Subscribes to topics based on the provided topic
    - Defines callbacks to handle incoming messages
    - Based on the received command, it performs different actions, such as:
        - Creating files and publishing predictions and graphs to related 
          topics
        - Changing the sampling rate (sr)
    - Runs the MQTT communication in a separate thread
    
    Parameters
    ----------
    topic: str - The MQTT topic to which the client will subscribe
    
    Returns
    ----------
    client: The configured MQTT client
    '''
    
    def on_connect(client, userdata, flags, rc):
        '''
        Callback executed when the client connects to the MQTT broker
        Subscribes to the topic and prints the result code of the connection
        '''
        
        print("\tMQTT connected with result code "+str(rc))
        client.subscribe(topic+'/#')
        print(f"\tSubscribed to topic {topic}"+str(rc))
        
    def on_message(client, userdata, msg):
        '''
        Callback executed when a message is received
        Processes the received message and performs actions based on the 
        command content 
        '''
        
        print("   --> " + msg.topic + " "+ str(msg.payload))
        
        def handle_message():
            '''
            Processes the received message, performing different actions based 
            on the command. It can:
            - Create text files based on the command 
            - Publish predictions and graphs to response topic and subtopics
            '''
            
            # Send a reply back to the sender
            msg_reply = 'RPI received: ' + str(msg.payload)
            client.publish(topic[:len(topic)-4] + '/reply', msg_reply)
            
            decoded_msg = msg.payload.decode()
            
            global sr
            
            if decoded_msg == "a":
                
                print("Received 'a' command. Creating 'a.txt'.")
                open(START_FILE, 'w').close()
                
                time.sleep(10) # wait for BLE to receive messages... adjust if necessary
                print("Time elapsed.")
                prediction, data = process_data(DATA_FILE)
                client.publish(topic[:len(topic)-4] + '/reply/prediction', prediction)
                client.publish(topic[:len(topic)-4] + '/reply/graph', json.dumps(data))
                print("Published")
                
            elif decoded_msg == "h":
                
                print("Received 'h' command. Creating 'h.txt'.")
                SR_FILE = "h.txt"
                open(SR_FILE, 'w').close()
                sr = SAMPLING_RATE_H
                
            elif decoded_msg == "l":
                
                print("Received 'l' command. Creating 'l.txt'.")
                SR_FILE = "l.txt"
                open(SR_FILE, 'w').close()
                sr = SAMPLING_RATE_H
            
            elif decoded_msg in ("g", "u", "p", "f", "v", "n"):
    
                print(f"Received '{decoded_msg}' command. Retrieving test file")
                prediction, data = process_test(decoded_msg)   
                client.publish(topic[:len(topic)-4] + '/reply/prediction', prediction)
                client.publish(topic[:len(topic)-4] + '/reply/graph', json.dumps(data))
                print("Published")
                
            else:
                print("Ignoring invalid command.")
            
        # Start a thread to handle the message
        message_thread = th.Thread(target=handle_message)
        message_thread.start()
    
    broker = '192.168.1.98' 
    print('\tMQTT starting')
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, 1883, 60)
    
    # Start the MQTT loop in a separate thread
    loop_thread = th.Thread(target=client.loop_forever)
    loop_thread.start()
    
    return client



#%%
# PREDICTION FUNCTIONS

# Function to read and process data from 'dados.txt'
def read_data(file):
    '''
    Reads and processes data from a text file, returning a numpy array.

    Parameters
    ----------
    file : str
        Path to the text file containing numeric data.

    Returns
    -------
    data_array : np.ndarray or None
        Processed data as a numpy array. Returns None if the file does not exist.
    '''
    
    if not os.path.exists(file):
        return None

    with open(file, 'r') as f:
        data = [float(line.strip()) for line in f.readlines()]

    data_array = np.array(data)
    
    return data_array


def normalize_audio(data):
    '''
    Normalizes audio signal to be in the [-1,1] range.
    
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


# Function to extract features from audio data using librosa
def extract_audio_features(data, sr):
    '''
    Extracts audio features from the input data using the librosa library.

    Parameters
    ----------
    data : np.ndarray - Raw audio time series.
    sr : int - Sampling rate of the audio.

    Returns
    -------
    features : list - Extracted features.
    '''
    
    data = normalize_audio(data)
    
    features = []
    
    # 'data' is the raw audio signal 
    # Feature extraction with librosa
    zcr = librosa.feature.zero_crossing_rate(y=data)
    mel = librosa.feature.melspectrogram(y=data, sr=sr)  

    # FEATURE 1 - ZERO CROSSING RATE
    zc_rate = len(zcr[0])
    features.append(zc_rate)

    # FEATURE 2 - MAXIMUM POWER AT A GIVEN FREQUENCY
    S_dB = librosa.power_to_db(mel, ref=np.max)
    max_pw = np.max(S_dB[int(FREQ_POWER * len(S_dB))])
    features.append(max_pw)

    # FEATURE 3 - MEAN SPECTRAL BANDWIDTH
    spec_bw = librosa.feature.spectral_bandwidth(y=data, sr=sr)
    bw_mean = np.mean(spec_bw)
    features.append(bw_mean)
    
    # Feature 4 - Mean of MFCCs
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)  
    features.extend(mfcc_mean)  
    
    # Feature 5 - Dominant Frequency Range
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


# Function to make a prediction
def make_prediction(features):
    '''
    Makes a prediction based on the extracted features using a pre-trained model.
 
    Parameters
    ----------
    features : list - Extracted audio features.
 
    Returns
    -------
    predicted_class_label : str - Label of the predicted class.
    '''
    
    if features is not None:
    
        # Reshape the features to match model input
        features_array = np.array(features).reshape(1, -1)
            
        # Make a prediction
        prediction = model.predict(features_array)

        # Decode the label
        predicted_class_label = label_encoder.inverse_transform(prediction)
        
    return predicted_class_label[0]


def process_data(file):
    '''
    Reads data from a file, extracts features, and predicts the class.

    Parameters
    ----------
    file : str - Path to the file containing raw data.

    Returns
    -------
    prediction : str - Predicted class label.
    data : list - Processed raw data as a list.
    '''
    
    while True:

        data = read_data(file)
        
        if data is not None:

            features = extract_audio_features(data, sr)
            prediction = make_prediction(features)
            
            data = data.tolist()
            
            return prediction, data


def process_test(command):
    '''
    Selects a random test file based on the input command, extracts features, 
    and predicts the class.

    Parameters
    ----------
    command : str - Command character indicating the instrument type 
                    ('g', 'u', 'p', 'f', 'v', 'n').

    Returns
    -------
    prediction : str - Predicted class label.
    data : list - Processed raw data as a list.
    '''
    
    command_map = {
        'g': 'guitarra',
        'u': 'ukulele',
        'p': 'piano',
        'f': 'flauta',
        'v': 'voz',
        'n': 'rejeicao'
    }
    
    prefix = command_map[command]
    matching_files = [f for f in os.listdir(TEST_FOLDER) if f.startswith(prefix) and f.endswith(".wav")]
    
    selected_file = random.choice(matching_files)
    file_path = os.path.join(TEST_FOLDER, selected_file)
    
    data, sr = librosa.load(file_path, sr=None)
    data = data.tolist()
    
    features = extract_audio_features(data, sr)
    prediction = make_prediction(features)
    
    return prediction, data
    
    

#%%
# MAIN

def main():
    
    sub_topic = 'AAI/RD/cmd'
    mqtt_client = mqtt_setup(sub_topic)
    
    print("\nMessages received: ")
    try:
        # Keep the main thread running, performing no additional tasks
        while True:
            time.sleep(1)  # Prevent high CPU usage
    except KeyboardInterrupt:
        print("\tExiting program...")
    finally:
        mqtt_client.disconnect()  # Ensure proper disconnection
        print("\tMQTT disconnected.")
    
    
if __name__ == "__main__":
    main()    
    
    
    
    
    
    
    
