# IMPORTING AND LOADING

import streamlit as st

from streamlit.runtime.scriptrunner import add_script_run_ctx
import threading as th
import paho.mqtt.client as mqtt
from streamlit_autorefresh import st_autorefresh
import json
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd


#%%
# GLOBAL VARIABLES 
sr_low = 11025
sr_high = 44100
librosa_sr = 48000
wait_time = 8
autorefresh_interval = 2000


#%%
# MQTT thread function

def MQTT_TH(client):
    '''
    Sets up an MQTT client, connects to a broker, and processes incoming 
    messages. It performs the following actions:
    - Establishes a connection to the MQTT broker 
    - Subscribes to topics based on the provided topic
    - Defines callbacks to handle incoming messages
    '''

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(client, userdata, flags, rc):
        '''
        Callback executed when the client connects to the MQTT broker
        Subscribes to the topic and prints the result code of the connection
        '''
        
        print("Connected with result code "+str(rc))
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        
        client.subscribe(st.session_state['MyData']['TopicSub'])
    
    # The callback for when a PUBLISH message is received from the server.
    def on_message(client, userdata, msg):
        '''
        Callback executed when a message is received
        Processes the received message and performs actions based on the 
        command content 
        '''
        
        print(msg.topic + " " + str(msg.payload.decode()))
        
        # Parse incoming MQTT message and update session state
        if msg.topic == st.session_state['MyData']['TopicSub'].rsplit('/', 1)[0] + '/prediction':
            st.session_state['MyData']['Prediction'] = msg.payload.decode()
            
        elif msg.topic == st.session_state['MyData']['TopicSub'].rsplit('/', 1)[0] + '/graph':
            graph_values = json.loads(msg.payload.decode())
            st.session_state['MyData']['GraphData'] = graph_values
            
            print(len(graph_values))
            
    print('Initializing MQTT')
    client.on_connect = on_connect
    client.on_message = on_message
    st.session_state['MyData']['Run'] = True
    client.connect(st.session_state['MyData']['Broker'], 1883, 60)
    client.loop_forever()
    print('MQTT link ended')
    st.session_state['MyData']['Run'] = False



#%%
# SESSION STATE
# Stores states of variables between page refresh

if 'MyData' not in st.session_state:
    st.session_state['MyData'] = {
        'Run': False,
        'Broker': '192.168.1.98', 
        'TopicPub': 'AAI/RD/cmd',
        'TopicSub': 'AAI/RD/reply/#',
        'Prediction': None,
        'GraphData': None,
        'Sampling Rate': sr_low,
        'ButtonSource': None
    }
    
# MQTT session information
if 'mqttThread' not in st.session_state:
    #open client MQTT connection in an independent thread
    print('session state')
    st.session_state.mqttClient = mqtt.Client()
    st.session_state.mqttThread = th.Thread(target=MQTT_TH, args=[st.session_state.mqttClient]) 
    add_script_run_ctx(st.session_state.mqttThread) 


#%%
# Page design starts here!

# Main UI functions
def display_intro():
    """
    Creates and displays the welcome page of the application with formatted HTML content.
    
    No parameters or return values.
    """
    st.markdown("<br>", unsafe_allow_html=True)
    st.write(
        """
        <style>
            /* Importing Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&family=Lato:wght@400;700&family=Roboto+Slab:wght@700&display=swap');
        </style>
        
        <div style="text-align: center; font-size: large; color: black;">
        <!-- Main Title as h2 with the same size as the subtitle -->
        <h2 style="font-size: 1.3em; font-weight: bold; font-family: 'Poppins', sans-serif; color: #649cbd;">Welcome to the Magical Instrument Identifier! 🎸✨</h2>
        <p style="font-size: 1em;">This app lets you instantly identify the instrument you're playing. Ready to discover the magic?</p>
        
        <!-- Subtitle: "Here's what you can do" with Poppins -->
        <h2 style="font-size: 1.3em; font-weight: bold; margin-top: 30px; font-family: 'Poppins', sans-serif; color: #649cbd;">Here's what you can do:</h2>
        
        <p style="font-size: 1em;"><strong>🎶 <em>Record the Instrument</em></strong>: Start recording, and let the magic happen as we identify your instrument!</p>
        <p style="font-size: 1em;"><strong>🥁 <em>See Results</em></strong>: Instantly view the instrument's identity along with its sound wave!</p>
        <p style="font-size: 1em;"><strong>🛠️ <em>Configure Connections</em></strong>: Set up and connect to your MQTT broker to get predictions with real-time graphs.</p>
        
        <!-- Subtitle: "Ready to start?" with Poppins -->
        <h3 style="font-size: 1.3em; font-weight: bold; margin-top: 30px; font-family: 'Poppins', sans-serif; color: #649cbd;">Ready to start your musical journey? Let's go! 🎉</h3>
        
        </div>
        """,
        unsafe_allow_html=True
    )

def display_configurations():
    """
    Renders the MQTT configuration interface and sampling rate controls.
    Displays input fields for MQTT broker address and topics, connection status buttons,
    and sampling rate toggle buttons. Updates the session state when configurations change.
    
    No parameters or return values.
    """
    
    st.markdown(
    """
    <style>
        /* Importing Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&family=Lato:wght@400;700&family=Roboto+Slab:wght@700&display=swap');

        h1 {
            font-family: 'Poppins', sans-serif;
            font-size: 1.3rem; /* Make header smaller */
        }
    
    </style>
    """, unsafe_allow_html=True
    )
    
    st.markdown('<div class="centered"><h1>Configure MQTT Settings 🛠️</h1></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # MQTT Broker and Topics Input
    st.text_input('MQTT Broker:', value='192.168.1.98', key='Broker')
    st.text_input('Topic to Send Start Command:', value='AAI/RD/cmd', key='TopicPub')
    st.text_input('Topic to Receive Results:', value='AAI/RD/reply/#', key='TopicSub')

    # MQTT Connection Buttons (Connect or Disconnect)
    if st.session_state.get('MyData', {}).get('Run', False):
        if st.button('MQTT Disconnect 💔'):
            disconnect_mqtt()
    else:
        if st.button('MQTT Connect 💡'):
            connect_mqtt()

    # Sampling Rate Change Buttons
    if st.session_state['MyData']['Sampling Rate'] == sr_low:
        if st.button('Change Sampling Rate to 44100 Hz'):
            change_sampling_rate(sr_high, "h")
    else:
        if st.button('Change Sampling Rate to 11025 Hz'):
            change_sampling_rate(sr_low, "l")


def display_instrument_test():
    """
    Creates the main instrument testing interface with recording and playback controls.
    Displays:
    - A record button for capturing live audio
    - Prediction results with instrument images
    - Waveform visualization
    - Pre-recorded instrument test buttons
    
    No parameters or return values.
    """
    st.markdown(
        """
        <style>
            /* Importing Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&family=Lato:wght@400;700&family=Roboto+Slab:wght@700&display=swap');
    
            /* For the main title (h1) */
            h1 {
                font-family: 'Poppins', sans-serif;
                font-size: 1.3rem; /* Larger for the main title */
                text-align: center;
            }
    
            /* For sub-headers (h2) */
            h2 {
                font-family: 'Lato', sans-serif;
                font-size: 1rem;
                font-weight: 700;
                color: #333333; /* Dark gray for readability */
                text-align: center;
            }
    
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown('<div class="centered"><h1>What Instrument is This?</h1></div>', unsafe_allow_html=True)
    
    # Record Button Logic
    if st.button("Record 🎯"):   
        st.session_state['MyData']['ButtonSource'] = 'Record'
        st.session_state['MyData']['Prediction'] = None
        st.session_state['MyData']['GraphData'] = None
        record_instrument()
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create two columns for side-by-side layout
    col1, col2 = st.columns(2)
    
    # Prediction Results
    with col1:
        st.markdown('<div class="centered"><h2>Instrument Identification 🔮</h2></div>', unsafe_allow_html=True)
        prediction = st.session_state['MyData'].get('Prediction', None)
        display_prediction(prediction)
    
    # Sound Wave Graph
    with col2:
        st.markdown('<div class="centered"><h2>Sound Wave 📊</h2></div>', unsafe_allow_html=True)
        graph_data = st.session_state['MyData'].get('GraphData', None)
        
        # Use a different SR based on the button source
        button_source = st.session_state.get('MyData', {}).get('ButtonSource', '')
        if button_source == 'Record':
            display_waveform(graph_data, st.session_state['MyData']['Sampling Rate'])
        elif button_source == 'Instrument':
            display_waveform(graph_data)  # Use sampling rate from librosa (48000 Hz)
        else:
            display_waveform()  # Display empty graph


    st.markdown("<br>", unsafe_allow_html=True)

    # Test with Pre-Recorded Audio
    st.markdown('<div class="centered"><h2>Test with Pre-Recorded Audio 🔄</h2></div>', unsafe_allow_html=True)

    # Instrument Buttons
    instrument_commands = {
        "Guitar 🎸": "g",
        "Ukulele 🎻": "u",
        "Piano 🎹": "p",
        "Voice 🎤": "v",
        "Flute 🎼": "f",
        "Noise 🔊": "n",
    }

    # Create a single row with 6 columns
    cols = st.columns(6)
    
    for i, (label, command) in enumerate(instrument_commands.items()):
        with cols[i % 6]:  
            if st.button(label):
                st.session_state['MyData']['Prediction'] = None
                st.session_state['MyData']['GraphData'] = None
                st.session_state['MyData']['ButtonSource'] = 'Instrument'
                if st.session_state.get('MyData', {}).get('Run', True):
                    st.session_state.mqttClient.publish(st.session_state['MyData']['TopicPub'], command)
                else:
                    st.error("❌ Not connected. Ask your teacher!")



#%% 
# Display prediction and graph functions

def display_prediction(prediction: str):
    """
    Renders the prediction results with corresponding instrument images and formatted text.
    
    Parameters:
        prediction (str): The predicted instrument name or 'rejection' for noise
    
    Displays:
    - An image of the predicted instrument
    - Custom formatted header with emoji
    - Loading image when no prediction is available
    
    Updates the display based on the prediction result from the model.
    """
    #placeholder_image = Image.new('RGB', (400, 400), color='white') 
    placeholder_image  = Image.open('loading.jpg').resize((200,200))
    
    images = {
        "guitar": "guitar.jpg",
        "ukulele": "ukulele.jpg",
        "piano": "piano.jpg",
        "voice": "voice.jpg",
        "flute": "flute.jpg",
        "rejection": "confusion.jpg"
    }

    image = placeholder_image
    
    if prediction:
        image = Image.open(images.get(prediction.lower(), "confusion.jpg"))
        image = image.resize((400, 400))
        st.image(image)
    
        if prediction.lower() == "guitar":
            st.markdown('<div class="centered"><h3>🎉 Guitar 🎸</h3></div>', unsafe_allow_html=True)
        elif prediction.lower() == "ukulele":
            st.markdown('<div class="centered"><h3>🎉 Ukulele 🎶</h3></div>', unsafe_allow_html=True)
        elif prediction.lower() == "piano":
            st.markdown('<div class="centered"><h3>🎉 Piano 🎹</h3></div>', unsafe_allow_html=True)
        elif prediction.lower() == "voice":
            st.markdown('<div class="centered"><h3>🎉 Voice 🎤</h3></div>', unsafe_allow_html=True)
        elif prediction.lower() == "flute":
            st.markdown('<div class="centered"><h3>🎉 Flute 🎼</h3></div>', unsafe_allow_html=True)
        elif prediction.lower() == "rejection":
            st.markdown('<div class="centered"><h3>🎉 Noise 🔊</h3></div>', unsafe_allow_html=True)
    
    else:
        image = image.resize((400, 400))
        st.image(image)

        
def display_waveform(graph_data=None, sampling_rate=librosa_sr):
    """
    Creates and displays a waveform plot of the audio signal using Plotly.
    
    Parameters:
        graph_data (list, optional): Audio signal amplitude values. If None, displays empty plot
        sampling_rate (int, optional): Sampling rate of the audio signal in Hz. Defaults to librosa's rate
    
    Creates a 400x400 line plot showing:
    - Time on x-axis in seconds
    - Signal amplitude on y-axis
    """
    if not graph_data: # Create an empty plot (no data)
        fig = px.line(
            x=[0], y=[0], title="",
            labels={"Time (s)": "Time (s)", "Amplitude": "Amplitude"}
        )
    else:
        num_samples = len(graph_data)
        duration = num_samples / sampling_rate
        x = np.linspace(0, duration, num_samples)
        df = pd.DataFrame({"Time (s)": x, "Amplitude": graph_data})

        fig = px.line(
            df, x="Time (s)", y="Amplitude", title="",
            labels={"Time (s)": "Time (s)", "Amplitude": "Amplitude"}
        )

    fig.update_layout(
        font=dict(size=12),
        margin=dict(l=20, r=20, t=20, b=20),
        width=400,
        height=400
    )
    st.plotly_chart(fig, use_container_width=False)



#%%
# Utility functions for MQTT connection
def connect_mqtt():
    """
    Establishes MQTT connection and initializes background thread for message handling.
    Creates new thread if none exists, starts the MQTT client connection,
    and updates the session state to reflect connection status.
    Displays success message upon successful connection.
    
    No parameters or return values.
    """
    if not st.session_state.mqttThread.is_alive():
        st.session_state.mqttThread = th.Thread(target=MQTT_TH, args=[st.session_state.mqttClient])
        add_script_run_ctx(st.session_state.mqttThread)
    st.session_state.mqttThread.start()
    st.success("Connected MQTT")


def disconnect_mqtt():
    """
    Terminates active MQTT connection and updates connection status.
    Disconnects the MQTT client, updates the session state Run flag to False,
    and displays a success message confirming disconnection.
    
    No parameters or return values.
    """
    st.session_state.mqttClient.disconnect()
    st.session_state['MyData']['Run'] = False
    st.success("Disconnected MQTT")



#%%
# Utility functions for prediction and sampling rate handling

def change_sampling_rate(new_rate: int, command: str):
    """
    Updates the audio sampling rate and sends command to update backend processing.
    
    Parameters:
        new_rate (int): New sampling rate in Hz to be set
        command (str): Single character command ('h' for high, 'l' for low) to send via MQTT
    
    Updates session state with new sampling rate and notifies user of the change.
    """
    st.session_state['MyData']['Sampling Rate'] = new_rate
    st.session_state.mqttClient.publish(st.session_state['MyData']['TopicPub'], command)
    st.success(f"Changed Sampling Rate to {new_rate} Hz!")


def record_instrument():
    """
    Initiates audio recording process and handles connection status feedback.
    Publishes recording command 'a' via MQTT if connected, displays 10-second wait message,
    or shows error if MQTT connection is not active.
    Requires active MQTT connection to function.
    
    No parameters or return values.
    """
    if st.session_state.get('MyData', {}).get('Run', True):
        st.session_state.mqttClient.publish(st.session_state['MyData']['TopicPub'], "a")
        st.success("Attempting to record! Wait 10 seconds ⏳")
    else:
        st.error("❌ Not connected. Ask your teacher!")
 
    

#%%
def main():
    
    # Set up the page configuration
    st.set_page_config(page_title="Instrument Recognition")
    st_autorefresh(interval=autorefresh_interval, key="fizzbuzzcounter")
    
    # Page Title and Custom Styles
    st.markdown(
        """
        <style>
            /* Importing Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&family=Lato:wght@400;700&family=Roboto+Slab:wght@700&display=swap');
            
            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background: linear-gradient(to bottom, #9dbded, #40516b);
                color: #fcfcfc;
                font-family: 'Poppins', sans-serif;
            }                           
            
            /* Input fields in sidebar */
            [data-testid="stSidebar"] .stTextInput input {
                border-radius: 5px;
                border: 1px solid #ccc;
                padding: 6px;
            }
            
            /* Align sidebar elements */
            [data-testid="stSidebar"] > div {
                padding: 10px;
            }
            
            /* Title Styling */
            .title {
                text-align: center;
                color: #3c85b0; /* Blue color */
                font-size: 2rem; /* Adjust size to make it bigger */
                font-family: 'Poppins', sans-serif; /* Bold and modern font */
            }

            /* Centering Elements */
            .centered {
                display: flex;
                justify-content: center;
                align-items: center;
                text-align: center;
            }

            /* Button Styling */
            .stButton > button {
                background-color: #5676a8;
                color: white;
                border: none;
                margin: 0 auto;
                display: block;
                border-radius: 5px;
                transition: background-color 0.3s;
            }
            .stButton > button:hover {
                background-color: #acc3e8;
            }

            /* Plotly Chart Centering */
            .stPlotlyChart {
                margin: 0 auto;
            }
            body {
                background-color: #00ff00;
            }
        </style>
        <h1 class="title">🎵 Instrument Recognition 🎵</h1>
        """,
        unsafe_allow_html=True,
    )
    
    st.sidebar.markdown("<h4 style='font-size: 24px; font-weight: bold; color: #fcfcfc;'>Welcome!</h4>", unsafe_allow_html=True)
    menu = st.sidebar.radio("Go to:", ["Introduction", "Configurations", "Instrument Identification"])
    
    # Display different menus based on the selected option
    if menu == "Introduction":
        display_intro()
        st.sidebar.write("Learn about the app and how it works.")
    elif menu == "Configurations":
        display_configurations()
        st.sidebar.write("Set up your MQTT connection and preferences.")
    elif menu == "Instrument Identification":
        display_instrument_test()
        st.sidebar.write("Test your instrument recognition.")

#%%
# Run the app
if __name__ == "__main__":
    main()