# Musical Instrument Recognition with RPI and M5Stick, using Python and C

A machine learning model was trained to recognize musical instruments (guitar, piano, ukulele, and voice) using extracted audio signal features. The system integrates a PC, a Raspberry Pi, and an M5Stick: the PC, running a Streamlit interface designed for children, communicates with the Raspberry Pi via MQTT to request audio readings; the Raspberry Pi collects the audio from the M5Stick over BLE, performs the classification using the trained ML model, and returns the prediction to the PC via MQTT for real-time display.


![Alt text](images/instrument_recognition.jpg)
