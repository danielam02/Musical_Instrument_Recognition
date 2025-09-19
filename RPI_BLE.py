
# IMPORTING

import asyncio
import uuid
from aioconsole import ainput

from bleak import BleakScanner, BleakClient
import json
import os



#%%
# GLOBAL VARIABLES

DEVICE_NAME = "m5-stack"
SERVICE_UUID = uuid.UUID("c48aec0d-0962-43f5-b5a6-b2051d7bfca6")
CHAR_UUID = uuid.UUID("c48aec0d-0962-43f5-b5a6-b2051d7bfca6")

START_FILE = "a.txt"
DATA_FILE = "dados.txt"
SRH_FILE = "h.txt"
SRL_FILE = "l.txt"



#%%
# BLE COMMUNICATION

async def run_ble(loop):
    '''
    Performs BLE operations including:
    - Connecting to a device with the specified name DEVICE_NAME
    - Reading and writing GATT characteristics from/to the device
    - Subscribing to notifications from the device
    - Checking for files START_FILE, SRH_FILE and SRL_FILE and sending the
      corresponding commands to the device
    - Writing received data to DATA_FILE
    
    Parameters:
    loop: The asyncio event loop to be used for asynchronous operations.
    '''
    
    print("Searching devices...")
    devices = await BleakScanner.discover()

    device = list(filter(lambda d: d.name == DEVICE_NAME, devices))
    if len(device) == 0:
        raise RuntimeError(f"Failed to find a device name '{DEVICE_NAME}'")

    address = device[0].address
    print(f"Connecting to the device... (address: {address})")
    
    flag = False
    while flag == False: 
        try:
            async with BleakClient(address, loop=loop) as client:
                flag = True
                print("Done")            
                print("Message from the device...")
                
                value = await client.read_gatt_char(CHAR_UUID)
                print(value.decode())
        
                print("Sending message to the device...")
                message = bytearray(b"RPI ready")
                await client.write_gatt_char(CHAR_UUID, message, True)
                
                # Receive and process the BLE data from the m5-stick
                def callback(sender, data):
                    
                    print(str(data, 'utf-8'))
                    data=json.loads(str(data, 'utf-8'))
                    print("Length Data received", len(data["data"]))
                    
                    # Save each number in a new line
                    with open(DATA_FILE, "a") as file:
                        for value in data["data"]:
                            file.write(f"{value}\n") 
                        
                print("Subscribing to characteristic changes...")
                await client.start_notify(CHAR_UUID, callback)
                
                while True:
                            
                    # Check if the START_FILE exists to send aquisition command
                    if os.path.exists(START_FILE):
                        
                        if os.path.exists(DATA_FILE):
                            print("Removing existing data file.")
                            # Start the data collection from beginning
                            os.remove(DATA_FILE) 
                        
                        print("Sending 'a' to M5Stick.")
                        message = bytearray(b"a")
                        os.remove(START_FILE) 
                        await client.write_gatt_char(CHAR_UUID, message, True)
                    
                    # Check for Sampling Rate related files
                    if os.path.exists(SRH_FILE):
                        print("Sending 'h' to M5Stick.")
                        message = bytearray(b"h")
                        await client.write_gatt_char(CHAR_UUID, message, True)
                        os.remove(SRH_FILE)
                        
                    if os.path.exists(SRL_FILE):
                        print("Sending 'l' to M5Stick.")
                        message = bytearray(b"l")
                        await client.write_gatt_char(CHAR_UUID, message, True)
                        os.remove(SRL_FILE)
                        
                    await asyncio.sleep(1)
                
                # Waits for an input from user to end process
                result = await ainput('Press any key to exit')
                print("Disconnecting from device")
        
        except Exception as e:
            print(f"Retrying... Error: {e}")
      
      

#%%
# MAIN

def main():
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_ble(loop))
    
    
if __name__ == "__main__":
    main()    
