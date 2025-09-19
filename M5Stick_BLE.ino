#include <M5StickCPlus.h>
#include <driver/i2s.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>
#include <ArduinoJson.h>

//BLE Defines
#define DEVICE_NAME         "m5-stack"
#define SERVICE_UUID        "c48aec0d-0962-43f5-b5a6-b2051d7bfca6"
#define CHARACTERISTIC_UUID "c48aec0d-0962-43f5-b5a6-b2051d7bfca6"

BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;
bool startAcquisitionFlag = false; // Global flag

//MIC defines
#define PIN_CLK     0
#define PIN_DATA    34
#define READ_LEN    (2 * 16384)
#define SAMPLE_RATE_HIGH 44100
#define SAMPLE_RATE_LOW 11025
#define GAIN_FACTOR 30
uint8_t BUFFER[READ_LEN] = {0};

//General global variables
uint16_t oldy[160];
int16_t *adcBuffer = NULL;
uint16_t sr = SAMPLE_RATE_LOW;

//BLE callbacks
//Callback on connection
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      M5.Lcd.println("BLE connect");
      deviceConnected = true;
    };

    void onDisconnect(BLEServer* pServer) {
      M5.Lcd.println("BLE disconnect");
      deviceConnected = false;
    };
};


//displays signal preview in display
//just shows the first 160 samples of aquisition
void showSignal() {
    int y;
    for (int n = 0; n < 160; n++) {
        y = adcBuffer[n] * GAIN_FACTOR;
        y = map(y, INT16_MIN, INT16_MAX, 10, 70);
        M5.Lcd.drawPixel(n, oldy[n], WHITE);
        M5.Lcd.drawPixel(n, y, BLACK);
        oldy[n] = y;
    }
}


//Callback to read and write BLE messages
class MyCallbacks: public BLECharacteristicCallbacks {
  void onRead(BLECharacteristic *pCharacteristic) {
    //M5.Lcd.println("Tx to RPI");
    pCharacteristic->setValue("Message from M5Stick");
  }
  
  void onWrite(BLECharacteristic *pCharacteristic) {
    //M5.Lcd.println("Rx from RPI");
    std::string value = pCharacteristic->getValue();
    M5.Lcd.println(value.c_str());

    // Set the flag when receiving the "a" message
    if (value == "a") {
    M5.Lcd.println("Start acquisition received!");
    startAcquisitionFlag = true; 
    }
    else if (value == "h") {
    M5.Lcd.println("Higher sampling rate!");
    sr = SAMPLE_RATE_HIGH;
    i2sInit(); 
    }
    else if (value == "l") {
    M5.Lcd.println("Lower sampling rate!");
    sr = SAMPLE_RATE_LOW;
    i2sInit(); 
    }
  }
};


//microphone inicialization
//microphone comunicates through I2C protocol
void i2sInit() {
  i2s_driver_uninstall(I2S_NUM_0);
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM),
        .sample_rate = sr,
        .bits_per_sample =
            I2S_BITS_PER_SAMPLE_16BIT,  // is fixed at 12bit, stereo, MSB
        .channel_format = I2S_CHANNEL_FMT_ALL_RIGHT,
#if ESP_IDF_VERSION > ESP_IDF_VERSION_VAL(4, 1, 0)
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
#else
        .communication_format = I2S_COMM_FORMAT_I2S,
#endif
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count    = 2,
        .dma_buf_len      = 128,
    };

    i2s_pin_config_t pin_config;

#if (ESP_IDF_VERSION > ESP_IDF_VERSION_VAL(4, 3, 0))
    pin_config.mck_io_num = I2S_PIN_NO_CHANGE;
#endif

    pin_config.bck_io_num   = I2S_PIN_NO_CHANGE;
    pin_config.ws_io_num    = PIN_CLK;
    pin_config.data_out_num = I2S_PIN_NO_CHANGE;
    pin_config.data_in_num  = PIN_DATA;

    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);
    i2s_set_clk(I2S_NUM_0, sr, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_MONO);
}

//initial setup of: display, BLE and microphone
void setup() {
    M5.begin();

    //set display
    M5.Lcd.setRotation(3);
    M5.Lcd.fillScreen(WHITE);
    M5.Lcd.setTextColor(BLACK, WHITE);
    M5.Lcd.println("mic aquisiiton");

    //BLE start
    //M5.Lcd.println("BLE start.");
    BLEDevice::init(DEVICE_NAME);
    BLEServer *pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());
    BLEService *pService = pServer->createService(SERVICE_UUID);
    pCharacteristic = pService->createCharacteristic(
                                           CHARACTERISTIC_UUID,
                                           BLECharacteristic::PROPERTY_READ |
                                           BLECharacteristic::PROPERTY_WRITE |
                                           BLECharacteristic::PROPERTY_NOTIFY |
                                           BLECharacteristic::PROPERTY_INDICATE
                                         );
    pCharacteristic->setCallbacks(new MyCallbacks());
    pCharacteristic->addDescriptor(new BLE2902());
  
    pService->start();
    BLEAdvertising *pAdvertising = pServer->getAdvertising();
    pAdvertising->start();
    //M5.Lcd.println("BLE running.");

    //MIC comms iniciated
    i2sInit();
}

void loop() {
    int y;
    size_t bytesread;

    M5.update();  // Read the press state of the key.

    if(startAcquisitionFlag) {
        //M5.Lcd.println("A pressed");
        //Mic aquisition
        i2s_read(I2S_NUM_0, (char *)BUFFER, READ_LEN, &bytesread,
                 (100 / portTICK_RATE_MS));
        adcBuffer = (int16_t *)BUFFER;
        showSignal();

        //Serial.println(READ_LEN);

        //send data through BLE
        StaticJsonDocument<2048> doc;
        JsonArray data = doc.createNestedArray("data");
        char str3[1024]; 

        //divide data in packages of 100 samples to not overrun BLE buffer
        for (int m = 0; m < int(READ_LEN/200); m++) {    //only right microphone has useful info

          data.clear();
          for (int n = 0; n < 100; n++) {
              //M5.Lcd.println(adcBuffer[m*100+n]);
              y = adcBuffer[m*100+n] * GAIN_FACTOR;
              y = map(y, INT16_MIN, INT16_MAX, -100, 100);
              data.add(y);
              Serial.println(y); //to sample data with arduino IDE plot tool
          }

          serializeJson(doc, str3);
          if (deviceConnected == true) {
            //send data
            pCharacteristic->setValue(str3); 
            pCharacteristic->notify();  
          }
        }
        // Reset the flag after the acquisition is done
        startAcquisitionFlag = false;
          
    }
}
