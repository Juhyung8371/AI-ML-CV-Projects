
#include <SoftwareSerial.h>  // bluetooth

// Keyboard buffer size
const uint8_t BUFFER_SIZE = 8;

//Keyboard report buffer
uint8_t buf[BUFFER_SIZE] = { 0 };

// Bluetooth pins
const uint8_t PIN_INPUT_RX = 10;
const uint8_t PIN_INPUT_TX = 11;

// the pin for turning this device on and off
const uint8_t PIN_ONOFF = 7;

// the state of keypress
bool onoffState = HIGH;

// a flag to ensure that the releaseKey() only fire once
bool should_release_key_flag = false;

// RX, TX connect these pins to bluetooth module
SoftwareSerial bluetooth_serial(PIN_INPUT_RX, PIN_INPUT_TX);

// HC-06 bluetooth module configuration cheatsheet
// Type the command in the Serial Monitor
// AT           (simply checks connection)
// AT+VERSION   (sends the firmware verison)
// AT+NAMExxxxx (to change name to xxxxx)
// AT+PINnnnn   (to change password to 4 digit nnnn)

void setup() {
  // Open serial communications and wait for port to open:
  Serial.begin(9600);
  // set the data rate for the SoftwareSerial port
  bluetooth_serial.begin(9600);
  // setup the on/off input pin
  pinMode(PIN_ONOFF, INPUT_PULLUP);

  // just in case this needs time
  delay(200);
}

void loop() {
  // check for the on/off first
  if (digitalRead(PIN_ONOFF) == HIGH) {
    return;
  }

  if (bluetooth_serial.available()) {
    // String i = bluetooth_serial.readStirng();

    // one byte of data
    // it is going to be keycode
    uint8_t data = bluetooth_serial.read();

    // type the key from bluetooth
    pressKey(data);

    // raise the flag for relase key
    should_release_key_flag = true;

  }
  // release the key once so it won't spam releaseKey() constantly when nothing is being typed.
  else if (should_release_key_flag == true) {
    should_release_key_flag = false;
    releaseKey();
  }
  // delay so bluetooth has time to collect data
  // otherwise, it will spam releaseKey()
  // must be bigger than 1/FPS 
  delay(20);

  // For sending data back to computer
  // I don't need this functionality but keeping this for reference
  // if (Serial.available()) {
  //   bluetooth_serial.write(Serial.read());
  // }
}

void pressKey(uint8_t keycode) {
  buf[2] = keycode;
  Serial.write(buf, BUFFER_SIZE);  // Send keypress
}

// Function for Key Release
void releaseKey() {
  buf[0] = 0; // modifier 
  buf[2] = 0; // thing to write
  Serial.write(buf, BUFFER_SIZE);  // Send Release key
}
