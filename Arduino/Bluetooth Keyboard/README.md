# Bluetooth Keyboard using Arduino Uno

## Introduction
I wanted to make a wireless keyboard so I could control two separate computers using one keyboard.

## Research

To get started, there are a few things to consider:

1. Existing product research
   
   Equipping both computers with different keyboards is not an option since the goal is to reduce the number of devices on the table. 
   
   There is a device called a KVM (Keyboard, Video, Mouse) switch that allows one to connect multiple computers to a single set of input devices. However, this forces me to switch my keyboard control to another device completely, and I do not want that. Also, I need to press a physical button on the KVM to switch the control - I'd prefer something easier, like clicking a button on the main computer's screen. So, I must make my own to fit my particular needs.

2. Device/microprocessor

   I already had an Arduino Uno in hand, so I went with Uno. Conveniently, Arduino Uno can also be used as an HID with a [firmware change](https://www.instructables.com/How-to-Make-a-Arduino-HID-Keyboard/). 

   Arduino Leonardo has an ATMega32u4 chip, which has USB support built in. So, Leonardos can be used as a HID (human interface device) to emulate a keyboard or mouse, too.

   And a Raspberry Pi is just too expensive.

3. Wireless connection methods

   I wanted a method readily available on many computers - Bluetooth or Wi-Fi. I went with the [HC-06 Bluetooth module](https://www.aranacorp.com/en/arduino-and-bluetooth-module-hc-06/). However, the [ESP32 Wi-Fi/Bluetooth module](https://hackaday.com/2020/02/13/emulating-a-bluetooth-keyboard-with-the-esp32/) could've been more versatile. 

4. Programming language

   I went with Python because I'm familiar with it. I used the [PyBluez](https://people.csail.mit.edu/albert/bluez-intro/x232.html) module for the main PC and [SoftwareSerial](https://www.aranacorp.com/en/arduino-and-bluetooth-module-hc-06/) for the Arduino end.  

## Keyboard

First of all, I needed a way to detect keystrokes. I thought using the PyGame module would work well since I can switch my keyboard control to the second computer by simply focusing on the PyGame window.

If sending the keystrokes to the second computer constantly is the point, you could use the [pynput module](https://pythonhosted.org/pynput/keyboard.html#monitoring-the-keyboard).

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Arduino/Bluetooth%20Keyboard/%20images/flowchart.png' width=400> 

This is the keyboard flowchart. There is an initial keystroke delay to distinguish between a single stroke and a long press. 

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Arduino/Bluetooth%20Keyboard/%20images/Keyboard.png' width=600> 

This is the key code used for HID ([more keycodes here](https://www.win.tue.nl/~aeb/linux/kbd/scancodes-14.html)). Currently, I only send keys sequentially without considering modifiers (shift, ctrl, etc.). I can add more features by adding the modifier in the keyboard buffer. [This page](https://forum.flirc.tv/index.php?/topic/2209-usb-hid-codes-keys-and-modifier-keys-for-flirc_util-record_api-x-y/) states that modifier keys are specified by logically OR'ing these values together as a binary  

> Key modifiers are defined in the IEEE HID Spec as follows:
> 
> LEFT  CONTROL          1    # 00000001 as binary
> 
> LEFT  SHIFT            2    # 00000010
> 
> LEFT  ALT              4    # 00000100
> 
> LEFT  CMD|WIN          8    # 00001000
> 
> RIGHT CONTROL          16   # 00010000
> 
> RIGHT SHIFT            32   # 00100000
> 
> RIGHT ALT              64   # 01000000
> 
> RIGHT CMD|WIN          128  # 10000000

> LEFT ALT + LEFT SHIFT = 00000110b for the 1st element of the keyboard buffer.

([example](https://github.com/SFE-Chris/UNO-HIDKeyboard-Library/blob/master/HIDKeyboard.h))

## Arduino Setup

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Arduino/Bluetooth%20Keyboard/%20images/arduino-bluetooth-hc06-voltage-div.png' width=400> 

[Image source](https://www.aranacorp.com/en/arduino-and-bluetooth-module-hc-06/)

The Arduino setup is like the above. The one important thing is that the HC-06's receiver pin can only take 3.3v, but Arduino Uno provides 5v. Use a voltage divider (1k and 2k ohm) for this. 

## Result

It works. The connection is reliable and fast. I can type on the main computer normally. I can switch to the second computer by focusing on the PyGame window and start typing there.

However, it is a bit bulky. I should explore the ESP32-as-HID option I discussed in the Research section.

If needed, I can also expand this project to mouse control since the mouse is also an HID.


