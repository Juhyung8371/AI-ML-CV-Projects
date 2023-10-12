# Bluetooth Keyboard using Arduino Uno

## Introduction
I wanted to make a wireless keyboard so I could control two separate computers using one keyboard.

## Research

To get started, there are a few things to consider:

1. Existing product research
   
   Equipping both computers with different keyboards is not an option since the goal is to reduce the number of devices on the table. 
   
   There is a device called a KVM (Keyboard, Video, Mouse) switch that allows one to connect multiple computers to a single set of input devices. However, this forces me to switch my keyboard control to another device completely, and I do not want that. Also, I need to press a physical button on the KVM to switch the control - I'd prefer something easier, like clicking a button on the main computer's screen. So, I'll need to make my own to fit my special needs.

2. Device/microprocessor

   I already had an Arduino Uno in hand, so I went with Uno. Conveniently, Arduino Uno can also be used as a HID with a firmware change. 

   Arduino Leonardo has an ATMega32u4 chip, which has USB support built in. So, Leonardos can be used as a HID (human interface device) to emulate a keyboard or mouse, too.

   And a Raspberry Pi is just too expensive.

3. Wireless connection methods

   I wanted a method readily available on many computers - Bluetooth or Wi-Fi. I went with the [HC-06 Bluetooth module](https://www.instructables.com/How-to-Make-a-Arduino-HID-Keyboard/). However, I think the [ESP32 Wi-Fi/Bluetooth module](https://hackaday.com/2020/02/13/emulating-a-bluetooth-keyboard-with-the-esp32/) could've been more versatile. 

4. Programming language

   I went with Python because I'm familiar with it, and many resources I needed were available online. 

## Keyboard

First of all, I needed a way to detect keystrokes. I thought using the PyGame module would work well since I can switch my keyboard control to the second computer by simply focusing on the PyGame window.

If sending the keystrokes to the second computer constantly is the point, then you could use the [pynput mpdule](https://pythonhosted.org/pynput/keyboard.html#monitoring-the-keyboard).

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Arduino/Bluetooth%20Keyboard/%20images/flowchart.png' width=600> 

This is the keyboard flowchart. There is an initial keystroke delay to distinguish between a single stroke and a long press. 

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Arduino/Bluetooth%20Keyboard/%20images/Keyboard.png' width=600> 

This is the key code used for HID ([more keycodes here](https://www.win.tue.nl/~aeb/linux/kbd/scancodes-14.html)). Currently, I am only sending keys sequentially, so the functionality is limited. I can add more features, such as the Shift-press feature, by adding the modifier in the keyboard buffer. [This page](https://forum.flirc.tv/index.php?/topic/2209-usb-hid-codes-keys-and-modifier-keys-for-flirc_util-record_api-x-y/) states that modifier keys are specified by logically OR'ing these values together as a binary  

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







