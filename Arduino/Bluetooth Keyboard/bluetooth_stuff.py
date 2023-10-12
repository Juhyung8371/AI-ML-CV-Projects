# PyCharm install is broken
# use: 'pip install git+https://github.com/pybluez/pybluez.git#egg=pybluez'
import bluetooth

# bluetooth info - change it to your device's mac address
DEVICE_MAC_ADDRESS = '00:00:00:00:00:00'

# use the port 1
port = 1
# use RFCOMM protocol
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
# connect to the keyboard
sock.connect((DEVICE_MAC_ADDRESS, port))
print('Connected')
# disconnect if timeout for 1 sec
sock.settimeout(1.0)


def send_key(int_val):
    """
    Send the key via bluetooth.
    :param int_val: unsigned int for keycode
    :return: None
    """
    # send 1 byte. So, the 'big' order doesn't really matter here.
    bytes_val = int_val.to_bytes(1, 'big')
    sock.send(bytes_val)
    print('Sent data: ' + str(int_val))


def close():
    """
    Close the bluetooth connection
    :return: None
    """
    sock.close()


def __scan():
    """
    Scan the nearby devices and print their
    name and MAC addresses.
    It's mostly for testing.
    :return: None
    """
    print("Scanning for bluetooth devices:")
    devices = bluetooth.discover_devices(lookup_names=True, flush_cache=True)
    number_of_devices = len(devices)
    print(number_of_devices, "devices found")

    for addr, name in devices:
        print("\n")
        print("Device Name: %s" % name)
        print("Device MAC Address: %s" % addr)
        print("\n")
    return
