import serial
import time

try:
    ser = serial.Serial("COM5", 115200, timeout=1)
    time.sleep(2)  # Arduino reset 等一下
    print("COM5 opened successfully")

    ser.write(b"<000000000000000000000000000000000000000000,50,3,-1,0.45,0.55>\n")
    print("packet sent")

    ser.close()
except Exception as e:
    print("failed:", repr(e))