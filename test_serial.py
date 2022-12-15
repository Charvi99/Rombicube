import serial

ser = serial.Serial('COM3', 38400, timeout=0,parity=serial.PARITY_EVEN, rtscts=1)
