from MTD69x_PiezoController import controller as ctrl
a = ctrl("COM5")
a.close()


import serial
ser = serial.Serial()
ser.baudrate = 115200
ser.timeout = 1
ser.write_timeout = 1
ser.setDTR(False)
ser.setRTS(False)
ser.port = "COM5"
ser.open()

ser.flushInput()
ser.write(('xvoltage?\n\r').encode('utf-8'))
ser.read_until('\r'.encode('utf-8'))