import serial
import time

x = 10.0  # 0-100V
y = 10.0
z = 10.0
# Use python -m serial.tools.list_ports
SerialPort = 'COM18'

startByte = b'\x7E'
stopByte = b'\x7D'

# 0-10 V. 0.4 mV resolution on input (0-10V) implies 0.05 pixels resolution
# on 0-5V implies 0.2mV resolution
# 5/0.0002 = 25000 minimum levels. 16 bits -> 65535


def bound_check_val(val):
    if val < 0.0:
        return 0.0
    elif val > 100.0:
        return 100.0
    else:
        return val


def bound_check_values(x, y, z):
    x_checked = bound_check_val(x)
    y_checked = bound_check_val(y)
    z_checked = bound_check_val(z)
    return x_checked, y_checked, z_checked


adc_bits = 16
adc_bytes = int(adc_bits / 8)
max_adc_val = 2 ** adc_bits - 1  # 0 - 65535 for 16 bits

x, y, z = bound_check_values(x, y, z)

x_adc = round((x / 100.0) * max_adc_val)
y_adc = round((y / 100.0) * max_adc_val)
z_adc = round((z / 100.0) * max_adc_val)

payload = bytearray()
payload.extend(startByte)
payload.extend(x_adc.to_bytes(length=adc_bytes, byteorder='big', signed=False))
payload.extend(y_adc.to_bytes(length=adc_bytes, byteorder='big', signed=False))
payload.extend(z_adc.to_bytes(length=adc_bytes, byteorder='big', signed=False))
payload.extend(stopByte)

print(payload)
print("Payload Length: ", len(payload))

ser = serial.Serial()
ser.port = SerialPort
ser.baudrate = 1000000
ser.setDTR(False)
ser.setRTS(False)
ser.open()

rep = 10000
t0 = time.time()
for x_adc in range(0, rep):
    payload = bytearray()
    payload.extend(startByte)
    payload.extend(x_adc.to_bytes(length=adc_bytes, byteorder='big', signed=False))
    payload.extend(y_adc.to_bytes(length=adc_bytes, byteorder='big', signed=False))
    payload.extend(z_adc.to_bytes(length=adc_bytes, byteorder='big', signed=False))
    payload.extend(stopByte)
    ser.write(payload)     # write a string
t1 = time.time()

elapsed = (t1 * 1000) - (t0 * 1000)
print("--- %2.5f ms ---" % elapsed)
byte_rate = ((len(payload) * rep) / elapsed) * 1000.0
print("--- %2.2f byte/seconds ---" % byte_rate)
commands_rate = (rep / elapsed) * 1000.0
print("--- %2.2f commands/seconds ---" % commands_rate)

time.sleep(5)
ser.close()             # close port