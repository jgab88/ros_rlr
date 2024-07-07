import serial
import time

# Open the serial port
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

# Send the command to get device health
command = b'\xA5\x52'
ser.write(command)

# Read the response
time.sleep(0.1)
response = ser.read(10)

# Close the serial port
ser.close()

# Print the response in hex format
print("Response:", response.hex())
