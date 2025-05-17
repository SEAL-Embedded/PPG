import serial
import csv
import os

ser = serial.Serial('COM3', 115200)  # Replace with your Pico's COM port

filename = 'data\\firstTests\\ppg_data.csv'
if os.path.exists(filename):
    os.remove(filename)
    print(f"Deleted existing {filename}")

# Use newline='' for consistent line endings across platforms
with open(filename, "w", newline='') as f:
    writer = csv.writer(f)

    print("Recording... Press Ctrl+C to stop.")
    try:
        while True:
            line = ser.readline().decode().strip()
            if line:
                parts = line.split(',')
                if len(parts) == 2:
                    # Remove accidental line breaks inside parts
                    parts = [p.strip() for p in parts]
                    writer.writerow(parts)
                    f.flush()
                    print(', '.join(parts))  # Just for console clarity
                else:
                    print(f"Skipping malformed line: {line}")
    except KeyboardInterrupt:
        print("\nStopped.")
        ser.close()