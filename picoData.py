import serial
import csv
import os

ser = serial.Serial('COM3', 115200)  # Replace with your Pico's COM port

filename = 'data\\firstTests\\ppg_data.csv'
if os.path.exists(filename):
    os.remove(filename)
    print(f"Deleted existing {filename}")

with open(filename, "w", newline='') as f:
    writer = csv.writer(f)

    print("Recording... Press Ctrl+C to stop.")
    try:
        while True:
            line = ser.readline().decode().strip()
            if line:
                writer.writerow([line])  # Wrap line in list to write as a row
                f.flush()
                print(line)
    except KeyboardInterrupt:
        print("\nStopped.")
        ser.close()

