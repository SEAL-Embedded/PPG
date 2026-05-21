import serial
import csv
from datetime import datetime

PORT = "COM5"
BAUD = 115200
OUTPUT_FILE = "ecg_data_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M%S"))

print("Connecting to {}...".format(PORT))

with serial.Serial(PORT, BAUD, timeout=1) as ser, \
     open(OUTPUT_FILE, "w", newline="") as csvfile:

    writer = csv.writer(csvfile)
    writer.writerow(["time_ms", "reading", "bpm", "leads_off"])

    print("Recording... Press Ctrl+C to stop.")
    print("Saving to:", OUTPUT_FILE)

    try:
        while True:
            line = ser.readline().decode("utf-8").strip()
            if not line:
                continue

            # Print everything to console; skip parsing comment lines
            print(line)
            if line.startswith("#"):
                continue

            parts = line.split(",")
            if len(parts) == 4:
                writer.writerow(parts)
                csvfile.flush()  # ensure data is written even if not stopped cleanly

    except KeyboardInterrupt:
        print("\nDone. Saved to", OUTPUT_FILE)