import serial
import struct
import csv
import re

PORT = 'COM3'
BAUD = 115200
FRAME_FMT = '>IH'                              # matches Pico: ustruct.pack('>IH', ts, sample)
FRAME_LEN = struct.calcsize(FRAME_FMT)         # 6
BATCH_FRAMES = 50                              # matches Pico batch_size
BATCH_BYTES = FRAME_LEN * BATCH_FRAMES         # 300

def read_exact(ser, n, buf):
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if chunk:
            buf.extend(chunk)
    out = bytes(buf[:n])
    del buf[:n]
    return out

def read_line(ser, buf):
    while True:
        idx = buf.find(b'\n')
        if idx >= 0:
            line = bytes(buf[:idx + 1])
            del buf[:idx + 1]
            return line
        chunk = ser.read(64)
        if chunk:
            buf.extend(chunk)

ser = serial.Serial(PORT, BAUD, timeout=1)
buf = bytearray()

# Sync: discard everything up to and including the first "Hz\n".
# The byte immediately after is guaranteed to be the start of a fresh 50-frame batch.
print("Syncing to first Freq line...")
while True:
    chunk = ser.read(64)
    if chunk:
        buf.extend(chunk)
    m = re.search(rb'Hz\r?\n', bytes(buf))
    if m:
        del buf[:m.end()]
        print("Synced.")
        break

with open("ppg_data.csv", "w", newline='') as f:
    writer = csv.writer(f)

    print("Recording... Ctrl+C to stop.")
    try:
        while True:
            data = read_exact(ser, BATCH_BYTES, buf)
            for i in range(BATCH_FRAMES):
                ts, samp = struct.unpack_from(FRAME_FMT, data, i * FRAME_LEN)
                writer.writerow([ts, samp])

            line = read_line(ser, buf).decode('ascii', errors='replace').strip()
            if line:
                print(line)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        ser.close()
