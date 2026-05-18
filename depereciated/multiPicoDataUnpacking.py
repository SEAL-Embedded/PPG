"""
Multi-PPG serial receiver.

Expected packet from the Pico (one packet per sample, per sensor):

    [ 0xAA | channel (u8) | timestamp_us (u32, big-endian) | sample (u16, big-endian) ]

That's 8 bytes total. Channel ids are 0..MAX_CHANNELS-1, one per MAX30102
behind the I2C multiplexer (TCA9548A has 8 lanes, so MAX_CHANNELS = 8 by
default). The Pico is expected to switch the mux to channel N, read the
FIFO for that sensor, and emit a packet tagged with N before moving on.

Channels are opened lazily on first sample, so a 2-sensor recording
leaves behind exactly two `ppg_data_ch{N}.csv` files rather than eight.
Each file uses the same `[timestamp_us, sample]` schema the old
single-channel pipeline produced, so signal_visualization/ppgvis.py can
consume any one of them unchanged.

The matching firmware lives at `pipico_code/ppgcode/main.py` (and a
reference copy at `main micropython code pi pico.py`).
"""

import os
import struct
import sys

import serial

PORT = "COM3"
BAUD = 115200
MAX_CHANNELS = 8
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKET_LEN = 8
SYNC_BYTE = 0xAA
PACKET_FMT = ">IH"  # matches firmware ustruct.pack('>IH', ...)


def open_channel_file(output_dir, channel, cache):
    """Lazily open a CSV writer the first time we see a channel id, so a
    2-sensor recording doesn't leave behind 6 empty files for the lanes
    that weren't populated. Each row is flushed on write so a hard kill
    from the launcher loses at most the last sample."""
    f = cache.get(channel)
    if f is not None:
        return f
    path = os.path.join(output_dir, f"ppg_data_ch{channel}.csv")
    f = open(path, "w", newline="")
    cache[channel] = f
    print(f"[ch{channel}] writing -> {path}")
    return f


def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    files = {}  # channel id -> file handle, populated on first sample seen

    samples_per_ch = [0] * MAX_CHANNELS
    buffer = b""
    text_line = b""  # accumulates non-binary status bytes (e.g. "Freq:..Hz")

    print(f"Recording from {PORT} @ {BAUD}. Ctrl+C to stop.")

    try:
        while True:
            chunk = ser.read(128)
            if chunk:
                buffer += chunk

            # Walk the buffer looking for valid 8-byte packets. A byte only
            # qualifies as a sync if (a) it is 0xAA, (b) the channel byte
            # that follows is in range, and (c) we have a full packet. Bytes
            # that don't fit get treated as ascii status output.
            while len(buffer) >= PACKET_LEN:
                if buffer[0] != SYNC_BYTE:
                    # Drain any leading non-sync bytes as text — the firmware
                    # also prints "Freq:..Hz" lines to the same stream.
                    nl = buffer.find(b"\n")
                    aa = buffer.find(bytes([SYNC_BYTE]))
                    cut = len(buffer) if (nl == -1 and aa == -1) else min(
                        x for x in (nl, aa) if x != -1
                    )
                    text_line += buffer[:cut]
                    buffer = buffer[cut:]
                    if buffer.startswith(b"\n"):
                        line = text_line.decode("ascii", errors="replace").strip()
                        if line:
                            print(f"[pico] {line}")
                        text_line = b""
                        buffer = buffer[1:]
                    continue

                channel = buffer[1]
                if channel >= MAX_CHANNELS:
                    # Not a real packet — 0xAA appeared inside a timestamp
                    # or sample payload. Drop the sync byte and resync.
                    buffer = buffer[1:]
                    continue

                timestamp_us, sample = struct.unpack(
                    PACKET_FMT, buffer[2:PACKET_LEN]
                )

                f = open_channel_file(OUTPUT_DIR, channel, files)
                f.write(f"{timestamp_us},{sample}\n")
                f.flush()
                samples_per_ch[channel] += 1

                # Periodic compact status so the user sees activity across
                # all sensors without one channel drowning out the others.
                total = sum(samples_per_ch)
                if total % 500 == 0:
                    counts = " ".join(
                        f"ch{i}={n}" for i, n in enumerate(samples_per_ch) if n
                    )
                    print(f"[recv] {total} samples ({counts})")

                buffer = buffer[PACKET_LEN:]

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        for f in files.values():
            try:
                f.close()
            except Exception:
                pass
        try:
            ser.close()
        except Exception:
            pass
        print("Per-channel sample counts:")
        for ch, n in enumerate(samples_per_ch):
            print(f"  ch{ch}: {n}")


if __name__ == "__main__":
    sys.exit(main())
