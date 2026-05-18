"""
Combined PPG + ECG serial receiver.

Reads 8-byte framed packets from a single Pi Pico running the merged
acquisition firmware (see main.py / pipico_code/...).

    [ 0xAA | channel (u8) | timestamp_us (u32 BE) | sample (u16 BE) ]

Channel ids:
    0..7   PPG  -> session_<timestamp>/ppg_data_ch{N}.csv  (timestamp_us, sample)
    0xE0   ECG  -> session_<timestamp>/ecg_data.csv        (timestamp_us, sample, leads_off)
                  sample == 0xFFFF means leads-off; written as sample=0, leads_off=1.

Output schema is header-less to stay compatible with the existing PPG
visualization pipeline (signal_visualization/ppgvis.py). All files for a
single run land in one session_<timestamp>/ folder.

Status prints from the firmware ("Freq:..Hz", discovery messages) share
the serial stream as ASCII between newlines and are routed to stdout as
[pico] lines.
"""

import os
import struct
import sys
from datetime import datetime

import serial

# PORT / BAUD / SESSION_DIR can be overridden by the webapp launcher via env
# vars so the same script keeps working when run standalone (the defaults
# below match the original hardcoded values).
PORT = os.environ.get("SEAL_PPG_PORT", "COM3")
BAUD = int(os.environ.get("SEAL_PPG_BAUD", "115200"))

MAX_PPG_CHANNELS = 8
ECG_CHANNEL = 0xE0
ECG_LEADS_OFF_SENTINEL = 0xFFFF

PACKET_LEN = 8
SYNC_BYTE = 0xAA
PAYLOAD_FMT = ">IH"  # matches firmware ustruct.pack('>BBIH', ...)

VALID_CHANNELS = set(range(MAX_PPG_CHANNELS)) | {ECG_CHANNEL}


def make_session_dir():
    # Honour SEAL_PPG_SESSION_DIR so the webapp can pre-create the folder and
    # know the exact target path without parsing this script's stdout.
    override = os.environ.get("SEAL_PPG_SESSION_DIR")
    if override:
        os.makedirs(override, exist_ok=True)
        return override
    base = os.path.dirname(os.path.abspath(__file__))
    name = "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    return path


def open_ppg_file(session_dir, channel, cache):
    """Lazy-open a PPG CSV the first time we see a channel id, so a
    2-sensor recording doesn't leave behind empty files for unused lanes."""
    f = cache.get(channel)
    if f is not None:
        return f
    path = os.path.join(session_dir, f"ppg_data_ch{channel}.csv")
    f = open(path, "w", newline="")
    cache[channel] = f
    print(f"[ch{channel}] writing -> {path}")
    return f


def open_ecg_file(session_dir, cache):
    f = cache.get(ECG_CHANNEL)
    if f is not None:
        return f
    path = os.path.join(session_dir, "ecg_data.csv")
    f = open(path, "w", newline="")
    cache[ECG_CHANNEL] = f
    print(f"[ecg] writing -> {path}")
    return f


def channel_label(channel):
    return "ecg" if channel == ECG_CHANNEL else f"ch{channel}"


def main():
    session_dir = make_session_dir()
    print(f"Session: {session_dir}")

    ser = serial.Serial(PORT, BAUD, timeout=1)
    files = {}                  # channel id -> file handle
    counts = {ch: 0 for ch in VALID_CHANNELS}
    buffer = b""
    text_line = b""             # accumulates non-binary status bytes

    print(f"Recording from {PORT} @ {BAUD}. Ctrl+C to stop.")

    try:
        while True:
            chunk = ser.read(256)
            if chunk:
                buffer += chunk

            # Walk the buffer looking for valid 8-byte packets. A byte
            # only qualifies as a sync if (a) it is 0xAA, (b) the channel
            # byte that follows is a known channel id, and (c) we have a
            # full 8 bytes. Anything else gets treated as ASCII status.
            while len(buffer) >= PACKET_LEN:
                if buffer[0] != SYNC_BYTE:
                    # Drain leading non-sync bytes as text, stopping at
                    # the next 0xAA or newline (whichever comes first).
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
                if channel not in VALID_CHANNELS:
                    # 0xAA appeared inside a timestamp or sample payload.
                    # Drop the sync byte and look for the next one.
                    buffer = buffer[1:]
                    continue

                timestamp_us, sample = struct.unpack(
                    PAYLOAD_FMT, buffer[2:PACKET_LEN]
                )

                if channel == ECG_CHANNEL:
                    f = open_ecg_file(session_dir, files)
                    leads_off = 1 if sample == ECG_LEADS_OFF_SENTINEL else 0
                    out_sample = 0 if leads_off else sample
                    f.write(f"{timestamp_us},{out_sample},{leads_off}\n")
                else:
                    f = open_ppg_file(session_dir, channel, files)
                    f.write(f"{timestamp_us},{sample}\n")
                f.flush()
                counts[channel] += 1
                buffer = buffer[PACKET_LEN:]

                # Periodic compact status so the user sees activity across
                # all sensors without one channel drowning out the others.
                total = sum(counts.values())
                if total and total % 1000 == 0:
                    parts = [
                        f"{channel_label(ch)}={counts[ch]}"
                        for ch in sorted(counts)
                        if counts[ch]
                    ]
                    print(f"[recv] {total} samples ({' '.join(parts)})")

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
        print("Final per-channel counts:")
        for ch in sorted(counts):
            if counts[ch]:
                print(f"  {channel_label(ch)}: {counts[ch]}")
        print(f"Session dir: {session_dir}")


if __name__ == "__main__":
    sys.exit(main())