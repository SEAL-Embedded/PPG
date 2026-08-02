"""
Combined multi-PPG + ECG acquisition on a single Pi Pico — batched
stdout version.

Same framing and channel ids as before; the only behavioral change is
that 8-byte frames are written into a pre-allocated bytearray and
flushed to stdout in batches, which avoids per-sample USB-CDC overhead.

Frame format (unchanged):
    [ 0xAA | channel (u8) | timestamp_us (u32 BE) | sample (u16 BE) ]

Channel ids:
    0..7   PPG, one per mux lane
    0xE0   ECG (sample == 0xFFFF means leads-off)
"""

from machine import I2C, Pin, ADC
from utime import ticks_diff, ticks_us, ticks_ms, sleep_ms
from time import sleep
import ustruct
import sys

from max30102 import MAX30102, MAX30105_PULSE_AMP_LOW

# --- PPG / I2C ---
I2C_SDA_PIN = 16
I2C_SCL_PIN = 17
# GP16/GP17 sit on the RP2040's hardware I2C0 peripheral (machine.I2C(0)).
# 1 MHz is Fast Mode Plus — outside the MAX30102's officially specced
# 400 kHz, but it works in practice with short wires and external pull-ups.
# Drop back to 400_000 if you see read errors or device-not-found.
I2C_FREQ = 1_000_000
I2C_BUS_ID = 0
TCA9548A_ADDR = 0x70
MAX30102_ADDR = 0x57
MAX_PPG_CHANNELS = 8

# --- ECG ---
ECG_ADC_PIN = 28
ECG_LO_PLUS_PIN = 10
ECG_LO_MINUS_PIN = 11
ECG_CHANNEL = 0xE0
ECG_LEADS_OFF_SENTINEL = 0xFFFF
ECG_INTERVAL_US = 2500           # 400 Hz

# Spacing between consecutive MAX30102 FIFO entries (µs). At
# set_sample_rate(3200) + set_fifo_average(8) the sensor produces one
# FIFO entry every 1/(3200/8) = 2500 µs. Used to back-date drained
# samples so timestamps reflect actual acquisition times rather than
# the moment we pulled the burst off the bus. Update if you change
# fifo_average: PPG_SAMPLE_PERIOD_US = 1e6 / (sample_rate / fifo_average).
PPG_SAMPLE_PERIOD_US = 2500

# --- Batching ---
FRAME_SIZE = 8
BATCH_FRAMES = 256                # 64 * 8 = 512 bytes per flush
FLUSH_INTERVAL_MS = 55           # hard latency cap if batch doesn't fill

# How often to emit the throughput status line. Rates are computed per
# stream (ECG on its own, PPG per channel) over each interval, so the
# printed ECG figure is the ACTUAL achieved ECG sampling rate.
REPORT_INTERVAL_US = 2_000_000


class Mux:
    """Thin wrapper around the TCA9548A. Caches the active channel so we
    don't repeat the i2c.writeto when servicing the same sensor twice in
    a row."""

    def __init__(self, i2c, addr=TCA9548A_ADDR):
        self.i2c = i2c
        self.addr = addr
        self._current = -1
        try:
            self.i2c.writeto(self.addr, b"\x00")
        except OSError:
            pass

    def select(self, channel):
        if channel == self._current:
            return
        self.i2c.writeto(self.addr, bytes([1 << channel]))
        self._current = channel


class FrameBuffer:
    """Pre-allocated byte buffer for outgoing frames. add() writes one
    8-byte packet at the current offset; flush() emits whatever has
    accumulated and resets. No allocations in the hot path."""

    def __init__(self, capacity_frames):
        self.capacity = capacity_frames
        self.buf = bytearray(capacity_frames * FRAME_SIZE)
        self.count = 0
        self.pending_status = []  # ASCII lines to emit after next flush

    def add(self, channel, timestamp_us, sample):
        ustruct.pack_into(
            ">BBIH",
            self.buf,
            self.count * FRAME_SIZE,
            0xAA,
            channel,
            timestamp_us & 0xFFFFFFFF,
            sample & 0xFFFF,
        )
        self.count += 1
        return self.count >= self.capacity

    def queue_status(self, line):
        self.pending_status.append(line)

    def flush(self):
        if self.count > 0:
            # memoryview avoids copying when count < capacity
            sys.stdout.buffer.write(
                memoryview(self.buf)[: self.count * FRAME_SIZE]
            )
            self.count = 0
        if self.pending_status:
            for line in self.pending_status:
                print(line)
            self.pending_status = []


def configure_sensor(sensor):
    sensor.setup_sensor()
    sensor.set_adc_range(16384)
    sensor.set_pulse_width(69)
    sensor.set_active_leds_amplitude(MAX30105_PULSE_AMP_LOW)
    sensor.set_sample_rate(3200)
    sensor.set_fifo_average(8)
    sensor.set_led_mode(1)


def discover_sensors(i2c, mux):
    found = []
    for ch in range(MAX_PPG_CHANNELS):
        mux.select(ch)
        sleep_ms(5)
        try:
            devices = i2c.scan()
        except OSError:
            devices = []
        if MAX30102_ADDR not in devices:
            continue

        sensor = MAX30102(i2c=i2c)
        if not sensor.check_part_id():
            print(f"ch{ch}: device at 0x57 is not a MAX30102/MAX30105")
            continue

        configure_sensor(sensor)
        found.append((ch, sensor))
        print(f"ch{ch}: MAX30102 ready")
    return found


def main():
    i2c = I2C(I2C_BUS_ID, sda=Pin(I2C_SDA_PIN), scl=Pin(I2C_SCL_PIN), freq=I2C_FREQ)
    mux = Mux(i2c)

    print("Discovering MAX30102 sensors behind TCA9548A...")
    sensors = discover_sensors(i2c, mux)
    print(f"Active PPG channels: {[ch for ch, _ in sensors]}")

    ecg_adc = ADC(Pin(ECG_ADC_PIN))
    lo_plus = Pin(ECG_LO_PLUS_PIN, Pin.IN)
    lo_minus = Pin(ECG_LO_MINUS_PIN, Pin.IN)
    print(
        f"ECG: ADC on GP{ECG_ADC_PIN}, LO+ GP{ECG_LO_PLUS_PIN}, "
        f"LO- GP{ECG_LO_MINUS_PIN}"
    )

    if sensors:
        mux.select(sensors[0][0])
        print(
            f"ch{sensors[0][0]} die temperature: "
            f"{sensors[0][1].read_temperature()}"
        )

    sleep(1)
    print("Starting combined PPG + ECG acquisition (batched)...")

    fb = FrameBuffer(BATCH_FRAMES)

    last_ecg_us = ticks_us()
    last_flush_ms = ticks_ms()
    ppg_index = 0

    # Per-stream sample counters for the throughput report. Kept separate
    # (not one combined total) so the printed ECG figure is the ACTUAL ECG
    # rate — the old firmware summed ECG + every PPG channel into one
    # "Freq:" number, which read ~2100 Hz and completely hid that ECG
    # itself had collapsed to ~185 Hz.
    ecg_count = 0
    ppg_count = 0
    report_start_us = ticks_us()

    def service_ecg():
        # Emit one ECG sample iff its cadence deadline (ECG_INTERVAL_US) has
        # passed. Called at the top of every loop iteration AND before every
        # single PPG FIFO read, so however long a PPG drain runs, ECG is
        # serviced between individual I2C reads — bounding ECG latency to one
        # sample-read (~0.1 ms) instead of a whole FIFO burst. This is the
        # fix for the old bimodal ECG spacing (alternating 2.5 ms / ~10 ms
        # gaps) that halved the effective ECG rate. Returns True if the frame
        # buffer filled and the caller should flush.
        nonlocal last_ecg_us, ecg_count
        now = ticks_us()
        if ticks_diff(now, last_ecg_us) < ECG_INTERVAL_US:
            return False
        last_ecg_us = now
        if lo_plus.value() or lo_minus.value():
            filled = fb.add(ECG_CHANNEL, now, ECG_LEADS_OFF_SENTINEL)
        else:
            filled = fb.add(ECG_CHANNEL, now, ecg_adc.read_u16())
        ecg_count += 1
        return filled

    while True:
        # 1. ECG first, every iteration.
        if service_ecg():
            fb.flush()
            last_flush_ms = ticks_ms()

        # 2. Service one PPG sensor, round-robin. Drain its FIFO one sample
        #    at a time, re-checking ECG (service_ecg) before every read so
        #    the I2C burst can't starve ECG. Back-date the drained samples by
        #    PPG_SAMPLE_PERIOD_US so the downstream median-dt fs inference
        #    doesn't see a burst of near-identical timestamps.
        if sensors:
            ch, sensor = sensors[ppg_index]
            mux.select(ch)
            n = sensor.fifo_available()
            if n:
                now = ticks_us()
                base = now - (n - 1) * PPG_SAMPLE_PERIOD_US
                for i in range(n):
                    if service_ecg():
                        fb.flush()
                        last_flush_ms = ticks_ms()
                    sample = sensor.read_fifo_sample()
                    if fb.add(ch, base + i * PPG_SAMPLE_PERIOD_US, sample):
                        fb.flush()
                        last_flush_ms = ticks_ms()
                ppg_count += n
            ppg_index = (ppg_index + 1) % len(sensors)

        # 3. Latency-cap flush (buffer-full flushes happen inline above).
        if ticks_diff(ticks_ms(), last_flush_ms) >= FLUSH_INTERVAL_MS:
            fb.flush()
            last_flush_ms = ticks_ms()

        # 4. Per-stream throughput report — queued, emitted at next flush so
        #    it can't tear a binary frame. Prints the true ECG rate plus the
        #    per-channel PPG rate.
        elapsed = ticks_diff(ticks_us(), report_start_us)
        if elapsed >= REPORT_INTERVAL_US:
            ecg_hz = ecg_count * 1_000_000 / elapsed
            ppg_hz = (ppg_count * 1_000_000 / elapsed / len(sensors)) if sensors else 0.0
            fb.queue_status(f"ECG:{ecg_hz:.1f}Hz PPG:{ppg_hz:.1f}Hz/ch")
            ecg_count = 0
            ppg_count = 0
            report_start_us = ticks_us()


if __name__ == "__main__":
    main()
