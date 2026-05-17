"""
Combined multi-PPG + ECG acquisition on a single Pi Pico.

PPG: each MAX30102 (0x57) sits behind one channel of a TCA9548A (0x70)
on the SoftI2C bus (SDA=GP16, SCL=GP17). PPG sensors are discovered at
startup and serviced round-robin, one FIFO read per iteration.

ECG: AD8232 analog output on GP28 (ADC2), leads-off detection on
GP10 (LO+) and GP11 (LO-). Sampled on a strict 2500 µs cadence (400 Hz)
checked at the top of every loop iteration, so cadence is preserved
regardless of how busy the PPG side is.

Frame format on the serial wire (same 8 bytes for every sample):

    [ 0xAA | channel (u8) | timestamp_us (u32 BE) | sample (u16 BE) ]

Channel ids:
    0..7   PPG, one per mux lane
    0xE0   ECG (sample == 0xFFFF means leads-off)

Status prints ("Freq:..Hz", discovery messages) share the same serial
stream as ASCII between newlines; the PC receiver routes them to its
own log line.
"""

from machine import SoftI2C, Pin, ADC
from utime import ticks_diff, ticks_us, sleep_ms
from time import sleep
import ustruct
import sys

from max30102 import MAX30102, MAX30105_PULSE_AMP_LOW

# --- PPG / I2C ---
I2C_SDA_PIN = 16
I2C_SCL_PIN = 17
I2C_FREQ = 400000
TCA9548A_ADDR = 0x70
MAX30102_ADDR = 0x57
MAX_PPG_CHANNELS = 8

# --- ECG ---
ECG_ADC_PIN = 28
ECG_LO_PLUS_PIN = 10
ECG_LO_MINUS_PIN = 11
ECG_CHANNEL = 0xE0
ECG_LEADS_OFF_SENTINEL = 0xFFFF  # unreachable by RP2040 ADC (multiples of 16)
ECG_INTERVAL_US = 2500           # 400 Hz


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


def configure_sensor(sensor):
    """Same configuration as the previous single-purpose PPG firmware,
    so sample rate / pulse width / averaging stay comparable."""
    sensor.setup_sensor()
    sensor.set_adc_range(16384)
    sensor.set_pulse_width(69)
    sensor.set_active_leds_amplitude(MAX30105_PULSE_AMP_LOW)
    sensor.set_sample_rate(3200)
    sensor.set_fifo_average(8)
    sensor.set_led_mode(1)


def discover_sensors(i2c, mux):
    """Walk every mux lane and instantiate a MAX30102 for each one that
    answers at 0x57. Returns a list of (channel, sensor) pairs in
    channel order."""
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


def emit_packet(channel, timestamp_us, sample):
    """One 8-byte framed packet to the host. Big-endian throughout."""
    sys.stdout.buffer.write(
        ustruct.pack(
            ">BBIH",
            0xAA,
            channel,
            timestamp_us & 0xFFFFFFFF,
            sample & 0xFFFF,
        )
    )


def main():
    i2c = SoftI2C(sda=Pin(I2C_SDA_PIN), scl=Pin(I2C_SCL_PIN), freq=I2C_FREQ)
    mux = Mux(i2c)

    print("Discovering MAX30102 sensors behind TCA9548A...")
    sensors = discover_sensors(i2c, mux)
    print(f"Active PPG channels: {[ch for ch, _ in sensors]}")

    # ECG front-end. GP10/GP11 are AD8232 LO+/LO- (digital), GP28 is the
    # analog AD8232 OUTPUT pin.
    ecg_adc = ADC(Pin(ECG_ADC_PIN))
    lo_plus = Pin(ECG_LO_PLUS_PIN, Pin.IN)
    lo_minus = Pin(ECG_LO_MINUS_PIN, Pin.IN)
    print(
        f"ECG: ADC on GP{ECG_ADC_PIN}, LO+ GP{ECG_LO_PLUS_PIN}, "
        f"LO- GP{ECG_LO_MINUS_PIN}"
    )

    # Optional die-temperature readout on the first PPG sensor, as a
    # sanity check the bus is healthy before we start streaming.
    if sensors:
        mux.select(sensors[0][0])
        print(f"ch{sensors[0][0]} die temperature: {sensors[0][1].read_temperature()}")

    sleep(1)
    print("Starting combined PPG + ECG acquisition...")

    last_ecg_us = ticks_us()
    ppg_index = 0
    samples_in_batch = 0
    batch_size = 200
    t_batch_start = ticks_us()

    while True:
        # 1. ECG cadence first. Checking at the top of every iteration
        # bounds the worst-case ECG jitter to roughly one mux-switch +
        # one FIFO read (a few hundred microseconds).
        now = ticks_us()
        if ticks_diff(now, last_ecg_us) >= ECG_INTERVAL_US:
            last_ecg_us = now
            if lo_plus.value() or lo_minus.value():
                emit_packet(ECG_CHANNEL, now, ECG_LEADS_OFF_SENTINEL)
            else:
                emit_packet(ECG_CHANNEL, now, ecg_adc.read_u16())
            samples_in_batch += 1

        # 2. One PPG sensor per iteration, round-robin. Pop at most one
        # FIFO sample so we don't starve ECG if a sensor's FIFO is full.
        if sensors:
            ch, sensor = sensors[ppg_index]
            mux.select(ch)
            sensor.check()
            if sensor.available():
                ts = ticks_us()
                sample = sensor.pop_red_from_storage()
                emit_packet(ch, ts, sample)
                samples_in_batch += 1
            ppg_index = (ppg_index + 1) % len(sensors)

        # 3. Periodic aggregate throughput print (ASCII, between binary
        # frames — the receiver routes it to its own log line).
        if samples_in_batch >= batch_size:
            duration_us = ticks_diff(ticks_us(), t_batch_start)
            if duration_us > 0:
                freq = (samples_in_batch * 1_000_000) / duration_us
                print(f"Freq:{freq:.1f}Hz")
            samples_in_batch = 0
            t_batch_start = ticks_us()


if __name__ == "__main__":
    main()
