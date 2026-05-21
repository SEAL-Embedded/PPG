"""
Multi-PPG acquisition over a TCA9548A I2C multiplexer.

Each MAX30102 (fixed I2C address 0x57) sits behind one channel of the
TCA9548A (default address 0x70). To talk to sensor N, we first write a
one-hot byte (1 << N) to the mux, then run the normal MAX30102 driver
against the same I2C bus.

For every sample popped from any sensor we emit one 8-byte packet:

    [ 0xAA | channel (u8) | timestamp_us (u32, big-endian) | sample (u16, big-endian) ]

The PC-side receiver (`picoDataUnpacking.py`) splits the stream back out
per channel. Status `print()`s share the same serial stream; the receiver
treats anything between newlines that doesn't frame as ASCII status.
"""

from machine import SoftI2C, Pin
from utime import ticks_diff, ticks_us, sleep_ms
from time import sleep
import ustruct
import sys

from max30102 import MAX30102, MAX30105_PULSE_AMP_LOW

# I2C pins on the Pi Pico (same as the previous single-sensor firmware).
I2C_SDA_PIN = 16
I2C_SCL_PIN = 17
I2C_FREQ = 400000

TCA9548A_ADDR = 0x70
MAX30102_ADDR = 0x57
MAX_CHANNELS = 8


class Mux:
    """Thin wrapper around the TCA9548A. Caches the active channel so we
    don't repeat the i2c.writeto when the same sensor is read twice in a
    row."""

    def __init__(self, i2c, addr=TCA9548A_ADDR):
        self.i2c = i2c
        self.addr = addr
        self._current = -1
        # Disable all channels at startup so an unconfigured mux doesn't
        # leave two sensors fighting for the bus.
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
    """Same configuration the single-sensor firmware used — preserved so
    sample rate / pulse width / averaging stay comparable across the two
    setups."""
    sensor.setup_sensor()
    sensor.set_adc_range(16384)
    sensor.set_pulse_width(69)
    sensor.set_active_leds_amplitude(MAX30105_PULSE_AMP_LOW)
    sensor.set_sample_rate(3200)
    sensor.set_fifo_average(8)
    sensor.set_led_mode(1)


def discover_sensors(i2c, mux):
    """Walk every mux lane and instantiate a MAX30102 for each one that
    answers at 0x57. Returns a list of (channel, sensor) pairs in channel
    order."""
    found = []
    for ch in range(MAX_CHANNELS):
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
    i2c = SoftI2C(sda=Pin(I2C_SDA_PIN), scl=Pin(I2C_SCL_PIN), freq=I2C_FREQ)
    mux = Mux(i2c)

    print("Discovering MAX30102 sensors behind TCA9548A...")
    sensors = discover_sensors(i2c, mux)

    if not sensors:
        print("No MAX30102 sensors found on any mux channel.")
        return

    print(f"Active channels: {[ch for ch, _ in sensors]}")

    # Optional die-temperature readout for the first sensor as a sanity
    # check — matches the spirit of the old firmware's print.
    mux.select(sensors[0][0])
    print(f"ch{sensors[0][0]} die temperature: {sensors[0][1].read_temperature()}")

    sleep(1)
    print("Starting high-speed multi-PPG acquisition...")

    batch_size = 50
    t_start = ticks_us()
    samples_collected = 0

    while True:
        for ch, sensor in sensors:
            mux.select(ch)
            sensor.check()
            while sensor.available():
                timestamp_us = ticks_us()
                sample = sensor.pop_red_from_storage()
                # 8-byte framed packet: sync, channel, t, sample.
                # Big-endian throughout to match the PC receiver.
                sys.stdout.buffer.write(
                    ustruct.pack(">BBIH", 0xAA, ch, timestamp_us & 0xFFFFFFFF, sample & 0xFFFF)
                )
                samples_collected += 1

                if samples_collected % batch_size == 0:
                    duration_us = ticks_diff(ticks_us(), t_start)
                    if duration_us > 0:
                        freq = (batch_size * 1_000_000) / duration_us
                        # Aggregate Hz across all sensors. The receiver
                        # routes this text line to its own log channel.
                        print(f"Freq:{freq:.1f}Hz")
                    t_start = ticks_us()


if __name__ == "__main__":
    main()
