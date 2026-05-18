""" BASIC USAGE EXAMPLE
This example shows how to use the MAX30102 sensor to collect data from the RED and IR channels.

The sensor is connected to the I2C bus, and the I2C bus is scanned to ensure that the sensor is connected.
The sensor is also checked to ensure that it is a MAX30102 or MAX30105 sensor.

The sensor is set up with the following parameters:
- Sample rate: 400 Hz
- Averaged samples: 8
- LED brightness: medium
- Pulse width: 411 µs
- Led mode: 2 (RED + IR)

The temperature is read at the beginning of the acquisition.

Then, in a loop the data is printed to the serial port, so that it can be plotted with a Serial Plotter.
Also the real acquisition frequency (i.e. the rate at which samples are collected from the sensor) is computed
and printed to the serial port. It differs from the sample rate, because the sensor processed the data and
averages the samples before putting them into the FIFO queue (by default, 8 samples are averaged).

Author: n-elia
"""

# Some ports need to import 'sleep' from 'time' module
from machine import I2C, SoftI2C, Pin, UART
from utime import ticks_diff, ticks_us
from time import sleep
import ustruct
import sys

from max30102 import MAX30102, MAX30105_PULSE_AMP_LOW

MUX_ADDR = 0x70
MUX_CHANNELS = [2,5]
led = Pin("LED", Pin.OUT)

def select_mux_channel(i2c, channel):
    i2c.writeto(MUX_ADDR, bytes([1 << channel]))
def main():
    # I2C software instance
    i2c = SoftI2C(sda=Pin(16),  # Here, use your I2C SDA pin
                  scl=Pin(17),  # Here, use your I2C SCL pin
                  freq=400000)  # Fast: 400kHz, slow: 100kHz

    # Examples of working I2C configurations:
    # Board             |   SDA pin  |   SCL pin
    # ------------------------------------------
    # ESP32 D1 Mini     |   22       |   21
    # TinyPico ESP32    |   21       |   22
    # Raspberry Pi Pico |   16       |   17
    # TinyS3			|	 8		 |    9

    # Sensor instance
    #sensor = MAX30102(i2c=i2c)  # An I2C instance is required

    # Select multiplexer channel first
    select_mux_channel(i2c, 2)   # use channel 2 first
    sleep(0.1)
    # Sensor instance
    sensor = MAX30102(i2c=i2c)

    # Scan I2C bus to ensure that the sensor is connected
    if sensor.i2c_address not in i2c.scan():
        print("Sensor not found.")
        return
    elif not (sensor.check_part_id()):
        # Check that the targeted sensor is compatible
        print("I2C device ID not corresponding to MAX30102 or MAX30105.")
        return
    else:
        print("Sensor connected and recognized.")

    # It's possible to set up the sensor at once with the setup_sensor() method.
    # If no parameters are supplied, the default config is loaded:
    # Led mode: 2 (RED + IR)
    # ADC range: 16384
    # Sample rate: 400 Hz
    # Led power: maximum (50.0mA - Presence detection of ~12 inch)
    # Averaged samples: 8
    # pulse width: 411
    print("Setting up sensor with default configuration.", '\n')
    sensor.setup_sensor()

    # It is also possible to tune the configuration parameters one by one.
    # Set the sample rate to 400: 400 samples/s are collected by the sensor
    # Set the number of samples to be averaged per each reading
    # Set LED brightness to a medium value
    #sensor.set_active_leds_amplitude(MAX30105_PULSE_AMP_MEDIUM)
    
#     sensor.set_adc_range(16384)
#     sensor.set_pulse_width(411)
#     sensor.set_active_leds_amplitude(0xFF)
#     sensor.set_sample_rate(3200)
#     sensor.set_fifo_average(8)
#     sensor.set_led_mode(1)
    sensor.set_adc_range(16384)
    sensor.set_pulse_width(411)          # longer LED pulse, easier to see
    sensor.set_sample_rate(400)          # slower, easier to observe
    sensor.set_fifo_average(1)
    sensor.set_led_mode(2)               # RED + IR
    sensor.set_active_leds_amplitude(0xFF)  # maximum LED brightness

    sleep(1)

    # The readTemperature() method allows to extract the die temperature in °C    
    print("Reading temperature in °C.", '\n')
    print(sensor.read_temperature())

    # Select whether to compute the acquisition frequency or not
    compute_frequency = True

    print("Starting data acquisition from RED & IR registers...", '\n')
    sleep(1)  # 50ms

    batch_size = 50  # Smaller batches = more frequent updates
    t_start = ticks_us()
    samples_collected = 0
    print("Starting high-speed acquisition...")

    while True:
#         led.on()
#         sleep(0.5)
#         led.off()
#         sleep(0.5)
        
        for ch in MUX_CHANNELS:
            select_mux_channel(i2c, ch)
            sensor.check()

            if sensor.available():
                timestamp_us = ticks_us()
                sample = sensor.pop_red_from_storage()

                # channel, timestamp, sample
                sys.stdout.buffer.write(ustruct.pack('>BIH', ch, timestamp_us, sample))

                samples_collected += 1

            # Print immediately (minimal formatting)
            
            # Print frequency stats every batch
            
            if samples_collected % batch_size == 0:
                duration_us = ticks_diff(ticks_us(), t_start)
                freq = (batch_size * 1_000_000) / duration_us
                print(f"Freq:{freq:.1f}Hz")  # Compact frequency report
                t_start = ticks_us()
            

if __name__ == '__main__':
    main()