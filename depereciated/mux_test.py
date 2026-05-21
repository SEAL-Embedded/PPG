from machine import Pin, I2C
import time

# Your wiring:
# GP16 = SDA
# GP17 = SCL
i2c = I2C(0, sda=Pin(16), scl=Pin(17), freq=100_000)

MUX_ADDR = 0x70  # PCA9548A default address

def mux_select(channel):
    i2c.writeto(MUX_ADDR, bytes([1 << channel]))
    time.sleep_ms(10)

def mux_disable_all():
    i2c.writeto(MUX_ADDR, bytes([0x00]))
    time.sleep_ms(10)

print("Scanning main I2C bus...")
main_devices = i2c.scan()
print("Main bus devices:", [hex(addr) for addr in main_devices])

if MUX_ADDR not in main_devices:
    print("PCA9548A not found at", hex(MUX_ADDR))
    print("Try changing MUX_ADDR to 0x71-0x77 if A0/A1/A2 are not all grounded.")
else:
    print("PCA9548A found at", hex(MUX_ADDR))
    print("Scanning PCA9548A channels...")

    for channel in range(8):
        mux_select(channel)
        devices = i2c.scan()
        devices = [addr for addr in devices if addr != MUX_ADDR]

        print("Channel", channel, ":", [hex(addr) for addr in devices])

    mux_disable_all()
    print("Done.")