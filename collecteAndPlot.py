import subprocess
import msvcrt  # Windows only
import time

# === Start picoData.py in background ===
proc = subprocess.Popen(["python", "picoData.py"])
print("Recording... Press 'k' to stop early.")

try:
    while proc.poll() is None:  # While it's still running
        time.sleep(1)
        if msvcrt.kbhit():
            key = msvcrt.getch().decode().lower()
            if key == 'k':
                print("Detected 'k'. Stopping recording...")
                proc.terminate()
                break
except KeyboardInterrupt:
    print("KeyboardInterrupt — stopping recording.")
    proc.terminate()

# === Wait for it to clean up ===
proc.wait()

# === Run analysis or postprocessing ===
print("Running next script...")
subprocess.run(["python", "vis.py"])
