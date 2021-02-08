import Jetson.GPIO as GPIO
import time

# Pin Definitions
output_pin = 7  # BOARD pin 7

# Pin Setup:
GPIO.setmode(GPIO.BOARD)  
GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)

print("Press CTRL+C to exit")

try:
    while True:
        # bep, bep, bep ...
        for _ in range(3):
            GPIO.output(output_pin, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(output_pin, GPIO.LOW)
            time.sleep(0.02)
        GPIO.output(output_pin, GPIO.LOW)
        time.sleep(0.5)

finally:
    GPIO.output(output_pin, GPIO.LOW)
    GPIO.cleanup()