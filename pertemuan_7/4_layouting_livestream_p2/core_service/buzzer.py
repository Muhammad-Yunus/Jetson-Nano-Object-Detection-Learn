import Jetson.GPIO as GPIO
import time

class Buzzer():
    def __init__(self, output_pin = 7):
        self.is_running = False
        self.output_pin = output_pin
        self.GPIO = GPIO
        self.GPIO.setmode(self.GPIO.BOARD)  
        self.GPIO.setwarnings(False)
        self.GPIO.setup(output_pin, self.GPIO.OUT, initial=self.GPIO.LOW)

    def main(self):
        while True :
            if self.is_running :
                print("play beep...")
                for _ in range(3):
                    self.GPIO.output(self.output_pin, self.GPIO.HIGH)
                    time.sleep(0.1)
                    self.GPIO.output(self.output_pin, self.GPIO.LOW)
                    time.sleep(0.02)   
                self.is_running = False
                
    def cleanup():
        self.GPIO.cleanup()