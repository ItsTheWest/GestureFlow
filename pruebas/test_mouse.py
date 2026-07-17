import time
from pynput.mouse import Controller
mouse = Controller()
print("Starting pos:", mouse.position)
mouse.position = (500, 500)
time.sleep(0.1)
print("After abs move:", mouse.position)
mouse.move(50, 50)
time.sleep(0.1)
print("After rel move:", mouse.position)
