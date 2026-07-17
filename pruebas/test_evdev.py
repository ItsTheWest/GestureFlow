import evdev
from evdev import UInput, ecodes as e
import time

try:
    cap = {
        e.EV_REL: (e.REL_X, e.REL_Y),
        e.EV_KEY: (e.BTN_LEFT, e.BTN_RIGHT)
    }
    ui = UInput(cap, name="virtual-mouse")
    print("UInput created successfully.")
    
    time.sleep(1)
    ui.write(e.EV_REL, e.REL_X, 100)
    ui.write(e.EV_REL, e.REL_Y, 100)
    ui.syn()
    print("Moved relative 100,100")
    
    time.sleep(1)
    ui.close()
except Exception as err:
    print(f"Error: {err}")
