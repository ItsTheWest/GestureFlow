import evdev
from evdev import UInput, ecodes as e, AbsInfo
import time

try:
    cap = {
        e.EV_ABS: [
            (e.ABS_X, AbsInfo(value=0, min=0, max=1366, fuzz=0, flat=0, resolution=0)),
            (e.ABS_Y, AbsInfo(value=0, min=0, max=768, fuzz=0, flat=0, resolution=0))
        ],
        e.EV_KEY: [e.BTN_TOUCH, e.BTN_LEFT]
    }
    ui = UInput(cap, name="virtual-abs-mouse", version=0x1)
    print("Abs UInput created.")
    
    time.sleep(1)
    ui.write(e.EV_ABS, e.ABS_X, 10)
    ui.write(e.EV_ABS, e.ABS_Y, 10)
    ui.syn()
    print("Moved to 10,10")
    
    time.sleep(1)
    ui.write(e.EV_ABS, e.ABS_X, 1000)
    ui.write(e.EV_ABS, e.ABS_Y, 500)
    ui.syn()
    print("Moved to 1000,500")
    
    time.sleep(1)
    ui.close()
except Exception as err:
    print(f"Error: {err}")
