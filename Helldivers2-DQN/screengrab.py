import pyautogui
from screeninfo import get_monitors

def take_screenshot(monitor_index=0):
    # Get information about all connected monitors
    monitors = get_monitors()

    # Check if the specified monitor index is valid
    if monitor_index >= len(monitors):
        raise ValueError("Invalid monitor index")

    # Select the monitor
    monitor = monitors[monitor_index]

    # Take a screenshot of the specified monitor
    screenshot = pyautogui.screenshot(region=(monitor.x, monitor.y, monitor.width, monitor.height))

    # Save the screenshot to a file
    screenshot.save('screen.png')