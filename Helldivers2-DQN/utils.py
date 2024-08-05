import os
import cv2
import torch
import random
import numpy as np
from PIL import Image


def move_file(current_name, base_new_name, target_directory="learned_policy"):
    # Check if the file exists in the current directory
    if os.path.exists(current_name):
        # Ensure the target directory exists; if not, create it
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Calculate the next file number based on the number of files in the directory
        next_file_number = len(os.listdir(target_directory)) + 1

        # Construct the new file name with the number appended
        new_name = f"{base_new_name}_{next_file_number}.pth"

        # Construct the new file path including the directory
        new_path = os.path.join(target_directory, new_name)

        # Rename and move the file
        os.rename(current_name, new_path)
        return new_path
    
def get_random_image(image_dir):
    # Define the directory labels
    labels = ["Left", "Right", "Up", "Down"]
    
    # Pick a random label from the list
    true_label = random.choice(labels)
    true_label_index = labels.index(true_label)
    
    # Build the path to the selected label directory
    label_directory = os.path.join(image_dir, true_label)
    
    # List all PNG files in the selected directory
    png_files = [file for file in os.listdir(label_directory) if file.endswith('.png')]
    
    # If there are no PNG files, return an error message or handle as needed
    if not png_files:
        return "No PNG files found in the selected directory.", true_label_index
    
    # Pick a random PNG file from the list
    random_png = random.choice(png_files)
    
    # Build the full path to the randomly selected PNG file
    file_path = os.path.join(label_directory, random_png)
    
    return file_path, true_label_index

def calculate_eps_decay(eps_start, eps_end, n_steps, eps_percentage):
    # Calculate the rate epsilon should decay
    effective_steps = n_steps * eps_percentage
    decrement_per_step = (eps_start - eps_end) / effective_steps
    return decrement_per_step

def preprocess_image(image, target_size, binary):
    # Convert the OpenCV image to a PIL image
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Convert image to grayscale
    img = img.convert("L")
    
    # Resize the image to the target size using the LANCZOS resampling method
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert the PIL image to a numpy array
    img_array = np.array(img)
    
    # Apply a threshold to convert the image to binary (black and white) if specified
    if binary:
        img_array = np.where(img_array > 128, 255, 0)
    
    # Convert the numpy array to a tensor, ensure the data type is float for normalization
    img_tensor = torch.from_numpy(img_array).float()
    
    # Normalize the image tensor to 0-1 by dividing by 255
    img_tensor /= 255.0
    
    # Add a channel dimension: (H, W) -> (1, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    
    # Now the shape is [1, target_size, target_size]
    return img_tensor
    
def env_step(args, predicted_label, true_label, processed_image):
    # If correct next_state = processed_image
    if predicted_label == true_label:
        return processed_image, args.reward, True
    # else it is all 0's
    else:
        processed_image[:] = 0
        return processed_image, 0, False
    




import ctypes
from ctypes import wintypes
import time
user32 = ctypes.WinDLL('user32', use_last_error=True)
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP       = 0x0002
KEYEVENTF_UNICODE     = 0x0004
MAPVK_VK_TO_VSC = 0
# msdn.microsoft.com/en-us/library/dd375731
wintypes.ULONG_PTR = wintypes.WPARAM
class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx",          wintypes.LONG),
                ("dy",          wintypes.LONG),
                ("mouseData",   wintypes.DWORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))
class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk",         wintypes.WORD),
                ("wScan",       wintypes.WORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))
    def __init__(self, *args, **kwds):
        super(KEYBDINPUT, self).__init__(*args, **kwds)
        if not self.dwFlags & KEYEVENTF_UNICODE:
            self.wScan = user32.MapVirtualKeyExW(self.wVk,
                                                 MAPVK_VK_TO_VSC, 0)
class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg",    wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))
class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                    ("mi", MOUSEINPUT),
                    ("hi", HARDWAREINPUT))
    _anonymous_ = ("_input",)
    _fields_ = (("type",   wintypes.DWORD),
                ("_input", _INPUT))
LPINPUT = ctypes.POINTER(INPUT)
def PressKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))
def ReleaseKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode,
                            dwFlags=KEYEVENTF_KEYUP))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))
def WPress():
    PressKey(0x57)
    time.sleep(0.1)
    ReleaseKey(0x57)
    # you can change 0x30 to any key you want. For more info look at :
    # msdn.microsoft.com/en-us/library/dd375731

def SPress():
    PressKey(0x53)
    time.sleep(0.1)
    ReleaseKey(0x53)
    # you can change 0x30 to any key you want. For more info look at :
    # msdn.microsoft.com/en-us/library/dd375731
def APress():
    PressKey(0x41)
    time.sleep(0.1)
    ReleaseKey(0x41)
    # you can change 0x30 to any key you want. For more info look at :
    # msdn.microsoft.com/en-us/library/dd375731
def DPress():
    PressKey(0x44)
    time.sleep(0.1)
    ReleaseKey(0x44)
    # you can change 0x30 to any key you want. For more info look at :
    # msdn.microsoft.com/en-us/library/dd375731