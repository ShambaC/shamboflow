"""Shamboflow : a Tensorflow competitor

A state of the art machine learning library with blazing fast speeds and GPU support
"""

# Startup checks and constants here

# Constant to decide whether to use GPU for computations
IS_CUDA = False
try :
    import cupy as cp
    IS_CUDA = cp.cuda.is_available()

    if IS_CUDA :
        print("CUDA enabled GPU found")
        print("-----------------------")

        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        driver_version = cp.cuda.runtime.driverGetVersion()

        print(f"CUDA Version: {cuda_version // 1000}.{(cuda_version % 1000) // 10}")
        print(f"CUDA Driver Version: {driver_version // 1000}.{(driver_version % 1000) // 10}")

        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"Name: {props['name'].decode()}")
        print(f"Compute Capability: {props['major']}.{props['minor']}")
        print(f"Total Memory: {props['totalGlobalMem'] / 1e9:.2f} GB")
        print("\n-----------------------------------------------\n")
except :
    pass

# Add Colorama fixes to terminal
from colorama import just_fix_windows_console
just_fix_windows_console()

from shamboflow import models
from shamboflow import layers
from shamboflow import callbacks