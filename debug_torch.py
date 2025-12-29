import os
import sys

# Try to find site-packages
import site
packages = site.getsitepackages()
print(f"Site packages: {packages}")

torch_path = ""
for p in packages:
    t = os.path.join(p, 'torch', 'lib')
    if os.path.exists(t):
        torch_path = t
        print(f"Found torch lib at: {torch_path}")
        try:
            os.add_dll_directory(torch_path)
            print("Added torch lib to DLL directory")
        except AttributeError:
             # Python < 3.8
            os.environ['PATH'] = torch_path + ';' + os.environ['PATH']
            print("Added torch lib to PATH")
        break

try:
    import torch
    print(f"Torch imported successfully: {torch.__version__}")
except OSError as e:
    print(f"Still failed: {e}")
    # Try to identify missing dependencies if possible (hard in pure python without tools)
except Exception as e:
    print(f"Other error: {e}")
