import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Version: {sys.version}")
print(f"CWD: {os.getcwd()}")
print("\nSys Path:")
for p in sys.path:
    print(f"  {p}")

print("\n--- Testing Imports ---")
try:
    import sentencepiece
    print(f"sentencepiece: SENTENCEPIECE_FOUND (Location: {sentencepiece.__file__})")
except ImportError as e:
    print(f"sentencepiece: FAILED ({e})")

try:
    import google.protobuf
    print(f"protobuf: PROTOBUF_FOUND (Location: {google.protobuf.__file__})")
except ImportError as e:
    print(f"protobuf: FAILED ({e})")
