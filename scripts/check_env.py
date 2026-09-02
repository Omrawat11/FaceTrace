"""Environment and installation diagnostic script for FaceTrace."""

import platform
import sys

def main() -> None:
    print("=" * 60)
    print("FaceTrace - Environment Diagnostic Check")
    print("=" * 60)
    print(f"Python Version: {platform.python_version()} ({sys.executable})")
    print(f"Platform:       {platform.platform()}")
    print("-" * 60)

    modules_to_check = [
        ("pydantic", "Pydantic core"),
        ("pydantic_settings", "Pydantic Settings"),
        ("dotenv", "Python Dotenv"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("insightface", "InsightFace Biometrics"),
        ("onnxruntime", "ONNX Runtime"),
        ("requests", "Requests HTTP"),
        ("web3", "Web3.py Ethereum"),
        ("eth_account", "Eth Account"),
        ("streamlit", "Streamlit UI"),
        ("pytest", "Pytest Framework"),
    ]

    installed = 0
    for mod_name, label in modules_to_check:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {label:<28} : {ver}")
            installed += 1
        except ImportError:
            print(f"  [MISSING] {label:<24} : Not yet installed")

    print("-" * 60)
    print(f"Status: {installed}/{len(modules_to_check)} key dependencies installed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
