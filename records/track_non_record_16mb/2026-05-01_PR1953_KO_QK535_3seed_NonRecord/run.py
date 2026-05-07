import os
import runpy
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if __name__ == "__main__":
    runpy.run_path(os.path.join(HERE, "train_gpt.py"), run_name="__main__")
