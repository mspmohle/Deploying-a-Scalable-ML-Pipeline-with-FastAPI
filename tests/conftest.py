import os, sys
# add project root to sys.path so `import ml.*` works in tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
