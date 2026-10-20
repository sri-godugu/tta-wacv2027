#!/usr/bin/env python3
"""fig5_strip.py -- alias for render_strip.py (the name used in the supplementary material); same arguments."""
import os, runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_strip.py"), run_name="__main__")
