import os

# Don't open the live agent graph web server while testing.
os.environ.setdefault("LIVE_GRAPH", "0")
