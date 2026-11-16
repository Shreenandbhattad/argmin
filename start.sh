#!/bin/sh
# Start the argmin upload server. Open the printed address on a phone on the same Wi-Fi.
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python server.py
