"""
Configuration file
"""

import os

def environment():
    """
    Get some environment variables
    """
    predapp_ip = os.environ.get("PREDAPP_IP", "127.0.0.1")
    predapp_port = os.environ.get("PREDAPP_PORT", 8080)
    return predapp_ip, predapp_port
