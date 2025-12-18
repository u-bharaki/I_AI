import sys
import os
from datetime import datetime

class TerminalLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        name = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.file = open(os.path.join(log_dir, name), "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()

def start_logging():
    logger = TerminalLogger()
    sys.stdout = logger
    return logger

def stop_logging(logger):
    logger.close()
