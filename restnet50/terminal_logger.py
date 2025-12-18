import sys
import os
from datetime import datetime


class TerminalLogger:

    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")

        self.log_file = open(log_file, 'w', encoding='utf-8')
        self.terminal = sys.stdout

        print(f"[TerminalLogger] Log dosyası oluşturuldu: {log_file}")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Anında kaydet

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        sys.stdout = self.terminal
        self.log_file.close()


def start_logging(log_dir="logs"):
    logger = TerminalLogger(log_dir)
    sys.stdout = logger
    return logger


def stop_logging(logger):
    if logger:
        logger.close()
        print("[TerminalLogger] Log kaydı tamamlandı.")