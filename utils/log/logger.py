import logging
import os
from pathlib import Path

# Global configuration for logging
_global_log_config = {
    "log_dir": "logs",
    "run_name": None,
    "console": True,
}

def configure_logging(log_dir: str = "logs", run_name: str = None, console: bool = True):
    _global_log_config["log_dir"] = log_dir
    _global_log_config["run_name"] = run_name
    _global_log_config["console"] = console

class Logger:
    def __init__(self, name, log_dir=None, run_name=None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        self._local_log_dir = log_dir
        self._local_run_name = run_name
        
        self._is_setup = False
        self.log_dir = None

    def _setup_handlers(self):
        if self._is_setup:
            return

        current_log_dir = self._local_log_dir or _global_log_config["log_dir"]
        current_run_name = self._local_run_name or _global_log_config["run_name"]

        if current_run_name:
            current_log_dir = os.path.join(current_log_dir, current_run_name)

        Path(current_log_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = current_log_dir

        if self.logger.handlers:
            self.logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Stream handler (console)
        if _global_log_config["console"]:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

        # File handler
        log_file = os.path.join(current_log_dir, f"{self.name}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self._is_setup = True

    def debug(self, msg, *args, **kwargs):
        self._setup_handlers()
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._setup_handlers()
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._setup_handlers()
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._setup_handlers()
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._setup_handlers()
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._setup_handlers()
        self.logger.exception(msg, *args, **kwargs)

    def get_log_dir(self):
        self._setup_handlers()
        return self.log_dir

# Create a default logger instance
logger = Logger(__name__)

__all__ = ["Logger", "logger", "configure_logging"]
