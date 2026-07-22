import logging
import os
import threading
from pathlib import Path

# Global configuration for logging
_global_log_config = {
    "log_dir": "logs",
    "run_name": None,
    "console": True,
}
_thread_context = threading.local()

def configure_logging(log_dir: str = "logs", run_name: str = None, console: bool = True):
    _global_log_config["log_dir"] = log_dir
    _global_log_config["run_name"] = run_name
    _global_log_config["console"] = console

def set_thread_log_context(context: str | None):
    _thread_context.context = context

def clear_thread_log_context():
    if hasattr(_thread_context, "context"):
        del _thread_context.context

def get_thread_log_context() -> str | None:
    return getattr(_thread_context, "context", None)

class Logger:
    def __init__(self, name, log_dir=None, run_name=None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        self._local_log_dir = log_dir
        self._local_run_name = run_name
        
        self._setup_loggers = {}
        self.log_dir = None

    def _context(self):
        return getattr(_thread_context, "context", None)

    def _setup_handlers(self):
        context = self._context()
        logger_name = self.name if not context else f"{self.name}.{context}"
        if logger_name in self._setup_loggers:
            return self._setup_loggers[logger_name]

        current_log_dir = self._local_log_dir or _global_log_config["log_dir"]
        current_run_name = self._local_run_name or _global_log_config["run_name"]

        if current_run_name:
            current_log_dir = os.path.join(current_log_dir, current_run_name)

        Path(current_log_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = current_log_dir

        active_logger = logging.getLogger(logger_name)
        active_logger.setLevel(logging.DEBUG)
        active_logger.propagate = False

        if active_logger.handlers:
            active_logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Stream handler (console)
        if _global_log_config["console"]:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            active_logger.addHandler(stream_handler)

        # File handler
        log_file = os.path.join(current_log_dir, f"{logger_name}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        active_logger.addHandler(file_handler)

        self._setup_loggers[logger_name] = active_logger
        return active_logger

    def debug(self, msg, *args, **kwargs):
        self._setup_handlers().debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._setup_handlers().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._setup_handlers().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._setup_handlers().error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._setup_handlers().critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._setup_handlers().exception(msg, *args, **kwargs)

    def get_log_dir(self):
        self._setup_handlers()
        return self.log_dir

# Create a default logger instance
logger = Logger(__name__)

__all__ = [
    "Logger",
    "logger",
    "configure_logging",
    "set_thread_log_context",
    "clear_thread_log_context",
    "get_thread_log_context",
]
