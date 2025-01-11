import tools  # noqa
import logging as log_module


def retrieve_logging(verbosity: int) -> int:
    if verbosity < -1:
        return log_module.CRITICAL
    if verbosity == -1:
        return log_module.ERROR
    if verbosity == 0:
        return log_module.WARNING
    if verbosity == 1:
        return log_module.INFO
    if verbosity == 2:
        return log_module.DEBUG
    return log_module.NOTSET


def set_base_log_level(verbosity: int) -> None:
    log_module.basicConfig(level=retrieve_logging(verbosity))


def set_log_level(loggers: list[str], level: int = log_module.WARNING) -> None:
    for lgr in loggers:
        log_module.getLogger(lgr).setLevel(level=level)
