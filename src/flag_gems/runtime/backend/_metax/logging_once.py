# Copyright 2026 FlagOS Contributors. Licensed under Apache-2.0.
"""Avoid constructing already filtered activation records in the owned once mode."""
import logging
import sys

from flag_gems.logging_utils import LogOncePerLocationFilter

_DEBUG = logging.Logger.debug
_FILTER = LogOncePerLocationFilter.filter
_FACTORY = logging.LogRecord
_MAKE_RECORD = logging.Logger.makeRecord
_LOG = logging.Logger._log


def _owned_filters(logger):
    if (type(logger) is not logging.Logger
            or getattr(logger.debug, '__func__', None) is not _DEBUG
            or getattr(logger.makeRecord, '__func__', None) is not _MAKE_RECORD
            or getattr(logger._log, '__func__', None) is not _LOG
            or logging.getLogRecordFactory() is not _FACTORY):
        return None
    filters = []
    current = logger
    while current is not None:
        if type(current) is not logging.Logger or current.filters or current.disabled:
            return None
        for handler in current.handlers:
            if (type(handler) is not logging.FileHandler
                    or not getattr(handler, '_flaggems_owned', False)
                    or len(handler.filters) != 1):
                return None
            once = handler.filters[0]
            if (type(once) is not LogOncePerLocationFilter
                    or getattr(once.filter, '__func__', None) is not _FILTER
                    or type(once.logged_locations) is not set):
                return None
            filters.append(once)
        if not current.propagate:
            return filters or None
        current = current.parent
    return None


def debug_once(logger, message):
    """Keep caller identity, first emission and dynamic configuration semantics."""
    frame = sys._getframe(1)
    location = (frame.f_code.co_filename, frame.f_lineno)
    # The logging configuration lock also serializes first emissions from this
    # call site. No tensor, handler, filter, or decision is cached across calls.
    with logging._lock:
        owned = _owned_filters(logger)
        if owned is not None:
            if all(location in once.logged_locations for once in owned):
                return
            logger.debug(message, stacklevel=2)
            return
    # Third-party handler/filter/factory and non-once modes retain record creation.
    logger.debug(message, stacklevel=2)
