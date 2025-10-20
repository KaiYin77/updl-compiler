#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

# Color constants
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"


# Log levels
LOG_LEVEL_OFF = 0
LOG_LEVEL_ERROR = 1
LOG_LEVEL_WARN = 2
LOG_LEVEL_INFO = 3
LOG_LEVEL_DEBUG = 4
LOG_LEVEL_TRACE = 5

# Default log level
current_log_level = LOG_LEVEL_INFO


def set_log_level(level):
    """Set the global log level"""
    global current_log_level
    current_log_level = level


def get_log_level():
    """Get the current log level"""
    return current_log_level


def log_error(message):
    """Log error message in red"""
    if current_log_level >= LOG_LEVEL_ERROR:
        print(f"{RED}[ERROR] {message}{RESET}")


def log_warn(message):
    """Log warning message in yellow"""
    if current_log_level >= LOG_LEVEL_WARN:
        print(f"{YELLOW}[WARN] {message}{RESET}")


def log_info(message):
    """Log info message in white"""
    if current_log_level >= LOG_LEVEL_INFO:
        print(f"{WHITE}[INFO] {message}{RESET}")


def log_debug(message):
    """Log debug message in blue"""
    if current_log_level >= LOG_LEVEL_DEBUG:
        print(f"{BLUE}[DEBUG] {message}{RESET}")


def log_trace(message):
    """Log trace message in cyan"""
    if current_log_level >= LOG_LEVEL_TRACE:
        print(f"{CYAN}[TRACE] {message}{RESET}")
