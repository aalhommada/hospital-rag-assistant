#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""

import os
import sys

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
