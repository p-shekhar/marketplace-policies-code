from __future__ import annotations

import sys

from marketplace_policies_code.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if not any(command in args for command in {"check", "figures", "tables", "reproduce"}):
        args = ["reproduce", *args]
    main(args)
