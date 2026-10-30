from .cli import main

try:
    main()
except (RuntimeError, FileNotFoundError) as exc:
    raise SystemExit(f"ERROR: {exc}") from None
