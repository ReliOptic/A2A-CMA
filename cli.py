"""Top-level shim so ``python cli.py ...`` works without setting PYTHONPATH."""

from a2a_cma_cli.cli import main

raise SystemExit(main())
