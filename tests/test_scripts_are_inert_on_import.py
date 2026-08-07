"""No script may do anything merely by being imported.

WHY THIS FILE EXISTS. scripts/freeze.py used to run its body at module level, so
importing it opened the serial port and commanded all six joints. Anything that
imports a script -- a test collector, an editor's autocomplete, a lint pass, a
human checking whether the file parses -- drove the arm as a side effect. Found
2026-08-07 by an import check that froze the arm mid-session.

It was harmless THERE, because a freeze writes each joint's present position and
travels nowhere, which is precisely why nobody noticed. The next script written
to that pattern will not be a no-op.

Static, so it costs nothing and needs no hardware: it reads the AST rather than
importing anything.
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SCRIPTS = sorted(p for p in (ROOT / "scripts").glob("*.py"))

# Statement types that are inert at import: declarations, and the guard itself.
INERT = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
         ast.ClassDef, ast.Assign, ast.AnnAssign, ast.If, ast.Expr)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_the_script_needs_to_be_run_to_do_anything(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))

    guarded = any(isinstance(n, ast.If)
                  and ast.unparse(n.test).startswith("__name__")
                  for n in tree.body)
    assert guarded, (
        f"{path.name} has no `if __name__ == \"__main__\"` guard, so its body "
        f"runs on import. If it touches the arm, importing it drives the arm.")

    loose = [ast.unparse(n).splitlines()[0] for n in tree.body
             if not isinstance(n, INERT)]
    assert not loose, (
        f"{path.name} has top-level statements outside the guard: {loose}")


def test_this_check_covers_every_script():
    assert len(SCRIPTS) > 20, "the glob stopped matching; this test is asleep"
