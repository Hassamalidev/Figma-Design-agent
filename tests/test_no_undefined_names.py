"""Static check: no module uses a name it never imported or defined.

From a real failure. `build_critic_client` was called in `web/app.py`'s run
worker but never imported, so every dashboard run died with a NameError the
moment the plugin connected -- while the CLI and the benchmark, which wire the
agent separately, worked fine. The whole suite passed.

This is the cheap, permanent guard against that class of bug: it needs no
linter dependency and covers every code path, including ones no test executes.
"""
from __future__ import annotations

import ast
import builtins
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGES = ("agent", "web", "tools", "bridge", "bench", "knowledge")

# Module-level names Python injects; not imports, not assignments.
MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__"}


def project_modules() -> list[pathlib.Path]:
    files = [p for p in ROOT.glob("*.py")]
    for package in PACKAGES:
        files.extend((ROOT / package).rglob("*.py"))
    return [p for p in files if ".venv" not in p.parts]


def _bound_names(tree: ast.AST) -> set[str]:
    """Every name this module binds, by any means.

    Deliberately over-inclusive (it ignores scope), because the goal is to
    catch a name that exists NOWHERE in the module -- the NameError case --
    without reimplementing Python's scoping rules.
    """
    names = set(dir(builtins)) | MODULE_DUNDERS
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
    return names


def test_every_module_defines_every_name_it_uses():
    modules = project_modules()
    assert len(modules) > 15, "the module scan found suspiciously few files"

    problems: list[str] = []
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bound = _bound_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in bound:
                    rel = path.relative_to(ROOT)
                    problems.append(f"{rel}:{node.lineno} uses undefined name {node.id!r}")

    assert not problems, "undefined names (a NameError waiting to happen):\n  " + "\n  ".join(problems)


def test_the_check_would_actually_catch_a_missing_import(tmp_path):
    """A guard that cannot fail is not a guard."""
    broken = tmp_path / "broken.py"
    broken.write_text("def go():\n    return helper()\n", encoding="utf-8")

    tree = ast.parse(broken.read_text(encoding="utf-8"))
    bound = _bound_names(tree)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    assert "helper" in used - bound
