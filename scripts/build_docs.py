"""Regenerate the public API pages and build the GitHub Pages site.

Run from an installed checkout with ``poetry run python scripts/build_docs.py``.
This command builds locally; publishing is a separate ``mkdocs gh-deploy`` step.
"""

import importlib.metadata
import inspect
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from pdoc import doc, doc_ast, doc_types, extract, render
from pydantic import BaseModel
from pydantic.dataclasses import is_pydantic_dataclass


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "mhagenta",
    "mhagenta.states",
    "mhagenta.environment",
    "mhagenta.environment.environment",
    "mhagenta.defaults.communication",
    "mhagenta.defaults.communication.rest",
    "mhagenta.core.connection.connector",
    "mhagenta.core.processes.mha_module",
    "mhagenta.utils.common.classes",
    "mhagenta.core.agent_launcher",
    "mhagenta.core.module_launcher",
    "mhagenta.environment.environment_launcher",
)

API_OVERVIEW = """Public API for MHAgentA, a framework for containerized modular hybrid agents.

## User guide

Read the [user guide](guide/) for installation, a complete minimal agent, messaging,
environment setup, persistence, and packaging instructions.

## API modules

- [Behaviour bases](mhagenta/bases.html)
- [State aliases](mhagenta/states.html) and [outboxes](mhagenta/outboxes.html)
- [Environment behaviour and runtime](mhagenta/environment.html)
- [RabbitMQ communication helpers](mhagenta/defaults/communication.html)
- [REST communication helpers](mhagenta/defaults/communication/rest.html)
- [Directory cards, messages, and domain objects](mhagenta/utils/common/classes.html)
- [Connector interface](mhagenta/core/connection/connector.html)
- [Module runtime and persistence](mhagenta/core/processes/mha_module.html)

## Container entry points

- [Agent launcher](mhagenta/core/agent_launcher.html)
- [Module launcher](mhagenta/core/module_launcher.html)
- [Environment launcher](mhagenta/environment/environment_launcher.html)
"""


def prepare_class(cls: doc.Class) -> None:
    """Use public Pydantic signatures without changing the application classes."""
    is_model = issubclass(cls.obj, BaseModel)
    if is_model:
        # Pydantic's private annotations include typing-only dependencies and names
        # shadowed by deprecated methods. Its public field metadata is resolved.
        cls._var_annotations = {
            name: field.annotation for name, field in cls.obj.model_fields.items()
        }
    elif any(base.__module__ == "mhagenta.utils.common.classes" and base.__name__ == "TagCard"
             for base in cls.obj.__mro__):
        # pdoc 16 does not include PEP 695 class parameters when evaluating an
        # attribute annotation found inside __init__, including inherited ones.
        annotations = {}
        for base in reversed(cls.obj.__mro__):
            if base is object:
                continue
            declared = dict(doc_ast.walk_tree(base).annotations)
            declared.update(getattr(base, "__annotations__", {}))
            localns = dict(vars(base))
            localns.update({p.__name__: p for p in getattr(base, "__type_params__", ())})
            annotations.update(doc_types.resolve_annotations(
                declared, inspect.getmodule(base), localns, f"{base.__module__}.{base.__qualname__}",
            ))
        cls._var_annotations = annotations

    if is_model or is_pydantic_dataclass(cls.obj):
        constructor = doc.Function(
            cls.modulename, f"{cls.qualname}.__init__", cls.obj.__init__,
            (cls.modulename, f"{cls.qualname}.__init__"),
        )
        signature = inspect.signature(cls.obj)
        constructor.signature = signature.replace(
            parameters=[inspect.Parameter("self", inspect.Parameter.POSITIONAL_ONLY),
                        *signature.parameters.values()],
            return_annotation=inspect.Signature.empty,
        )
        constructor.docstring = ""
        constructor.source = ""  # The generated validator is not the public constructor.
        cls.members["__init__"] = constructor


def render_api(destination: Path, version: str) -> None:
    """Render the selected public modules with consistent signatures and links."""
    render.configure(docformat="google", footer_text=f"MHAgentA {version}")
    modules = {name: doc.Module.from_name(name) for name in extract.walk_specs(MODULES)}
    modules["mhagenta"].docstring = API_OVERVIEW
    for module in modules.values():
        for member in module.members.values():
            if isinstance(member, doc.Class):
                prepare_class(member)
    for module in modules.values():
        output = destination / f"{module.fullname.replace('.', '/')}.html"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render.html_module(module, modules), encoding="utf-8", newline="\n")
    (destination / "index.html").write_text(render.html_index(modules), encoding="utf-8", newline="\n")
    (destination / "search.js").write_text(render.search_index(modules), encoding="utf-8", newline="\n")


def main() -> None:
    """Render current docstrings, preserve the old bases URL, and run a strict site build."""
    version = importlib.metadata.version("mhagenta")
    docs = ROOT / "docs"
    guide = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = guide.replace("(docs/images/", "(images/").replace(
        "(mhagenta/core/connection/connector.py)", "(mhagenta/core/connection/connector.html)",
    )
    (docs / "guide.md").write_text(
        "<!-- Generated from README.md by scripts/build_docs.py; edit the README instead. -->\n\n" + guide,
        encoding="utf-8", newline="\n",
    )
    with TemporaryDirectory(prefix="mhagenta-docs-") as temporary:
        rendered = Path(temporary)
        render_api(rendered, version)
        for source in rendered.rglob("*"):
            if source.is_file():
                destination = docs / source.relative_to(rendered)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

    # The package was renamed from base to bases; retain previously published links.
    (docs / "mhagenta" / "base.html").write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '  <meta charset="utf-8">\n'
        '  <title>MHAgentA behaviour bases</title>\n'
        '  <meta http-equiv="refresh" content="0; url=bases.html">\n'
        '  <script>window.location.replace("bases.html" + window.location.hash);</script>\n'
        '</head>\n<body><a href="bases.html">Continue to behaviour bases</a></body>\n</html>\n',
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict"],
        cwd=ROOT,
        check=True,
    )
    print(f"Built MHAgentA {version} documentation in {ROOT / 'build' / 'site'}")


if __name__ == "__main__":
    main()
