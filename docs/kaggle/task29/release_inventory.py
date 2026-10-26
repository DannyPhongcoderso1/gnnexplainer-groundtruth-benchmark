"""Offline packaging inventory; no installation or scientific execution."""

import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "docs/audit/elliptic_pg_stability"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def approved_sources(ignore_builder=False):
    value = json.loads((AUDIT / "TASK29_STEP02_1_CHANGE_MANIFEST.json").read_text(encoding="utf-8"))
    rows = value["modified_files"] + value["added_source_files"] + value["unchanged_files"]
    bad = [
        r["path"]
        for r in rows
        if not (ignore_builder and Path(r["path"]).name == "build_package.py")
        and sha(r["path"]) != r.get("after_sha256", r.get("sha256"))
    ]
    if bad:
        raise RuntimeError(f"Unexpected Step02.1 drift: {bad}")
    return {r["path"]: r.get("after_sha256", r.get("sha256")) for r in rows}


PIN_ROOTS = {"yaml": "PyYAML", "sklearn": "scikit-learn", "torch_geometric": "torch-geometric"}


def inventory(source_files, pins, required):
    modules = {}
    for name, path in source_files.items():
        if not name.endswith(".py"):
            continue
        if name.startswith("execution/"):
            module = Path(name).stem
        elif name.startswith("benchmark/src/"):
            module = name.removeprefix("benchmark/src/")[:-3].replace("/", ".")
        elif name.startswith("benchmark/scripts/"):
            module = name.removeprefix("benchmark/")[:-3].replace("/", ".")
            modules[Path(name).stem] = (name, Path(path))
        else:
            continue
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")
        modules[module] = (name, Path(path))
    namespaces = {".".join(n.split(".")[:i]) for n in modules for i in range(1, len(n.split(".")))}
    todo = list(required) + [
        n for n in modules if not n.startswith(("gnnexplainer_benchmark", "graphxai", "scripts"))
    ]
    seen, imports, unresolved = set(), [], []
    normalized_pins = {n.lower().replace("_", "-"): (n, v) for n, v in pins.items()}

    def resolve(module, origin):
        root = module.split(".")[0]
        if module in modules or module in namespaces:
            imports.append({"from": origin, "module": module, "kind": "PACKAGED_SOURCE"})
            if module in modules and module not in seen:
                todo.append(module)
            # Importing a child also executes its package initializers.
            for i in range(1, len(module.split("."))):
                parent = ".".join(module.split(".")[:i])
                if parent in modules and parent not in seen:
                    todo.append(parent)
        elif root in sys.stdlib_module_names or root == "__future__":
            imports.append({"from": origin, "module": module, "kind": "PYTHON_STDLIB"})
        else:
            pin = PIN_ROOTS.get(root, root).lower().replace("_", "-")
            if pin in normalized_pins:
                name, version = normalized_pins[pin]
                imports.append(
                    {
                        "from": origin,
                        "module": module,
                        "kind": "PINNED_DEPENDENCY",
                        "package": name,
                        "version": version,
                    }
                )
            else:
                unresolved.append({"from": origin, "module": module})

    while todo:
        name = todo.pop()
        if name in seen:
            continue
        if name not in modules:
            resolve(name, "declared_entrypoint_or_import")
            seen.add(name)
            continue
        seen.add(name)
        member, path = modules[name]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=member)
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolve(alias.name, member)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".") if package else []
                    base = ".".join(parts[: len(parts) - node.level + 1] + ([base] if base else []))
                if base:
                    resolve(base, member)
                for alias in node.names:
                    child = base + "." + alias.name if base else alias.name
                    if child in modules:
                        resolve(child, member)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            ):
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    resolve(node.args[0].value, member)
    if unresolved:
        raise RuntimeError(f"Unpinned/unpackaged imports: {unresolved}")
    return {
        "schema": 1,
        "status": "STATIC_CLOSURE_VERIFIED",
        "python_files": len({p for _, p in modules.values()}),
        "reachable_source_modules": sorted(n for n in seen if n in modules),
        "imports": imports,
        "unresolved": [],
        "dynamic_policy": "preflight runtime_gate enumerates protocol.required_imports; pinned library internals delegated to their pinned distributions",
        "optional_external_cli": "persistence.py uses separately authorized outer Kaggle CLI; disabled, not a scientific Python dependency",
    }


if __name__ == "__main__":
    snapshot = approved_sources()
    for path in (ROOT / "docs/kaggle/task29").rglob("*"):
        if path.is_file():
            snapshot[path.as_posix()] = sha(path)
    for path in AUDIT.glob("*"):
        if path.is_file():
            snapshot[path.as_posix()] = sha(path)
    path = Path.home() / "AppData/Local/Temp/task29_step022_before.json"
    path.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"Approved Step02.1 hashes verified; captured {len(snapshot)} files: {path}")
