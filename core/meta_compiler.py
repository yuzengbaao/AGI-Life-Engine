import os
import re
import ast
import importlib.util
import time
import uuid
from typing import Dict, Any


DISALLOWED_CALLS = {"eval", "exec"}
DISALLOWED_ATTRS = {"os.system", "subprocess.run", "subprocess.Popen", "shutil.rmtree"}
ALLOWED_IMPORT_ROOTS = {"math", "time", "json", "typing", "random", "datetime"}


def sanitize_source(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:python)?", "", t, flags=re.IGNORECASE).strip()
    if t.endswith("```"):
        t = t[: -3].rstrip()
    return t


def _validate_imports(node: ast.AST) -> Dict[str, Any]:
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    return {"ok": False, "error": f"disallowed import: {alias.name}"}
        if isinstance(n, ast.ImportFrom):
            root = (n.module or "").split(".")[0]
            if root and root not in ALLOWED_IMPORT_ROOTS:
                return {"ok": False, "error": f"disallowed import: {n.module}"}
    return {"ok": True}


def _validate_calls(node: ast.AST) -> Dict[str, Any]:
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id in DISALLOWED_CALLS:
                return {"ok": False, "error": f"disallowed call: {n.func.id}"}
        if isinstance(n, ast.Attribute):
            base = getattr(n.value, "id", "")
            full = f"{base}.{n.attr}" if base else n.attr
            if full in DISALLOWED_ATTRS:
                return {"ok": False, "error": f"disallowed attribute: {full}"}
    return {"ok": True}


def validate_python(source: str) -> Dict[str, Any]:
    try:
        node = ast.parse(source)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    r = _validate_imports(node)
    if not r.get("ok"):
        return r
    r = _validate_calls(node)
    if not r.get("ok"):
        return r
    return {"ok": True}


def write_module(source: str, name: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{name}.py"
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
    return path


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def compile_from_text(name: str, source_text: str, out_dir: str = "workspace/plugins") -> Dict[str, Any]:
    clean = sanitize_source(source_text)
    validation = validate_python(clean)
    if not validation.get("ok"):
        return {"success": False, "error": validation.get("error")}
    if not name:
        name = f"plugin_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    path = write_module(clean, name, out_dir)
    try:
        module = load_module(path, name)
        return {"success": True, "name": name, "path": path, "module": module}
    except Exception as e:
        return {"success": False, "error": str(e), "path": path}

