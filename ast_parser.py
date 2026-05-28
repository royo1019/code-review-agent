"""AST-based code understanding for the Code Review Agent.

Walks a Python repository and builds three structured indexes:
1. Symbol table — every function/class defined in the codebase.
2. Call graph — which functions call which other functions.
3. Import graph — which files import which other modules/files.

The indexes are consumed by ``rag.py`` to enrich the LLM prompt with
codebase structure context alongside RAG retrieval.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


SKIP_DIRS = {
    ".git",
    "venv",
    ".venv",
    "env",
    ".env",
    "__pycache__",
    "node_modules",
    ".tox",
    "dist",
    "build",
    "eggs",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
}

MAX_FILE_SIZE_BYTES = 1_000_000  # skip files larger than 1MB

PYTHON_BUILTINS = {
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "callable", "chr", "classmethod", "compile", "complex", "delattr",
    "dict", "dir", "divmod", "enumerate", "eval", "exec", "filter",
    "float", "format", "frozenset", "getattr", "globals", "hasattr",
    "hash", "help", "hex", "id", "input", "int", "isinstance",
    "issubclass", "iter", "len", "list", "locals", "map", "max",
    "memoryview", "min", "next", "object", "oct", "open", "ord", "pow",
    "print", "property", "range", "repr", "reversed", "round", "set",
    "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super",
    "tuple", "type", "vars", "zip",
}


# ─── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class FunctionInfo:
    """Metadata for a single function or method definition."""

    name: str
    filepath: str
    lineno: int
    args: List[str] = field(default_factory=list)
    returns: Optional[str] = None
    calls: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_method: bool = False
    class_name: Optional[str] = None


@dataclass
class ClassInfo:
    """Metadata for a single class definition."""

    name: str
    filepath: str
    lineno: int
    bases: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class ImportInfo:
    """Metadata for a single import statement."""

    filepath: str
    module: str
    names: List[str] = field(default_factory=list)
    is_from_import: bool = False
    lineno: int = 0


@dataclass
class ASTIndex:
    """Aggregate index of functions, classes, imports, and call relationships."""

    functions: Dict[str, List[FunctionInfo]] = field(default_factory=dict)
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    imports: Dict[str, List[ImportInfo]] = field(default_factory=dict)
    call_graph: Dict[str, List[str]] = field(default_factory=dict)
    reverse_call_graph: Dict[str, List[str]] = field(default_factory=dict)
    file_to_functions: Dict[str, List[str]] = field(default_factory=dict)


# ─── Helpers ───────────────────────────────────────────────────────────


def _safe_unparse(node: ast.AST) -> str:
    """Return source-form of an AST node. Falls back to ``ast.dump`` on Python < 3.9."""
    try:
        return ast.unparse(node)
    except AttributeError:
        try:
            return ast.dump(node)
        except Exception:
            return ""
    except Exception:
        return ""


def _first_docstring_line(node: ast.AST) -> Optional[str]:
    """Return the first non-empty line of a node's docstring, or None."""
    try:
        doc = ast.get_docstring(node)
    except Exception:
        return None
    if not doc:
        return None
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def _extract_call_name(call_node: ast.Call) -> Optional[str]:
    """Extract a callable's name from an ``ast.Call`` node.

    Handles ``foo()`` (Name) and ``obj.foo()`` (Attribute). Returns None if the
    callable shape isn't recognized.
    """
    func = call_node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_function_calls(func_node: ast.AST) -> List[str]:
    """Walk a function body and return the deduplicated list of called names."""
    seen: Dict[str, None] = {}
    for child in ast.walk(func_node):
        if isinstance(child, ast.Call):
            name = _extract_call_name(child)
            if name:
                seen.setdefault(name, None)
    return list(seen.keys())


def _extract_function_args(func_node: ast.AST) -> List[str]:
    """Return positional/keyword/star arg names, excluding ``self`` and ``cls``."""
    args: List[str] = []
    try:
        arguments = func_node.args  # type: ignore[attr-defined]
    except AttributeError:
        return args

    for arg in getattr(arguments, "posonlyargs", []) or []:
        if arg.arg not in ("self", "cls"):
            args.append(arg.arg)
    for arg in arguments.args:
        if arg.arg not in ("self", "cls"):
            args.append(arg.arg)
    if getattr(arguments, "vararg", None) is not None:
        args.append(arguments.vararg.arg)
    for arg in getattr(arguments, "kwonlyargs", []) or []:
        if arg.arg not in ("self", "cls"):
            args.append(arg.arg)
    if getattr(arguments, "kwarg", None) is not None:
        args.append(arguments.kwarg.arg)
    return args


def _build_method_map(tree: ast.Module) -> Dict[int, str]:
    """Map ``lineno`` of each method node to its containing class name.

    Only direct children of ``ClassDef.body`` are considered methods; deeply
    nested functions inside methods are not counted as methods themselves.
    """
    method_map: Dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_map[child.lineno] = node.name
    return method_map


# ─── Core parsing functions ────────────────────────────────────────────


def parse_file(filepath: str) -> Optional[ast.Module]:
    """Read and parse a Python file into an AST module.

    Returns ``None`` if the file cannot be read or parsed for any reason —
    never raises. Unparseable files (syntax errors, encoding issues, missing
    files) are logged at warning level and skipped.
    """
    try:
        with open(filepath, "r", errors="ignore") as f:
            content = f.read()
    except FileNotFoundError:
        logger.warning(f"File disappeared before parsing: {filepath}")
        return None
    except OSError as e:
        logger.warning(f"Cannot read {filepath}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error reading {filepath}: {e}")
        return None

    try:
        return ast.parse(content)
    except SyntaxError as e:
        logger.warning(f"Cannot parse {filepath}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Cannot parse {filepath}: {e}")
        return None


def extract_functions(tree: ast.Module, filepath: str) -> List[FunctionInfo]:
    """Extract every function (sync + async) defined in the AST.

    Includes nested functions and class methods. Lambdas are excluded since
    they have no name. Methods are flagged with ``is_method=True`` and their
    ``class_name`` is populated.
    """
    method_map = _build_method_map(tree)
    functions: List[FunctionInfo] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        try:
            args = _extract_function_args(node)
            returns = _safe_unparse(node.returns) if node.returns is not None else None
            decorators = [_safe_unparse(d) for d in node.decorator_list]
            docstring = _first_docstring_line(node)
            calls = _collect_function_calls(node)

            class_name = method_map.get(node.lineno)
            is_method = class_name is not None

            functions.append(FunctionInfo(
                name=node.name,
                filepath=filepath,
                lineno=node.lineno,
                args=args,
                returns=returns,
                calls=calls,
                decorators=decorators,
                docstring=docstring,
                is_method=is_method,
                class_name=class_name,
            ))
        except Exception as e:
            logger.warning(f"Failed to extract function {getattr(node, 'name', '?')} in {filepath}: {e}")
            continue

    return functions


def extract_classes(tree: ast.Module, filepath: str) -> List[ClassInfo]:
    """Extract every class defined in the AST module.

    Bases include both regular parent classes and metaclass/keyword bases when
    available via ``ast.unparse``. Methods are the names of FunctionDef nodes
    directly inside the class body.
    """
    classes: List[ClassInfo] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        try:
            bases = [_safe_unparse(b) for b in node.bases]
            # include keyword bases like metaclass=
            for kw in node.keywords:
                rendered = _safe_unparse(kw)
                if rendered:
                    bases.append(rendered)

            methods: List[str] = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(child.name)

            classes.append(ClassInfo(
                name=node.name,
                filepath=filepath,
                lineno=node.lineno,
                bases=bases,
                methods=methods,
                docstring=_first_docstring_line(node),
            ))
        except Exception as e:
            logger.warning(f"Failed to extract class {getattr(node, 'name', '?')} in {filepath}: {e}")
            continue

    return classes


def extract_imports(tree: ast.Module, filepath: str) -> List[ImportInfo]:
    """Extract every import statement from the AST module.

    Handles ``import X``, ``import X, Y``, ``from X import Y``, ``from . import Y``,
    ``from .X import Y``, and ``from X import *``.
    """
    imports: List[ImportInfo] = []

    for node in ast.walk(tree):
        try:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        filepath=filepath,
                        module=alias.name,
                        names=[alias.asname or alias.name],
                        is_from_import=False,
                        lineno=node.lineno,
                    ))
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                # Encode relative-import level as leading dots, e.g. "from . import x"
                # or "from .utils import x" → ".utils".
                if node.level:
                    module_name = ("." * node.level) + module_name
                names = [alias.name for alias in node.names]
                imports.append(ImportInfo(
                    filepath=filepath,
                    module=module_name,
                    names=names,
                    is_from_import=True,
                    lineno=node.lineno,
                ))
        except Exception as e:
            logger.warning(f"Failed to extract import in {filepath} at line {getattr(node, 'lineno', '?')}: {e}")
            continue

    return imports


def build_index(repo_path: str) -> ASTIndex:
    """Walk ``repo_path``, parse every ``.py`` file, and return the aggregated index.

    Skips ignored directories (``.git``, ``venv``, ``__pycache__`` etc.) and any
    file larger than 1MB (usually generated code). Files that fail to parse are
    silently skipped after a warning log — this never crashes on a bad repo.
    """
    index = ASTIndex()
    file_count = 0

    if not repo_path or not os.path.isdir(repo_path):
        logger.warning(f"AST build_index: invalid repo_path {repo_path!r}")
        return index

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in files:
            if not filename.endswith(".py"):
                continue

            filepath = os.path.join(root, filename)

            try:
                if os.path.getsize(filepath) > MAX_FILE_SIZE_BYTES:
                    logger.debug(f"AST: skipping large file {filepath}")
                    continue
            except FileNotFoundError:
                # File vanished between walk and stat
                continue
            except OSError as e:
                logger.warning(f"AST: cannot stat {filepath}: {e}")
                continue

            tree = parse_file(filepath)
            if tree is None:
                continue

            file_count += 1

            try:
                funcs = extract_functions(tree, filepath)
                classes = extract_classes(tree, filepath)
                imps = extract_imports(tree, filepath)
            except Exception as e:
                logger.warning(f"AST extraction failed for {filepath}: {e}")
                continue

            # functions: bucket by name
            file_func_names: List[str] = []
            for fn in funcs:
                index.functions.setdefault(fn.name, []).append(fn)
                file_func_names.append(fn.name)
                key = f"{filepath}::{fn.name}"
                index.call_graph[key] = fn.calls
                for callee in fn.calls:
                    index.reverse_call_graph.setdefault(callee, []).append(key)
            index.file_to_functions[filepath] = file_func_names

            # classes
            for cls in classes:
                # Last write wins if duplicate class names exist; that's acceptable
                # for our use case (LLM context).
                index.classes[cls.name] = cls

            # imports
            if imps:
                index.imports.setdefault(filepath, []).extend(imps)

    logger.info(
        f"AST index built: {sum(len(v) for v in index.functions.values())} functions, "
        f"{len(index.classes)} classes across {file_count} files"
    )
    return index


# ─── Lookup helpers ────────────────────────────────────────────────────


def get_function_context(index: ASTIndex, function_name: str) -> Dict[str, Any]:
    """Return all known definitions of ``function_name`` plus call relationships.

    If the function name is not defined anywhere in the index, returns
    ``{"found": False, "function_name": function_name}``. Otherwise returns
    every occurrence (same name can exist in multiple files) with its callers
    and callees.
    """
    occurrences = index.functions.get(function_name)
    if not occurrences:
        return {"found": False, "function_name": function_name}

    called_by = index.reverse_call_graph.get(function_name, [])

    definitions: List[Dict[str, Any]] = []
    for fn in occurrences:
        key = f"{fn.filepath}::{fn.name}"
        calls = index.call_graph.get(key, fn.calls)
        definitions.append({
            "filepath": fn.filepath,
            "lineno": fn.lineno,
            "args": list(fn.args),
            "calls": list(calls),
            "called_by": list(called_by),
            "is_method": fn.is_method,
            "class_name": fn.class_name,
            "docstring": fn.docstring,
        })

    return {
        "found": True,
        "function_name": function_name,
        "definitions": definitions,
    }


def get_file_dependencies(index: ASTIndex, filepath: str) -> Dict[str, Any]:
    """Return the import graph view for a single file.

    Includes raw imports for the file, the list of files that import this file
    (matched by basename, since Python imports are by module name), and the
    functions defined in the file.
    """
    target_basename = os.path.splitext(os.path.basename(filepath))[0] if filepath else ""

    file_imports = index.imports.get(filepath, [])
    imports_dicts = [
        {
            "filepath": imp.filepath,
            "module": imp.module,
            "names": list(imp.names),
            "is_from_import": imp.is_from_import,
            "lineno": imp.lineno,
        }
        for imp in file_imports
    ]

    imported_by: List[str] = []
    if target_basename:
        for other_file, imp_list in index.imports.items():
            if other_file == filepath:
                continue
            for imp in imp_list:
                module = imp.module or ""
                module_tail = module.split(".")[-1] if module else ""
                if module_tail == target_basename or module.endswith("." + target_basename):
                    imported_by.append(other_file)
                    break
                # explicit name imports (from pkg import filename)
                if target_basename in imp.names:
                    imported_by.append(other_file)
                    break

    return {
        "filepath": filepath,
        "imports": imports_dicts,
        "imported_by": imported_by,
        "functions_defined": list(index.file_to_functions.get(filepath, [])),
    }


# ─── Diff → LLM-readable context ──────────────────────────────────────


_FUNC_CALL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_DIFF_LINE_RE = re.compile(r"^[+\-]")
_MAX_AST_CONTEXT_CHARS = 2000


def _extract_function_names_from_diff(diff_text: str) -> List[str]:
    """Scan diff text for ``name(`` patterns on changed lines and return unique names."""
    if not diff_text:
        return []

    names: Dict[str, None] = {}
    for raw_line in diff_text.splitlines():
        stripped = raw_line.lstrip()
        # only consider added/removed/context lines from a unified diff; we still
        # accept lines without +/- so this works on raw code snippets too.
        if stripped.startswith("@@") or stripped.startswith("+++") or stripped.startswith("---"):
            continue
        for match in _FUNC_CALL_RE.finditer(raw_line):
            name = match.group(1)
            # Filter out keywords that look like function calls (e.g. "if (", "while (")
            if name in {
                "if", "while", "for", "return", "elif", "and", "or", "not",
                "in", "is", "lambda", "yield", "await", "with", "assert",
                "raise", "del", "global", "nonlocal", "pass", "import", "from",
                "as", "class", "def", "try", "except", "finally", "else",
            }:
                continue
            names.setdefault(name, None)
    return list(names.keys())


def _extract_imported_modules_from_diff(diff_text: str) -> List[str]:
    """Best-effort scan for ``import X`` / ``from X import Y`` lines inside the diff."""
    if not diff_text:
        return []
    modules: Dict[str, None] = {}
    for raw in diff_text.splitlines():
        line = raw.lstrip("+- ").strip()
        if line.startswith("import "):
            rest = line[len("import "):].split(" as ")[0]
            for piece in rest.split(","):
                modules.setdefault(piece.strip(), None)
        elif line.startswith("from "):
            try:
                module = line.split()[1]
                modules.setdefault(module, None)
            except IndexError:
                continue
    return list(modules.keys())


def format_ast_context_for_llm(index: ASTIndex, diff_text: str) -> str:
    """Format a human-readable AST summary for the LLM prompt.

    Extracts function names from the diff (looking for ``name(`` patterns),
    resolves each against the AST index, and formats the result as a
    structured text block. Output is truncated to ~2000 chars to keep prompt
    size under control.
    """
    if not isinstance(diff_text, str):
        diff_text = "" if diff_text is None else str(diff_text)

    fn_names = _extract_function_names_from_diff(diff_text)

    if not fn_names:
        return "=== AST ANALYSIS ===\nNo function references detected in diff."

    lines: List[str] = ["=== AST ANALYSIS ===", "", "Functions referenced in this PR:", ""]

    files_touched: Dict[str, None] = {}
    for name in fn_names:
        ctx = get_function_context(index, name)
        if not ctx["found"]:
            if name in PYTHON_BUILTINS:
                lines.append(f"{name}: Python builtin")
            else:
                lines.append(f"{name}: not found in codebase")
            lines.append("")
            continue

        for defn in ctx["definitions"]:
            files_touched.setdefault(defn["filepath"], None)
            rel_path = defn["filepath"]
            header = f"{name} (defined in {rel_path}, line {defn['lineno']})"
            if defn["is_method"] and defn["class_name"]:
                header = f"{name} [method of {defn['class_name']}] (defined in {rel_path}, line {defn['lineno']})"
            lines.append(header)

            args_str = ", ".join(defn["args"]) if defn["args"] else "(none)"
            lines.append(f"  - Arguments: {args_str}")

            calls_str = ", ".join(defn["calls"]) if defn["calls"] else "(none)"
            lines.append(f"  - Calls: {calls_str}")

            if defn["called_by"]:
                callers_short = [c.split("::")[-1] for c in defn["called_by"][:10]]
                lines.append(f"  - Called by: {', '.join(callers_short)}")
            else:
                lines.append("  - Called by: (nothing calls this yet)")

            if defn["docstring"]:
                lines.append(f"  - Docstring: {defn['docstring']}")
            lines.append("")

    # Import relationships section, scoped to files we already mentioned plus
    # any explicit import lines that appear in the diff.
    import_lines: List[str] = []
    extra_modules = _extract_imported_modules_from_diff(diff_text)
    for path in files_touched.keys():
        deps = get_file_dependencies(index, path)
        if deps["imports"]:
            modules = sorted({imp["module"] for imp in deps["imports"] if imp["module"]})
            if modules:
                import_lines.append(f"{path} imports: {', '.join(modules)}")

    if extra_modules:
        import_lines.append(f"Diff explicitly imports: {', '.join(extra_modules)}")

    if import_lines:
        lines.append("=== IMPORT RELATIONSHIPS ===")
        lines.extend(import_lines)

    output = "\n".join(lines).rstrip() + "\n"

    if len(output) > _MAX_AST_CONTEXT_CHARS:
        output = output[: _MAX_AST_CONTEXT_CHARS - len(" ... (truncated for brevity)")]
        output = output.rstrip() + " ... (truncated for brevity)"

    return output
