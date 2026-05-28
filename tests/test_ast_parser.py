"""Unit tests for ast_parser.py — every spec case."""

from __future__ import annotations

import os
import textwrap

import pytest

from ast_parser import (
    build_index,
    extract_classes,
    extract_functions,
    extract_imports,
    format_ast_context_for_llm,
    get_file_dependencies,
    get_function_context,
    parse_file,
)


# ── extract_functions ─────────────────────────────────────────────────


def test_extract_functions_basic(temp_python_file):
    path = temp_python_file("def hello():\n    return 1\n")
    funcs = extract_functions(parse_file(path), path)
    assert len(funcs) == 1
    assert funcs[0].name == "hello"
    assert funcs[0].lineno == 1
    assert funcs[0].args == []


def test_extract_functions_with_args(temp_python_file):
    src = "def foo(a, b, c=None, *args, **kwargs):\n    return a\n"
    funcs = extract_functions(parse_file(temp_python_file(src)), "x.py")
    assert funcs[0].args == ["a", "b", "c", "args", "kwargs"]


def test_extract_functions_excludes_self_and_cls(temp_python_file):
    src = textwrap.dedent("""
        class A:
            def m(self, x): pass
            @classmethod
            def k(cls, y): pass
    """)
    funcs = extract_functions(parse_file(temp_python_file(src)), "x.py")
    by_name = {f.name: f for f in funcs}
    assert by_name["m"].args == ["x"]
    assert by_name["k"].args == ["y"]


def test_extract_functions_detects_calls(temp_python_file):
    src = "def foo():\n    bar()\n    obj.baz()\n"
    funcs = extract_functions(parse_file(temp_python_file(src)), "x.py")
    assert set(funcs[0].calls) >= {"bar", "baz"}


def test_extract_functions_method_detection(temp_python_file):
    src = "class A:\n    def m(self): pass\n"
    funcs = extract_functions(parse_file(temp_python_file(src)), "x.py")
    m = next(f for f in funcs if f.name == "m")
    assert m.is_method is True
    assert m.class_name == "A"


def test_extract_functions_async(temp_python_file):
    src = "async def foo(x):\n    return x\n"
    funcs = extract_functions(parse_file(temp_python_file(src)), "x.py")
    assert funcs[0].name == "foo"
    assert funcs[0].args == ["x"]


def test_extract_functions_nested(temp_python_file):
    src = textwrap.dedent("""
        def outer():
            def inner():
                pass
            return inner
    """)
    funcs = extract_functions(parse_file(temp_python_file(src)), "x.py")
    names = {f.name for f in funcs}
    assert "outer" in names and "inner" in names
    inner = next(f for f in funcs if f.name == "inner")
    assert inner.is_method is False


def test_extract_functions_decorators(temp_python_file):
    src = textwrap.dedent("""
        class A:
            @staticmethod
            def s(): pass
            @property
            def p(self): return 1
    """)
    funcs = extract_functions(parse_file(temp_python_file(src)), "x.py")
    s = next(f for f in funcs if f.name == "s")
    p = next(f for f in funcs if f.name == "p")
    assert "staticmethod" in s.decorators
    assert "property" in p.decorators


def test_extract_functions_docstring_first_line(temp_python_file):
    src = '''def foo():
    """First line.

    Second line.
    """
    pass
'''
    funcs = extract_functions(parse_file(temp_python_file(src)), "x.py")
    assert funcs[0].docstring == "First line."


# ── extract_classes ───────────────────────────────────────────────────


def test_extract_classes_basic(temp_python_file):
    src = "class Foo:\n    pass\n"
    classes = extract_classes(parse_file(temp_python_file(src)), "x.py")
    assert len(classes) == 1
    assert classes[0].name == "Foo"
    assert classes[0].lineno == 1
    assert classes[0].bases == []
    assert classes[0].methods == []


def test_extract_classes_with_bases(temp_python_file):
    src = "class Foo(Bar, Baz):\n    pass\n"
    classes = extract_classes(parse_file(temp_python_file(src)), "x.py")
    assert "Bar" in classes[0].bases
    assert "Baz" in classes[0].bases


def test_extract_classes_methods_listed(temp_python_file):
    src = textwrap.dedent("""
        class A:
            def m(self): pass
            async def n(self): pass
    """)
    classes = extract_classes(parse_file(temp_python_file(src)), "x.py")
    assert set(classes[0].methods) == {"m", "n"}


# ── extract_imports ───────────────────────────────────────────────────


def test_extract_imports_regular(temp_python_file):
    imps = extract_imports(parse_file(temp_python_file("import os\n")), "x.py")
    assert len(imps) == 1
    assert imps[0].module == "os"
    assert imps[0].is_from_import is False


def test_extract_imports_from(temp_python_file):
    src = "from payment import process_refund\n"
    imps = extract_imports(parse_file(temp_python_file(src)), "x.py")
    assert imps[0].module == "payment"
    assert imps[0].names == ["process_refund"]
    assert imps[0].is_from_import is True


def test_extract_imports_relative(temp_python_file):
    src = "from . import utils\nfrom .helpers import x\n"
    imps = extract_imports(parse_file(temp_python_file(src)), "x.py")
    modules = {i.module for i in imps}
    assert "." in modules
    assert ".helpers" in modules


def test_extract_imports_multi_split(temp_python_file):
    imps = extract_imports(parse_file(temp_python_file("import os, sys\n")), "x.py")
    modules = [i.module for i in imps]
    assert modules.count("os") == 1
    assert modules.count("sys") == 1


def test_extract_imports_star(temp_python_file):
    imps = extract_imports(parse_file(temp_python_file("from m import *\n")), "x.py")
    assert "*" in imps[0].names


# ── parse_file edge cases ─────────────────────────────────────────────


def test_parse_file_syntax_error(temp_python_file):
    path = temp_python_file("def foo(:\n    pass\n")
    assert parse_file(path) is None


def test_parse_file_empty(temp_python_file):
    path = temp_python_file("")
    tree = parse_file(path)
    assert tree is not None  # empty module is valid


def test_parse_file_missing():
    assert parse_file("/tmp/this/does/not/exist__xyz.py") is None


# ── build_index ──────────────────────────────────────────────────────


def test_build_index_empty_dir(tmp_path):
    idx = build_index(str(tmp_path))
    assert idx.functions == {}
    assert idx.classes == {}
    assert idx.imports == {}


def test_build_index_skips_venv(tmp_path):
    (tmp_path / "real.py").write_text("def visible():\n    pass\n")
    venv_dir = tmp_path / "venv" / "site-packages"
    venv_dir.mkdir(parents=True)
    (venv_dir / "skip.py").write_text("def hidden():\n    pass\n")

    idx = build_index(str(tmp_path))
    assert "visible" in idx.functions
    assert "hidden" not in idx.functions


def test_build_index_skips_large_files(tmp_path):
    big = tmp_path / "big.py"
    big.write_text("x = 1\n" * 200_000)  # ~1.4 MB
    (tmp_path / "small.py").write_text("def small_func():\n    pass\n")
    idx = build_index(str(tmp_path))
    assert "small_func" in idx.functions
    # big.py contained no function names, so absence is implicit; assert it was
    # skipped by checking it isn't in file_to_functions either.
    big_str = str(big)
    assert big_str not in idx.file_to_functions


def test_build_index_skips_non_python(tmp_path):
    (tmp_path / "x.js").write_text("function foo() {}\n")
    (tmp_path / "README.md").write_text("# hi\n")
    idx = build_index(str(tmp_path))
    assert idx.functions == {}


def test_call_graph_built_correctly(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    bar()\n")
    (tmp_path / "b.py").write_text("def bar():\n    pass\n")
    idx = build_index(str(tmp_path))
    keys = list(idx.call_graph.keys())
    foo_key = next(k for k in keys if k.endswith("::foo"))
    assert "bar" in idx.call_graph[foo_key]


def test_reverse_call_graph_built_correctly(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    bar()\n")
    (tmp_path / "b.py").write_text("def bar():\n    pass\n")
    idx = build_index(str(tmp_path))
    assert "bar" in idx.reverse_call_graph
    assert any("foo" in caller for caller in idx.reverse_call_graph["bar"])


def test_file_dependencies(tmp_path):
    (tmp_path / "lib.py").write_text("def util(): pass\n")
    (tmp_path / "use.py").write_text("from lib import util\n")
    idx = build_index(str(tmp_path))
    deps = get_file_dependencies(idx, str(tmp_path / "lib.py"))
    assert any("use.py" in p for p in deps["imported_by"])


# ── get_function_context ──────────────────────────────────────────────


def test_get_function_context_not_found(sample_repo):
    idx = build_index(sample_repo)
    res = get_function_context(idx, "definitely_not_in_codebase")
    assert res == {"found": False, "function_name": "definitely_not_in_codebase"}


def test_get_function_context_found(sample_repo):
    idx = build_index(sample_repo)
    res = get_function_context(idx, "get_user")
    assert res["found"] is True
    assert len(res["definitions"]) >= 1
    defn = res["definitions"][0]
    for key in ("filepath", "lineno", "args", "calls", "called_by", "is_method", "class_name", "docstring"):
        assert key in defn


# ── format_ast_context_for_llm ────────────────────────────────────────


def test_format_ast_context_detects_functions_in_diff(sample_repo):
    idx = build_index(sample_repo)
    out = format_ast_context_for_llm(idx, "+x = get_user(name)\n")
    assert "get_user" in out


def test_format_ast_context_empty_diff(sample_repo):
    idx = build_index(sample_repo)
    out = format_ast_context_for_llm(idx, "")
    assert "AST ANALYSIS" in out
    assert "No function references" in out


def test_format_ast_context_truncates_long_output(sample_repo):
    idx = build_index(sample_repo)
    # huge diff with hundreds of function calls
    diff = "\n".join(f"+x{i} = some_func_{i}()" for i in range(500))
    out = format_ast_context_for_llm(idx, diff)
    assert len(out) <= 2050  # ~2000 chars + truncation suffix
    assert "truncated" in out


def test_format_ast_context_handles_none(sample_repo):
    idx = build_index(sample_repo)
    out = format_ast_context_for_llm(idx, None)
    assert "AST ANALYSIS" in out
