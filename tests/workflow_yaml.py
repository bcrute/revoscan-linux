"""A deliberately small YAML subset parser, sized for this repository's own workflow file.

The offline test suite has no YAML dependency: PyYAML is not installed and adding
it is not an option, so a test that wants to assert something about
`.github/workflows/ci.yml` needs a parser of its own. This module is that parser
and nothing more. It implements the handful of YAML constructs that file uses -
block mappings, block sequences, plain and quoted scalars, comments, and the two
block scalar headers `|` and `>` - and it refuses everything else.

Refusing matters more than covering. Every construct outside the subset raises
`WorkflowYamlUnusable` with a line number, so an anchor, a flow collection, a tab
or a second document can never be silently dropped and leave a test asserting
against a quietly incomplete dictionary. When this parser returns a value, that
value accounts for the whole input.

Known, deliberate narrowness, beyond what the exceptions cover:

* Only `true` and `false` (lower case) become booleans, and only plain digit runs
  become integers. `null`, `~`, `yes`, `off`, floats, dates and the like stay
  strings; this parser resolves no tag it was not asked for.
* Quoting is honoured, so `'on'` is the string key `on` rather than the YAML 1.1
  boolean that a bare `on` would be.
* A duplicate key in one mapping is an error rather than an overwrite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["WorkflowYamlUnusable", "parse"]

_INTEGER = re.compile(r"^-?[0-9]+$")
_REFUSED_VALUE_STARTS = {
    "&": "anchors",
    "*": "aliases",
    "{": "flow mappings",
    "[": "flow sequences",
    "!": "tags",
    "?": "explicit keys",
}


class WorkflowYamlUnusable(Exception):
    """Raised when the input uses YAML that this deliberate subset does not implement."""


@dataclass(frozen=True)
class _Line:
    """One content-bearing line: its raw index, its indent column and its text."""

    number: int
    indent: int
    text: str


def parse(text: str) -> object:
    """Parse `text` as the YAML subset described in this module's docstring.

    Returns nested `dict`, `list`, `str`, `bool`, `int` and `None` values. Raises
    `WorkflowYamlUnusable` for any construct the subset does not cover.
    """
    return _Parser(text).parse()


def _indent_of(raw: str) -> int:
    return len(raw) - len(raw.lstrip(" "))


def _strip_comment(text: str, number: int) -> str:
    """Remove a trailing ` #` comment, respecting single and double quotes."""
    in_single = False
    in_double = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_single:
            if char == "'":
                in_single = False
        elif in_double:
            if char == "\\":
                index += 1
            elif char == '"':
                in_double = False
        elif char == "'":
            in_single = True
        elif char == '"':
            in_double = True
        elif char == "#" and (index == 0 or text[index - 1] in " \t"):
            return text[:index].rstrip()
        index += 1
    if in_single or in_double:
        raise WorkflowYamlUnusable(f"line {number}: unterminated quoted scalar")
    return text.rstrip()


def _scan_quoted(text: str, start: int, number: int) -> tuple[str, int]:
    """Read the quoted scalar at `start`; return its value and the index after it."""
    quote = text[start]
    parts: list[str] = []
    index = start + 1
    while index < len(text):
        char = text[index]
        if char == quote:
            if quote == "'" and text[index + 1 : index + 2] == "'":
                parts.append("'")
                index += 2
                continue
            return "".join(parts), index + 1
        if quote == '"' and char == "\\":
            escape = text[index + 1 : index + 2]
            simple = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "0": "\0"}
            if escape not in simple:
                raise WorkflowYamlUnusable(f"line {number}: unsupported escape '\\{escape}'")
            parts.append(simple[escape])
            index += 2
            continue
        parts.append(char)
        index += 1
    raise WorkflowYamlUnusable(f"line {number}: unterminated quoted scalar")


def _key_end(text: str, number: int) -> int | None:
    """Return the index of the `:` that ends a mapping key, or None if there is none."""
    if not text:
        return None
    if text[0] in "'\"":
        _, after = _scan_quoted(text, 0, number)
        if text[after : after + 1] == ":":
            return after
        return None
    index = 0
    while index < len(text):
        if text[index] == ":" and text[index + 1 : index + 2] in ("", " "):
            return index
        if text[index] in "'\"":
            return None
        index += 1
    return None


class _Parser:
    """Line-oriented recursive descent over the subset. One instance per document."""

    def __init__(self, text: str) -> None:
        self._raw = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self._lines: list[_Line] = []
        for number, raw in enumerate(self._raw, start=1):
            if "\t" in raw:
                raise WorkflowYamlUnusable(f"line {number}: tabs are not supported")
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "---" or stripped.startswith("--- ") or stripped == "...":
                raise WorkflowYamlUnusable(f"line {number}: document markers are not supported")
            content = _strip_comment(raw.strip(" "), number)
            if not content:
                continue
            self._lines.append(_Line(number - 1, _indent_of(raw), content))
        self._index = 0

    # Document ------------------------------------------------------------

    def parse(self) -> object:
        if not self._lines:
            return None
        first = self._lines[0]
        if first.indent != 0:
            raise WorkflowYamlUnusable(f"line {first.number + 1}: document must start unindented")
        value = self._parse_block(0)
        if self._index < len(self._lines):
            trailing = self._lines[self._index]
            raise WorkflowYamlUnusable(
                f"line {trailing.number + 1}: unexpected content after the document"
            )
        return value

    # Blocks --------------------------------------------------------------

    def _parse_block(self, indent: int) -> object:
        line = self._lines[self._index]
        if self._is_sequence_entry(line.text):
            return self._parse_sequence(indent)
        return self._parse_mapping(indent)

    @staticmethod
    def _is_sequence_entry(text: str) -> bool:
        return text == "-" or text.startswith("- ")

    def _parse_mapping(self, indent: int) -> dict[str, object]:
        result: dict[str, object] = {}
        while self._index < len(self._lines):
            line = self._lines[self._index]
            if line.indent < indent:
                break
            if line.indent > indent:
                raise WorkflowYamlUnusable(f"line {line.number + 1}: inconsistent indentation")
            if self._is_sequence_entry(line.text):
                raise WorkflowYamlUnusable(
                    f"line {line.number + 1}: sequence entry where a mapping key was expected"
                )
            key, rest = self._split_key(line)
            if key in result:
                raise WorkflowYamlUnusable(f"line {line.number + 1}: duplicate key '{key}'")
            self._index += 1
            result[key] = self._parse_value(line, rest, indent)
        return result

    def _parse_sequence(self, indent: int) -> list[object]:
        items: list[object] = []
        while self._index < len(self._lines):
            line = self._lines[self._index]
            if line.indent < indent:
                break
            if line.indent > indent:
                raise WorkflowYamlUnusable(f"line {line.number + 1}: inconsistent indentation")
            if not self._is_sequence_entry(line.text):
                break
            items.append(self._parse_entry(line, indent))
        return items

    def _parse_entry(self, line: _Line, indent: int) -> object:
        if line.text == "-":
            self._index += 1
            nested = self._peek()
            if nested is not None and nested.indent > indent:
                return self._parse_block(nested.indent)
            return None
        body = line.text[2:]
        content = body.lstrip(" ")
        column = indent + 2 + (len(body) - len(content))
        self._lines[self._index] = _Line(line.number, column, content)
        inner = self._lines[self._index]
        if self._is_sequence_entry(content) or _key_end(content, line.number + 1) is not None:
            return self._parse_block(column)
        self._index += 1
        return self._parse_scalar(content, inner)

    # Keys and values -----------------------------------------------------

    def _split_key(self, line: _Line) -> tuple[str, str]:
        text = line.text
        end = _key_end(text, line.number + 1)
        if end is None:
            raise WorkflowYamlUnusable(f"line {line.number + 1}: expected 'key:' or 'key: value'")
        if text[0] in "'\"":
            key, _ = _scan_quoted(text, 0, line.number + 1)
        else:
            key = text[:end].rstrip()
        if not key:
            raise WorkflowYamlUnusable(f"line {line.number + 1}: empty mapping key")
        return key, text[end + 1 :].strip()

    def _parse_value(self, line: _Line, rest: str, indent: int) -> object:
        if rest in ("|", ">"):
            return self._read_block_scalar(line, keep_newlines=rest == "|")
        if rest.startswith(("|", ">")):
            raise WorkflowYamlUnusable(
                f"line {line.number + 1}: only bare '|' and '>' block scalars are supported"
            )
        if rest:
            return self._parse_scalar(rest, line)
        nested = self._peek()
        if nested is None:
            return None
        if nested.indent > indent:
            return self._parse_block(nested.indent)
        if nested.indent == indent and self._is_sequence_entry(nested.text):
            return self._parse_sequence(indent)
        return None

    def _parse_scalar(self, text: str, line: _Line) -> object:
        number = line.number + 1
        if text[0] in "'\"":
            value, after = _scan_quoted(text, 0, number)
            if text[after:].strip():
                raise WorkflowYamlUnusable(f"line {number}: trailing text after a quoted scalar")
            return value
        refused = _REFUSED_VALUE_STARTS.get(text[0])
        if refused is not None:
            raise WorkflowYamlUnusable(f"line {number}: {refused} are not supported")
        if text in ("true", "false"):
            return text == "true"
        if _INTEGER.match(text):
            return int(text)
        return text

    # Block scalars -------------------------------------------------------

    def _read_block_scalar(self, line: _Line, *, keep_newlines: bool) -> str:
        body: list[str] = []
        base: int | None = None
        last = line.number
        cursor = line.number + 1
        while cursor < len(self._raw):
            raw = self._raw[cursor]
            if "\t" in raw:
                raise WorkflowYamlUnusable(f"line {cursor + 1}: tabs are not supported")
            if not raw.strip():
                body.append("")
                cursor += 1
                continue
            column = _indent_of(raw)
            if base is None:
                if column <= line.indent:
                    break
                base = column
            if column < base:
                break
            if not keep_newlines and column > base:
                raise WorkflowYamlUnusable(
                    f"line {cursor + 1}: extra indentation inside a folded '>' scalar"
                )
            body.append(raw[base:].rstrip())
            last = cursor
            cursor += 1
        if base is None:
            raise WorkflowYamlUnusable(f"line {line.number + 1}: block scalar has no content")
        while body and body[-1] == "":
            body.pop()
        while self._index < len(self._lines) and self._lines[self._index].number <= last:
            self._index += 1
        if not body:
            return ""
        if keep_newlines:
            return "\n".join(body) + "\n"
        return self._fold(body) + "\n"

    @staticmethod
    def _fold(body: list[str]) -> str:
        """Join each run of non-blank lines with spaces; each blank line is one newline."""
        folded: list[str] = []
        run: list[str] = []
        for entry in body:
            if entry:
                run.append(entry)
                continue
            folded.append(" ".join(run))
            run = []
            folded.append("\n")
        folded.append(" ".join(run))
        return "".join(folded)

    def _peek(self) -> _Line | None:
        if self._index < len(self._lines):
            return self._lines[self._index]
        return None
