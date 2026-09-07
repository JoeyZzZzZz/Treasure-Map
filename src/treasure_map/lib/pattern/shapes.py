# Copyright (C) 2026 JoeyZzZzZz
# SPDX-License-Identifier: Apache-2.0
"""Call-sequence shape detection.

classify() buckets a function's callees into semantic classes; the detectors then test
for two coarse shapes. A detector returns a LIST of PatternMatch (candidate shapes / leads) —
empty when the shape is absent, and never a claimed bug. The DETECTORS registry is an explicit
tuple of plain callables (no inheritance), so adding a shape is one entry plus one function.

A list rather than an optional single match because the unit a candidate describes is not the
same for every shape: most are about a FUNCTION, while the copy shape is about one CALL — a
function that copies a fixed 4 bytes at one call and a caller-supplied length at the next holds
two different facts, and one row per function can only carry one of them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace

from treasure_map.lib.pattern.classes import (
    CMD,
    COPY,
    FMT_STRING,
    FORMAT,
    PATH_SINK,
    SOURCE,
    all_format_calls_literal,
    copy_callsites,
)
from treasure_map.lib.pattern.fingerprint import (
    FINGERPRINT_ALGO_VERSION,
    structural_fingerprint,
)
from treasure_map.lib.pattern.models import FuncRef, PatternKind, PatternMatch

# A quoted literal carrying %s is shell-ish if it names a system path, contains a shell
# metacharacter, or carries a command-style flag — i.e. it looks built to feed a shell.
_PATH_PREFIXES = ("/bin/", "/sbin/", "/usr/", "/tmp/")
_SHELL_METACHARS = ";|&>`"
_FLAG_RE = re.compile(r" -\w")
_QUOTED_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class CallClasses:
    """The callees of one function, bucketed into semantic classes."""

    source: frozenset[str]
    fmt: frozenset[str]
    cmd: frozenset[str]
    copy: frozenset[str]
    fmt_string: frozenset[str]
    path_sink: frozenset[str]


def classify(callees: list[str]) -> CallClasses:
    """Bucket callee names into source / format / cmd / copy / fmt_string / path_sink classes."""
    names = {c.strip() for c in callees if isinstance(c, str) and c.strip()}
    return CallClasses(
        source=frozenset(names & SOURCE),
        fmt=frozenset(names & FORMAT),
        cmd=frozenset(names & CMD),
        copy=frozenset(names & COPY),
        fmt_string=frozenset(names & FMT_STRING),
        path_sink=frozenset(names & PATH_SINK),
    )


def _is_shellish(literal: str) -> bool:
    if any(prefix in literal for prefix in _PATH_PREFIXES):
        return True
    if any(ch in literal for ch in _SHELL_METACHARS):
        return True
    return bool(_FLAG_RE.search(literal))


def _shellish_format_literal(pseudocode: str) -> str | None:
    """Return the first quoted literal containing %s that looks shell-ish, else None."""
    for literal in _QUOTED_LITERAL.findall(pseudocode):
        if "%s" in literal and _is_shellish(literal):
            return str(literal)
    return None


def _match(
    func_ref: FuncRef,
    pattern_kind: PatternKind,
    source_class: str,
    sink_class: str,
    call_sequence_shape: str,
    evidence: str,
    *,
    sink_callsite_index: int | None = None,
    sink_callsite_occurrence: int | None = None,
) -> PatternMatch:
    """Build one match, with its fingerprint filled in.

    The callsite anchor is deliberately NOT part of the fingerprint basis (see
    fingerprint.structural_fingerprint): the fingerprint describes the SHAPE, and two calls in one
    function are the same shape. Every per-callsite sibling therefore shares one fingerprint and
    folds into one pattern row, which is what keeps the recurrence ledgers (distinct pseudocode
    hashes, distinct runs) reading the same after this split as before it."""
    partial = PatternMatch(
        func_ref=func_ref,
        pattern_kind=pattern_kind,
        source_class=source_class,
        sink_class=sink_class,
        call_sequence_shape=call_sequence_shape,
        structural_fingerprint="",
        fingerprint_algo_version=FINGERPRINT_ALGO_VERSION,
        evidence=evidence,
        sink_callsite_index=sink_callsite_index,
        sink_callsite_occurrence=sink_callsite_occurrence,
    )
    return replace(partial, structural_fingerprint=structural_fingerprint(partial))


# Recall before precision: a dangerous sink callsite is a candidate even when no source is
# recognized in this function (the controlled input may arrive through a caller — a cross-function
# flow the intra-procedural scan cannot see). Source presence is a SCORING signal, not a detection
# gate: it sets source_class (external_input vs unknown) and the shape label; its absence lowers the
# downstream review score (see the analyzer / triage) rather than dropping the candidate. A bare
# sink that is never listed would be the most hidden false negative.


def _source_class(cc: CallClasses) -> str:
    return "external_input" if cc.source else "unknown"


def pattern_a(func_ref: FuncRef, callees: list[str], pseudocode: str) -> list[PatternMatch]:
    """Command-injection shape: a shell-ish %s command string is built and run.

    Requires format + command sink + a shell-ish %s literal (a constructed shell command).
    Source is no longer a gate — when absent, source_class is 'unknown' and the shape drops the
    'source->' prefix; the value may still arrive from a caller (e.g. an argv/optarg path).

    Function-level: at most one match, as before."""
    cc = classify(callees)
    if not (cc.fmt and cc.cmd):
        return []
    literal = _shellish_format_literal(pseudocode)
    if literal is None:
        return []
    has_src = bool(cc.source)
    return [
        _match(
            func_ref,
            "cmd_injection_shape",
            _source_class(cc),
            "cmd",
            "source->format->cmd" if has_src else "format->cmd",
            literal,
        )
    ]


def bare_cmd(func_ref: FuncRef, callees: list[str], pseudocode: str) -> list[PatternMatch]:
    """Bare command-sink fallback: a command sink with NO constructed shell command.

    Fires only when pattern_a does not (no shell-ish %s literal). This is the recall net for
    command-exec sinks (system/popen/exec*) that pattern_a's shape gate would otherwise drop —
    listed at a low score (the analyzer marks it / downweights it), never silently omitted.

    Function-level: at most one match, as before."""
    cc = classify(callees)
    if not cc.cmd:
        return []
    if cc.fmt and _shellish_format_literal(pseudocode) is not None:
        return []  # pattern_a owns the constructed-shell-command case
    return [
        _match(
            func_ref,
            "bare_cmd_shape",
            _source_class(cc),
            "cmd",
            "source->cmd" if cc.source else "cmd",
            sorted(cc.cmd)[0],
        )
    ]


def pattern_b(func_ref: FuncRef, callees: list[str], pseudocode: str) -> list[PatternMatch]:
    """Copy/overflow shape: ONE candidate per copy CALLSITE. Source is a scoring signal, not a gate.

    Per callsite, not per function, because the axis a copy is read on — the write length — belongs
    to the CALL. One function copying a literal 4 bytes at its first call and a caller-supplied
    length at its second produced a single row anchored at the first: the bounded call stood in for
    the unbounded one, the length shown was the safe one, and the downstream demotion for a
    constant length then sank the whole function out of the first screen. The unbounded call had no
    row anywhere to be found by, which is the least visible way to lose a candidate.

    Evidence is the copy callee AT THIS CALLSITE, and sink_callsite_index / _occurrence say which
    call it is, so the size reader downstream classifies THAT call's length.

    When the body spells out no call to any of the callees (``pcVar1 = memcpy;`` and an indirect
    call through the pointer — 49 of 1380 copy-carrying functions on one real firmware), the
    function still yields exactly ONE candidate with no callsite anchor: the function-level match
    it has always produced, byte-for-byte. Emitting nothing there would pay for the split with a
    silent recall loss, which is the trade this change exists to refuse."""
    cc = classify(callees)
    if not cc.copy:
        return []
    shape = "source->copy" if cc.source else "copy"
    sites = copy_callsites(pseudocode, cc.copy)
    if not sites:
        return [
            _match(func_ref, "overflow_shape", _source_class(cc), "copy", shape, sorted(cc.copy)[0])
        ]
    return [
        _match(
            func_ref,
            "overflow_shape",
            _source_class(cc),
            "copy",
            shape,
            site.sink_name,
            sink_callsite_index=site.index,
            sink_callsite_occurrence=site.occurrence,
        )
        for site in sites
    ]


def pattern_fmtstr(func_ref: FuncRef, callees: list[str], pseudocode: str) -> list[PatternMatch]:
    """Format-string-injection shape: a logger/printf-family sink with a NON-LITERAL format arg.

    The literal-format exemption is the FP-suppression that GATES this recall (the overwhelmingly
    common syslog/printf passes a fixed format string and must not flood the candidate set): a sink
    is a candidate only when not all of its calls pass a literal format argument — i.e. at least one
    call's format-string position is a variable / constructed value (a format-string-injection
    suspect). Source presence is a SCORING signal, not a gate (same as pattern_b): a non-literal
    format with no recognized in-function source is still listed, just lower. The risky sink is
    chosen deterministically (sorted) so the evidence anchor is stable.

    Function-level: at most one match, as before."""
    cc = classify(callees)
    if not cc.fmt_string:
        return []
    risky = sorted(s for s in cc.fmt_string if not all_format_calls_literal(pseudocode, s))
    if not risky:
        return []  # every format-string sink uses a fixed format -> exempt (no candidate)
    has_src = bool(cc.source)
    return [
        _match(
            func_ref,
            "fmt_string_shape",
            _source_class(cc),
            "fmt_string",
            "source->fmt_string" if has_src else "fmt_string",
            risky[0],
        )
    ]


def pattern_path(func_ref: FuncRef, callees: list[str], pseudocode: str) -> list[PatternMatch]:
    """Path/file-sink shape: a path/file sink (fopen/open/unlink/rename/...). Recall net for the
    whole path-sink class (a controllable path enables traversal / arbitrary file read-write).

    Source is a scoring signal, not a gate (same as pattern_b): the path may arrive from a caller,
    so a bare path sink with no recognized in-function source is still listed. The concrete sink is
    chosen deterministically (sorted) so the evidence anchor is stable. Controllability of the path
    argument (constant / free / unknown) is decided downstream on the per-sink PATH argument.

    Function-level: at most one match, as before."""
    cc = classify(callees)
    if not cc.path_sink:
        return []
    has_src = bool(cc.source)
    return [
        _match(
            func_ref,
            "path_sink_shape",
            _source_class(cc),
            "path_sink",
            "source->path_sink" if has_src else "path_sink",
            sorted(cc.path_sink)[0],
        )
    ]


Detector = Callable[[FuncRef, list[str], str], "list[PatternMatch]"]

# Explicit registry — one entry per shape, plain callables only. pattern_a and bare_cmd are
# mutually exclusive on the same function (bare_cmd defers when pattern_a's shell-ish literal is
# present).
#
# HOW MANY candidates a (function, sink class) yields is now the shape's own answer, not a rule of
# the registry: the four function-level shapes yield 0 or 1, and pattern_b yields one per copy
# callsite. The old "at most one" held only while every shape was about a function, and reading it
# as a guarantee is what let a function's second copy call go unrepresented.
DETECTORS: tuple[Detector, ...] = (pattern_a, bare_cmd, pattern_b, pattern_fmtstr, pattern_path)
