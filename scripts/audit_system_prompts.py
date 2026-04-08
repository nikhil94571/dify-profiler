#!/usr/bin/env python3
"""Static audit for worker system prompts.

This script checks for:
- required section coverage,
- artifact/input semantics coverage,
- example coverage,
- enum coverage against validators/schemas where applicable.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"
VALIDATORS_DIR = ROOT / "JSON validators"


SECTION_FRAGMENTS = [
    "PROJECT CONTEXT",
    "ROLE",
    "WORKFLOW POSITION",
    "INPUT",
    "HIGHEST-PRECEDENCE RULE",
    "DEFINITIONS",
    "WHAT YOU OWN VS WHAT YOU DO NOT OWN",
    "ALLOWED OUTPUT ENUMS",
    "ARTIFACT / INPUT SEMANTICS",
    "DECISION PROCEDURE",
    "EXAMPLES",
    "OUTPUT SCHEMA",
    "FINAL OUTPUT CONSTRAINTS",
]


@dataclass(frozen=True)
class EnumSource:
    validator_file: str
    constant_names: tuple[str, ...]


@dataclass(frozen=True)
class PromptSpec:
    min_example_headers: int
    semantics_terms: tuple[str, ...]
    enum_sources: tuple[EnumSource, ...] = ()


PROMPT_SPECS: dict[str, PromptSpec] = {
    "type_transform_worker_system_prompt.md": PromptSpec(
        min_example_headers=6,
        semantics_terms=(
            "light_contract_decisions",
            "semantic_context_json",
            "A2",
            "A3-T",
            "A3-V",
            "A4",
            "A9",
            "A13",
            "A14",
            "A16",
        ),
        enum_sources=(
            EnumSource(
                "type_validator.json",
                (
                    "ALLOWED_LOGICAL_TYPES",
                    "ALLOWED_STORAGE_TYPES",
                    "ALLOWED_TRANSFORM_ACTIONS",
                    "ALLOWED_STRUCTURAL_HINTS",
                    "ALLOWED_INTERPRETATION_HINTS",
                ),
            ),
        ),
    ),
    "missingness_worker_system_prompt.md": PromptSpec(
        min_example_headers=4,
        semantics_terms=(
            "light_contract_decisions",
            "semantic_context_json",
            "A2",
            "A4",
            "A13",
            "A14",
            "A16",
        ),
        enum_sources=(
            EnumSource(
                "missingness_validator.json",
                (
                    "ALLOWED_MISSINGNESS_DISPOSITIONS",
                    "ALLOWED_STRUCTURAL_VALIDITY",
                    "ALLOWED_RECOMMENDED_HANDLING",
                ),
            ),
        ),
    ),
    "grain_worker_system_prompt.md": PromptSpec(
        min_example_headers=4,
        semantics_terms=("A5", "A6", "A7", "A8", "A9", "A10"),
        enum_sources=(
            EnumSource(
                "grain_validator.json",
                (
                    "ALLOWED_GRAIN_TYPES",
                    "ALLOWED_REFERENCE_KINDS",
                    "ALLOWED_RELATIONSHIPS",
                    "ALLOWED_TABLE_STATUSES",
                    "ALLOWED_FAMILY_STATUSES",
                    "ALLOWED_ANSWER_TYPES",
                    "ALLOWED_ENCODING_HINTS",
                ),
            ),
        ),
    ),
    "semantic_context_interpreter_system_prompt.md": PromptSpec(
        min_example_headers=5,
        semantics_terms=(
            "light_contract_decisions",
            "semantic_context_input.dataset_context_and_collection_notes",
            "semantic_context_input.semantic_codebook_and_important_variables",
            "A2",
            "A8",
            "A9",
            "A16",
        ),
        enum_sources=(
            EnumSource(
                "semantic_context_validator.json",
                ("ALLOWED_KINDS", "ALLOWED_SKIP_REASONS"),
            ),
        ),
    ),
    "family_worker_system_prompt.md": PromptSpec(
        min_example_headers=5,
        semantics_terms=(
            "light_contract_decisions",
            "semantic_context_json",
            "family_decision",
            "a8_family_index",
            "b1_family_packet",
            "type_context_for_family",
            "missingness_context_for_family",
            "family_member_columns",
        ),
        enum_sources=(
            EnumSource(
                "family_validator.json",
                (
                    "ALLOWED_FAMILY_ROLES",
                    "ALLOWED_HANDLING",
                    "TYPE_LOGICAL_TYPES",
                    "TYPE_STORAGE_TYPES",
                    "TYPE_TRANSFORM_ACTIONS",
                    "TYPE_STRUCTURAL_HINTS",
                    "TYPE_INTERPRETATION_HINTS",
                    "MISSINGNESS_DISPOSITIONS",
                    "MISSINGNESS_HANDLING",
                ),
            ),
        ),
    ),
    "table_layout_worker_system_prompt.md": PromptSpec(
        min_example_headers=5,
        semantics_terms=(
            "light_contract_decisions",
            "semantic_context_json",
            "type_transform_worker_json",
            "missingness_worker_json",
            "family_worker_json",
            "A2",
            "A5",
            "A9",
            "A10",
            "A12",
            "A14",
        ),
        enum_sources=(
            EnumSource(
                "table_layout_validator.json",
                (
                    "MODEL_SHAPES",
                    "TABLE_KINDS",
                    "SOURCE_KINDS",
                    "BUILD_STRATEGIES",
                    "ASSIGNMENT_ROLES",
                ),
            ),
        ),
    ),
    "analysis_layout_worker_system_prompt.md": PromptSpec(
        min_example_headers=5,
        semantics_terms=(
            "light_contract_decisions",
            "semantic_context_json",
            "type_transform_worker_json",
            "missingness_worker_json",
            "family_worker_json",
            "table_layout_worker_json",
            "A2",
            "A8",
            "A10",
            "A14",
            "A16",
            "B1",
        ),
        enum_sources=(
            EnumSource(
                "analysis_layout_validator.json",
                (
                    "TABLE_KINDS",
                    "BUILD_STRATEGIES",
                    "DERIVATION_KINDS",
                    "NULL_HANDLING_POLICIES",
                ),
            ),
        ),
    ),
    "canonical_contract_reviewer_system_prompt.md": PromptSpec(
        min_example_headers=6,
        semantics_terms=(
            "canonical_column_contract_json",
            "light_contract_decisions",
            "semantic_context_json",
            "type_transform_worker_json",
            "missingness_worker_json",
            "family_worker_json",
            "table_layout_worker_json",
            "A2",
            "A3-T",
            "A3-V",
            "A4",
            "A9",
            "A13",
            "A14",
            "A16",
            "A17",
        ),
        enum_sources=(
            EnumSource(
                "canonical_contract_reviewer_validator.json",
                (
                    "CANONICAL_MODELING_STATUSES",
                    "CANONICAL_ASSIGNMENT_ROLES",
                    "TYPE_LOGICAL_TYPES",
                    "TYPE_STORAGE_TYPES",
                    "TYPE_TRANSFORM_ACTIONS",
                    "TYPE_STRUCTURAL_HINTS",
                    "TYPE_INTERPRETATION_HINTS",
                    "MISSINGNESS_DISPOSITIONS",
                    "MISSINGNESS_HANDLING",
                    "TYPE_DECISION_SOURCES",
                    "STRUCTURE_DECISION_SOURCES",
                    "MISSINGNESS_DECISION_SOURCES",
                    "SEMANTIC_DECISION_SOURCES",
                ),
            ),
        ),
    ),
}


def load_validator_sets(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text())
    result: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if isinstance(node.value, ast.Set):
            values: set[str] = set()
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    values.add(elt.value)
            if values:
                result[target.id] = values
    return result


def section_map(text: str) -> dict[str, str]:
    headings = list(re.finditer(r"^## [^\n]+$", text, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(headings):
        start = match.start()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        sections[match.group(0)] = text[start:end]
    return sections


def find_section_text(text: str, fragment: str) -> str:
    for heading, body in section_map(text).items():
        if fragment in heading:
            return body
    return ""


def count_example_headers(text: str) -> int:
    return len(re.findall(r"^### (Example|Case)\b", text, flags=re.MULTILINE))


def audit_prompt(path: Path, spec: PromptSpec) -> list[str]:
    errors: list[str] = []
    text = path.read_text()

    headings = re.findall(r"^## [^\n]+$", text, flags=re.MULTILINE)
    for fragment in SECTION_FRAGMENTS:
        if not any(fragment in heading for heading in headings):
            errors.append(f"missing section containing '{fragment}'")

    semantics_text = find_section_text(text, "ARTIFACT / INPUT SEMANTICS")
    if not semantics_text:
        errors.append("missing 'ARTIFACT / INPUT SEMANTICS' section")
    else:
        for term in spec.semantics_terms:
            if term not in semantics_text:
                errors.append(f"artifact/input semantics section does not mention '{term}'")

    example_count = count_example_headers(text)
    if example_count < spec.min_example_headers:
        errors.append(
            f"example coverage too small: found {example_count} example headers, expected at least {spec.min_example_headers}"
        )

    for enum_source in spec.enum_sources:
        validator_sets = load_validator_sets(VALIDATORS_DIR / enum_source.validator_file)
        for constant_name in enum_source.constant_names:
            allowed = validator_sets.get(constant_name)
            if not allowed:
                errors.append(f"could not load enum set '{constant_name}' from {enum_source.validator_file}")
                continue
            for literal in sorted(allowed):
                if literal not in text:
                    errors.append(f"missing enum literal '{literal}' from {constant_name}")

    return errors


def main() -> int:
    failures: list[str] = []

    for filename, spec in PROMPT_SPECS.items():
        path = PROMPTS_DIR / filename
        if not path.exists():
            failures.append(f"{filename}: file is missing")
            continue
        errors = audit_prompt(path, spec)
        for error in errors:
            failures.append(f"{filename}: {error}")

    if failures:
        print("Prompt audit failed:\n")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Prompt audit passed for all configured system prompts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
