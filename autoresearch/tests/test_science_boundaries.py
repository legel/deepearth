"""Regression tests for the probe/science/fusion ownership boundary."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


AUTORESEARCH = Path(__file__).resolve().parents[1]


class ScienceBoundaryTests(unittest.TestCase):
    def test_fixed_probes_do_not_implement_candidate_science(self):
        """Guard responsibilities, not replaceable scientific filenames or mechanisms."""
        violations = []
        fixed_probes = []
        for path in (AUTORESEARCH / "probes").rglob("*.py"):
            if "editable_files" in path.parts:
                continue
            if path.name not in {"probe.py", "traitprobe.py"}:
                continue
            tree = ast.parse(path.read_text())
            fixed_probes.append(path)
        self.assertTrue(fixed_probes, "no fixed probes were discovered")
        scientific_state = {"CONFIG", "CAPABILITY_CONFIG", "CHANNELS"}
        forbidden_calls = {
            "Earth4D", "backward", "cross_entropy", "binary_cross_entropy",
            "binary_cross_entropy_with_logits", "mse_loss",
        }
        for source in fixed_probes:
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    marker = ".probes.spacetime.editable_files"
                    if marker in node.module and node.module.split(marker, 1)[1]:
                        violations.append((source, node.lineno, "private scientific module import"))
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id in scientific_state:
                            violations.append((source, node.lineno, f"candidate state {target.id}"))
                elif isinstance(node, ast.ClassDef):
                    if any(
                        (isinstance(base, ast.Name) and base.id == "Module")
                        or (isinstance(base, ast.Attribute) and base.attr == "Module")
                        for base in node.bases
                    ):
                        violations.append((source, node.lineno, "trainable model class"))
                elif isinstance(node, ast.Call):
                    name = (node.func.id if isinstance(node.func, ast.Name)
                            else node.func.attr if isinstance(node.func, ast.Attribute) else "")
                    if name in forbidden_calls:
                        violations.append((source, node.lineno, f"candidate operation {name}"))
                    if (isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Attribute)
                            and node.func.value.attr == "optim"):
                        violations.append((source, node.lineno, "optimizer construction"))
        self.assertEqual(violations, [], f"candidate science leaked into fixed probes: {violations}")

    def test_validation_code_is_not_editable(self):
        editable_roots = sorted((AUTORESEARCH / "probes").glob("*/editable_files"))
        self.assertTrue(editable_roots, "no probe editable surfaces were discovered")
        for editable in editable_roots:
            leaks = [
                p
                for p in editable.rglob("*")
                if "__pycache__" not in p.parts
                and (p.name == "probe.py" or p.name.startswith("harness"))
                and (p.is_file() or (p.is_dir() and any(p.iterdir())))
            ]
            self.assertEqual(leaks, [], f"fixed validation leaked into {editable}: {leaks}")

    def test_downstream_code_uses_only_public_probe_science(self):
        downstream = AUTORESEARCH / "main"
        violations = []
        probe_imports = []
        for source in downstream.rglob("*.py"):
            tree = ast.parse(source.read_text())
            modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    modules.append(node.module or "")
                elif isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
            for module in (m for m in modules if ".autoresearch.probes." in m):
                probe_imports.append(module)
                suffix = module.split(".autoresearch.probes.", 1)[1]
                if ".editable_files." not in suffix or any(
                    part in suffix.split(".") for part in ("lib", "probe", "harness")
                ):
                    violations.append((source.relative_to(AUTORESEARCH), module))
        self.assertTrue(probe_imports, "main does not consume any probe-owned science")
        self.assertEqual(violations, [], f"downstream imports private validation/science: {violations}")


if __name__ == "__main__":
    unittest.main()
