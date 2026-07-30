"""Each autoresearch loop must stand alone.

A loop owns one probe, its data, and its own evals. If loop A imports loop B's code, then a change in B
silently moves A's numbers and no record says so -- the spacetime probe used to import load_trait and
load_species from the biological probe exactly that way.

Sharing is allowed only DOWNWARD, into code no loop owns: encoders/, and (for the fusion loop only)
core/. Anything else is a copy, deliberately.
"""
import ast
import os
import unittest
from pathlib import Path

AUTORESEARCH = Path(__file__).resolve().parents[1]        # this file lives in autoresearch/tests/
PROBES = ("probes/biological", "probes/spacetime")     # leaves: independent, siblings
FUSION = "main"                                        # apex: consumes probe results, runs last
LOOPS = PROBES + (FUSION,)


def imported_modules(path):
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return set()
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
    return found


class LoopIndependenceTests(unittest.TestCase):
    def test_the_dependency_graph_is_a_dag_leaves_to_apex(self):
        """encoders are leaves, probe loops sit above them, `main` (fusion) is the apex.

        Only one direction is legitimate: fusion will eventually consume each probe's finished encoder.
        A probe importing a SIBLING probe, or a probe importing fusion, is a cycle or a hidden coupling.
        The consolidation of encoders into their probe loops is deferred until the science is filled out;
        until then encoders/ is shared downward, which no rule here forbids.
        """
        for probe in PROBES:
            for path in (AUTORESEARCH / probe).rglob("*.py"):
                if "__pycache__" in str(path) or f"{os.sep}tests{os.sep}" in str(path):
                    continue
                for module in imported_modules(path):
                    self.assertNotIn("autoresearch.main", module,
                                     f"{path.name}: a probe must not depend on the fusion loop it feeds")

    def test_no_probe_imports_a_SIBLING_probe(self):
        """Siblings must not touch each other. `main` importing a probe's finished encoder is the ONE
        legitimate edge (see the DAG test) and is not flagged here. A loop's own tests may reach across
        to assert the DAG itself, so tests/ is excluded."""
        offenders = []
        for probe in PROBES:
            root = AUTORESEARCH / probe
            if not root.exists():
                continue
            for path in root.rglob("*.py"):
                if "__pycache__" in str(path) or f"{os.sep}tests{os.sep}" in str(path):
                    continue
                for module in imported_modules(path):
                    for sibling in PROBES:
                        mod_sibling = "autoresearch." + sibling.replace("/", ".") + "."
                        if sibling != probe and mod_sibling in module:
                            offenders.append(f"{path.relative_to(AUTORESEARCH)} imports {module}")
        self.assertEqual(offenders, [], "cross-loop imports break loop independence:\n  "
                                       + "\n  ".join(offenders))

    def test_only_the_fusion_loop_touches_the_fusion_model(self):
        """A probe loop recovers signal for the fusion layer; it never trains or imports it."""
        offenders = []
        for loop in ("biological", "spacetime"):
            root = AUTORESEARCH / loop
            for path in root.rglob("*.py"):
                if "__pycache__" in str(path):
                    continue
                for module in imported_modules(path):
                    if "deepearth.autoresearch.main.editable_files.fusion.fusion" in module:
                        offenders.append(str(path.relative_to(AUTORESEARCH)))
        self.assertEqual(offenders, [], "probe loops must not import the fusion model: " + str(offenders))

    def test_every_loop_has_the_same_directories(self):
        """Identical layout in every loop, so scope is never ambiguous.

        `records/` sits outside `editable_files/` because it is the one thing an experiment must not
        touch: hand-editing a score forges a result. The corpus is NOT per-loop — one
        `autoresearch/data/` with a directory per source, because three empty per-loop data directories
        held nothing but a README.
        """
        self.assertTrue((AUTORESEARCH / "data").is_dir(), "the shared corpus autoresearch/data/ is missing")
        for loop in LOOPS:
            root = AUTORESEARCH / loop
            for required in ("program", "editable_files", "editable_files/lib", "records"):
                self.assertTrue((root / required).is_dir(), f"{loop}/ is missing {required}/")
            # The harness may be ONE FILE (the target shape — spacetime is there) or still a package.
            harness = root / "editable_files" / "harness"
            self.assertTrue(harness.with_suffix(".py").is_file() or harness.is_dir(),
                            f"{loop}/ has no harness")

    def test_every_loop_states_its_own_program(self):
        programs = {"main": "autoresearch.md", "probes/biological": "program.md",
                    "probes/spacetime": "program.md"}
        for loop, name in programs.items():
            self.assertTrue((AUTORESEARCH / loop / "program" / name).is_file(),
                            f"{loop}/program/{name} is missing — a loop without a program has no objective")


if __name__ == "__main__":
    unittest.main()


class RecordPathTests(unittest.TestCase):
    """The board must resolve inside its own loop.

    A parents[] off-by-one once pointed RECORDS at `autoresearch/records/records.json` instead of
    `autoresearch/probes/spacetime/records/records.json`. trace.py then created a fresh empty board, found no
    prior record, and reported "RECORD = YES (new best!) prev_record = None" for a run that had beaten
    nothing. A path bug that silently mints records is worth a test.
    """

    def test_spacetime_board_resolves_inside_its_loop(self):
        import importlib
        module = importlib.import_module(
            "deepearth.autoresearch.probes.spacetime.editable_files.harness")
        expected = AUTORESEARCH / "probes" / "spacetime" / "records" / "records.json"
        self.assertEqual(module.RECORDS.resolve(), expected.resolve(),
                         f"board resolved to {module.RECORDS} — outside its loop")
