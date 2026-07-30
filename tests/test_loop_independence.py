"""Each autoresearch loop must stand alone.

A loop owns one probe, its data, and its own evals. If loop A imports loop B's code, then a change in B
silently moves A's numbers and no record says so -- the spacetime probe used to import load_trait and
load_species from the biological probe exactly that way.

Sharing is allowed only DOWNWARD, into code no loop owns: encoders/, and (for the fusion loop only)
core/. Anything else is a copy, deliberately.
"""
import ast
import unittest
from pathlib import Path

AUTORESEARCH = Path(__file__).resolve().parents[1] / "autoresearch"
LOOPS = ("main", "biological", "spacetime")


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
    def test_no_loop_imports_another_loops_code(self):
        offenders = []
        for loop in LOOPS:
            root = AUTORESEARCH / loop
            if not root.exists():
                continue
            for path in root.rglob("*.py"):
                if "__pycache__" in str(path):
                    continue
                for module in imported_modules(path):
                    for other in LOOPS:
                        if other != loop and f"autoresearch.{other}." in module:
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

        `editable_files/data` is the DATA lever — sources added, moved and removed by the signal they
        provide. `records/` sits OUTSIDE editable_files because it is the one thing an experiment must
        not touch: hand-editing a score forges a result.
        """
        for loop in LOOPS:
            root = AUTORESEARCH / loop
            for required in ("program", "editable_files", "editable_files/lib",
                             "editable_files/data", "records"):
                self.assertTrue((root / required).is_dir(), f"{loop}/ is missing {required}/")
            # The harness may be ONE FILE (the target shape — spacetime is there) or still a package.
            harness = root / "editable_files" / "harness"
            self.assertTrue(harness.with_suffix(".py").is_file() or harness.is_dir(),
                            f"{loop}/ has no harness")

    def test_every_loop_states_its_own_program(self):
        programs = {"main": "autoresearch.md", "biological": "program.md", "spacetime": "program.md"}
        for loop, name in programs.items():
            self.assertTrue((AUTORESEARCH / loop / "program" / name).is_file(),
                            f"{loop}/program/{name} is missing — a loop without a program has no objective")


if __name__ == "__main__":
    unittest.main()


class RecordPathTests(unittest.TestCase):
    """The board must resolve inside its own loop.

    A parents[] off-by-one once pointed RECORDS at `autoresearch/records/records.json` instead of
    `autoresearch/spacetime/records/records.json`. trace.py then created a fresh empty board, found no
    prior record, and reported "RECORD = YES (new best!) prev_record = None" for a run that had beaten
    nothing. A path bug that silently mints records is worth a test.
    """

    def test_spacetime_board_resolves_inside_its_loop(self):
        import importlib
        module = importlib.import_module(
            "deepearth.autoresearch.spacetime.editable_files.harness")
        expected = AUTORESEARCH / "spacetime" / "records" / "records.json"
        self.assertEqual(module.RECORDS.resolve(), expected.resolve(),
                         f"board resolved to {module.RECORDS} — outside its loop")
