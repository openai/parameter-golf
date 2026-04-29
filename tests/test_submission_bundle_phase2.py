import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_ROOT = (
    REPO_ROOT
    / "records"
    / "track_non_record_16mb"
    / "2026-05-02_Framework_Legal_ScoreFirst_PPMD_Mixtures"
)


EXPECTED_COPIES = {
    "docs/legality/ppmd-legality-proof.md": "plans/ppmd-legality-proof.md",
    "docs/legality/ppmd-legality-proof-result.md": "plans/ppmd-legality-proof-result.md",
    "docs/legality/ppmd-legality-proof-implementation.md": "plans/ppmd-legality-proof-implementation.md",
    "docs/legality/ppm_notes.md": "plans/ppm_notes.md",
    "docs/plans/path-a-ppmd-eval-plan.md": "plans/path-a-ppmd-eval-plan.md",
    "docs/plans/path-a-ppmd-eval-complete.md": "plans/path-a-ppmd-eval-complete.md",
    "docs/plans/path-a-ppmd-cpp-backend-plan.md": "plans/path-a-ppmd-cpp-backend-plan.md",
    "docs/plans/path-a-ppmd-cpp-backend-complete.md": "plans/path-a-ppmd-cpp-backend-complete.md",
    "docs/plans/path-a-ppmd-cuda-backend-plan.md": "plans/path-a-ppmd-cuda-backend-plan.md",
    "docs/plans/path-a-ppmd-cuda-runpod-execution-plan.md": "plans/path-a-ppmd-cuda-runpod-execution-plan.md",
    "docs/plans/path-b-byte-eval-plan.md": "plans/path-b-byte-eval-plan.md",
    "docs/plans/path-b-byte-eval-complete.md": "plans/path-b-byte-eval-complete.md",
    "docs/plans/path-b-byte-eval-redteam.md": "plans/path-b-byte-eval-redteam.md",
    "docs/plans/path-b-byte-eval-runpod-prompt.md": "plans/path-b-byte-eval-runpod-prompt.md",
    "tests/test_exp1876_ppmd_legality_audit.py": "tests/test_exp1876_ppmd_legality_audit.py",
    "tests/test_path_a_ppmd_eval.py": "tests/test_path_a_ppmd_eval.py",
    "tests/test_path_b_ppmd_eval.py": "tests/test_path_b_ppmd_eval.py",
    "scripts/eval_path_a_ppmd.py": "scripts/eval_path_a_ppmd.py",
    "scripts/eval_path_b_ppmd.py": "scripts/eval_path_b_ppmd.py",
    "audits/exp_1876_ppmd/static_provenance.json": "audits/exp_1876_ppmd/audit_outputs/static_provenance.json",
    "audits/exp_1876_ppmd/denominator_audit.json": "audits/exp_1876_ppmd/audit_outputs/denominator_audit.json",
    "audits/exp_1876_ppmd/coverage_audit.json": "audits/exp_1876_ppmd/audit_outputs/coverage_audit.json",
    "audits/exp_1876_ppmd/normalization_audit.json": "audits/exp_1876_ppmd/audit_outputs/normalization_audit.json",
    "results/exp_1876_ppmd/path_b_prod_8gpu_local_score/path_b_sliding_subset_8000000.json": "results/exp_1876_ppmd/path_b_prod_8gpu_local_score/path_b_sliding_subset_8000000.json",
    "results/exp_1876_ppmd/path_b_prod_8gpu/path_b_sliding_accounting_audit.json": "results/exp_1876_ppmd/path_b_prod_8gpu/path_b_sliding_accounting_audit.json",
    "results/exp_1876_ppmd/path_b_prod_8gpu/path_b_sliding_merge_manifest.json": "results/exp_1876_ppmd/path_b_prod_8gpu/path_b_sliding_merge_manifest.json",
    "results/exp_1876_ppmd/path_b_prod_8gpu_fullval/path_b_sliding_accounting_audit.json": "results/exp_1876_ppmd/path_b_prod_8gpu_fullval/path_b_sliding_accounting_audit.json",
    "results/exp_1876_ppmd/path_b_prod_8gpu_fullval/path_b_sliding_merge_manifest.json": "results/exp_1876_ppmd/path_b_prod_8gpu_fullval/path_b_sliding_merge_manifest.json",
    "results/exp_1876_ppmd/prod_8gpu_s42v2/artifacts.txt": "results/exp_1876_ppmd/prod_8gpu_s42v2/artifacts.txt",
    "results/exp_1876_ppmd/prod_8gpu_s42v2/launcher_state.json": "results/exp_1876_ppmd/prod_8gpu_s42v2/launcher_state.json",
}


class SubmissionBundlePhase2Test(unittest.TestCase):
    def test_expected_directories_exist(self):
        for relpath in (
            "docs",
            "docs/legality",
            "docs/plans",
            "tests",
            "scripts",
            "audits",
            "audits/exp_1876_ppmd",
            "results",
            "results/exp_1876_ppmd",
            "results/exp_1876_ppmd/path_b_prod_8gpu_local_score",
            "results/exp_1876_ppmd/path_b_prod_8gpu",
            "results/exp_1876_ppmd/path_b_prod_8gpu_fullval",
            "results/exp_1876_ppmd/prod_8gpu_s42v2",
        ):
            with self.subTest(relpath=relpath):
                self.assertTrue((SUBMISSION_ROOT / relpath).is_dir(), relpath)

    def test_required_files_are_copied_verbatim(self):
        for bundled_relpath, source_relpath in EXPECTED_COPIES.items():
            bundled_path = SUBMISSION_ROOT / bundled_relpath
            source_path = REPO_ROOT / source_relpath
            with self.subTest(bundled_relpath=bundled_relpath):
                self.assertTrue(bundled_path.is_file(), bundled_relpath)
                self.assertTrue(source_path.is_file(), source_relpath)
                self.assertEqual(bundled_path.read_bytes(), source_path.read_bytes())

    def test_readme_references_bundled_paths_and_updated_path_b_status(self):
        readme = (SUBMISSION_ROOT / "README.md").read_text(encoding="utf-8")

        for snippet in (
            "docs/legality/ppmd-legality-proof.md",
            "docs/plans/path-a-ppmd-eval-plan.md",
            "scripts/eval_path_a_ppmd.py",
            "scripts/eval_path_b_ppmd.py",
            "tests/test_exp1876_ppmd_legality_audit.py",
            "tests/test_path_a_ppmd_eval.py",
            "tests/test_path_b_ppmd_eval.py",
            "audits/exp_1876_ppmd/static_provenance.json",
            "results/exp_1876_ppmd/path_b_prod_8gpu_local_score/path_b_sliding_subset_8000000.json",
            "results/exp_1876_ppmd/path_b_prod_8gpu_fullval/path_b_sliding_accounting_audit.json",
            "results/exp_1876_ppmd/path_b_prod_8gpu_fullval/path_b_sliding_merge_manifest.json",
            "full-val sliding-window shard-generation artifacts already exist and are now bundled",
            "offline CPU merge/PPM-D postpass",
        ):
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, readme)

        self.assertNotIn(
            "requires rerunning the GPU sliding-window shard generation across the full validation split",
            readme,
        )


if __name__ == "__main__":
    unittest.main()