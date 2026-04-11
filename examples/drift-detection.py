"""Memory drift detection and auto-fix.

Memories reference file paths, function names, and commands that
go stale when the codebase changes. This example shows how to
detect and fix stale memories programmatically.
"""

from engram.config import Config
from engram.store import Store
from engram.drift import run_drift_check, auto_fix_drift


def main():
    config = Config.load()
    store = Store(config)
    store.init_db()

    print("running drift check...")
    report = run_drift_check(
        store,
        search_roots=None,     # set to ["~/project/src"] to verify function names
        project_root=None,     # set to "~/project" to check package.json scripts
        check_functions=False, # set True to grep for function/class names
    )

    # print results
    print(f"\ndrift score: {report.score}/100")
    print(f"  memories checked: {report.memories_checked}")
    print(f"  claims extracted: {report.claims_extracted}")
    print(f"  claims verified:  {report.claims_verified} ({report.claims_valid} valid)")
    print(f"  stale memories:   {report.stale_memories}")

    if report.issues:
        print(f"\nissues ({len(report.issues)}):")
        for issue in report.issues[:10]:
            print(f"  [{issue.severity}] {issue.code}: {issue.message}")
            print(f"    memory: {issue.memory_preview[:80]}...")

    # auto-fix (dry run first)
    if report.issues:
        print("\ndry-run auto-fix:")
        result = auto_fix_drift(store, report, dry_run=True)
        print(f"  would invalidate: {result['invalidated']}")
        print(f"  would flag stale: {result['flagged_stale']}")
        print(f"  would forget: {result['forgotten']}")

        # uncomment to actually fix:
        # result = auto_fix_drift(store, report, dry_run=False)
        # print(f"fixed {result['total_actions']} issues")

    store.close()


if __name__ == "__main__":
    main()
