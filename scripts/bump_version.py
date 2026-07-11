import argparse
import re
import subprocess
import sys
from pathlib import Path

def get_current_version(init_path: Path) -> str:
    """Read __version__ from __init__.py."""
    content = init_path.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError(f"Could not find __version__ in {init_path}")
    return match.group(1)

def update_files(init_path: Path, pyproject_path: Path, old_ver: str, new_ver: str):
    """Update version strings in __init__.py and pyproject.toml."""
    # 1. Update __init__.py
    init_content = init_path.read_text(encoding="utf-8")
    new_init_content = re.sub(
        rf'__version__\s*=\s*["\']{re.escape(old_ver)}["\']',
        f'__version__ = "{new_ver}"',
        init_content
    )
    init_path.write_text(new_init_content, encoding="utf-8")
    print(f"Updated {init_path.name}: {old_ver} -> {new_ver}")

    # 2. Update pyproject.toml
    pyproject_content = pyproject_path.read_text(encoding="utf-8")
    new_pyproject_content = re.sub(
        rf'version\s*=\s*["\']{re.escape(old_ver)}["\']',
        f'version = "{new_ver}"',
        pyproject_content,
        count=1
    )
    pyproject_path.write_text(new_pyproject_content, encoding="utf-8")
    print(f"Updated {pyproject_path.name}: {old_ver} -> {new_ver}")

def increment_version(version: str, bump_type: str) -> str:
    """Increment version numbers following semantic versioning rules."""
    parts = list(map(int, version.split(".")))
    if len(parts) != 3:
        raise ValueError(f"Invalid semantic version: {version}")

    if bump_type == "major":
        parts[0] += 1
        parts[1] = 0
        parts[2] = 0
    elif bump_type == "minor":
        parts[1] += 1
        parts[2] = 0
    else:  # patch
        parts[2] += 1

    return ".".join(map(str, parts))

def run_cmd(cmd: list[str], check: bool = True) -> str:
    """Run a shell command and return stdout."""
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if check and res.returncode != 0:
        print(f"Error executing: {' '.join(cmd)}")
        print(f"Stderr: {res.stderr}")
        sys.exit(res.returncode)
    return res.stdout.strip()

def main():
    parser = argparse.ArgumentParser(description="Increment package version, commit, tag, and push.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--major", action="store_true", help="Bump major version (e.g. 1.0.0)")
    group.add_argument("--minor", action="store_true", help="Bump minor version (e.g. 0.1.0)")
    group.add_argument("--patch", action="store_true", default=True, help="Bump patch version (default, e.g. 0.0.1)")
    
    parser.add_argument("--push", action="store_true", help="Push commit and tags to remote origin")
    parser.add_argument("-m", "--message", type=str, help="Optional custom commit message")
    
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    init_path = project_root / "src" / "ypotheto_compchem_mcp" / "__init__.py"
    pyproject_path = project_root / "pyproject.toml"

    # Get current version
    old_version = get_current_version(init_path)

    # Determine bump type
    bump_type = "patch"
    if args.major:
        bump_type = "major"
    elif args.minor:
        bump_type = "minor"

    # Increment version
    new_version = increment_version(old_version, bump_type)
    tag_name = f"v{new_version}"

    # Update files on disk
    update_files(init_path, pyproject_path, old_version, new_version)

    # Git Operations
    print("Staging updated files...")
    run_cmd(["git", "add", str(init_path), str(pyproject_path)])

    commit_msg = args.message or f"bump: release version {new_version}"
    print(f"Committing changes: '{commit_msg}'")
    run_cmd(["git", "commit", "-m", commit_msg])

    print(f"Tagging release: '{tag_name}'")
    run_cmd(["git", "tag", "-a", tag_name, "-m", f"Release version {new_version}"])

    if args.push:
        print("Pushing commit and tags to remote origin...")
        run_cmd(["git", "push"])
        run_cmd(["git", "push", "origin", tag_name])
        print("Successfully pushed to remote.")

    print(f"Done! Bumped from {old_version} to {new_version} ({tag_name}).")

if __name__ == "__main__":
    main()
