#!/usr/bin/env bash
set -euo pipefail

# CCPM Update Script
# Run from any project's root directory to update its CCPM system files.
#
# Usage:
#   /path/to/CCPM/.claude/scripts/update-ccpm.sh [--dry-run]
#
# Example:
#   cd ~/projects/my-app
#   ~/Desktop/CCPM/.claude/scripts/update-ccpm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CCPM_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET_ROOT="$(pwd)"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

# Safety: don't update CCPM itself
if [[ "$TARGET_ROOT" == "$CCPM_ROOT" ]]; then
  echo "❌ Cannot update CCPM from itself. Run this from your target project directory."
  exit 1
fi

# Verify target has a .claude directory
if [[ ! -d "$TARGET_ROOT/.claude" ]]; then
  echo "No .claude directory found in $TARGET_ROOT"
  echo ""
  read -rp "Create .claude and initialize CCPM? (y/n) " answer
  if [[ "$answer" != "y" ]]; then
    echo "Aborted."
    exit 0
  fi
  mkdir -p "$TARGET_ROOT/.claude"
fi

# System directories to sync (these are safe to overwrite)
SYNC_DIRS=(agents commands rules scripts)

# Files to never overwrite in the target
# (epics, prds, archived, and project CLAUDE.md are left alone)

echo "CCPM Update"
echo "  Source: $CCPM_ROOT"
echo "  Target: $TARGET_ROOT"
echo ""

if $DRY_RUN; then
  echo "  Mode: DRY RUN (no changes will be made)"
  echo ""
fi

RSYNC_FLAGS="-av"
if $DRY_RUN; then
  RSYNC_FLAGS="$RSYNC_FLAGS --dry-run"
fi

changed=0

for dir in "${SYNC_DIRS[@]}"; do
  src="$CCPM_ROOT/.claude/$dir/"
  dst="$TARGET_ROOT/.claude/$dir/"

  if [[ ! -d "$src" ]]; then
    continue
  fi

  echo "--- .claude/$dir/ ---"
  # shellcheck disable=SC2086
  output=$(rsync $RSYNC_FLAGS "$src" "$dst" 2>&1)
  echo "$output"

  # Count actual file changes (lines that don't start with building/sent/total/blank)
  file_changes=$(echo "$output" | grep -cvE '^(building|sending|sent |total |$|\./)' || true)
  changed=$((changed + file_changes))
  echo ""
done

if $DRY_RUN; then
  echo "Dry run complete. $changed file(s) would change."
  echo "Run without --dry-run to apply."
else
  echo "Update complete. $changed file(s) synced."
fi
