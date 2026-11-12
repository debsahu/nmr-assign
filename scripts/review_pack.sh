#!/bin/sh
# Clean-room external review bundle generator for nmr-assign (POSIX sh).
# Packages: (1) adversarial review prompt for Claude, (2) Task card from
# docs/roadmap/MASTER_ROADMAP.md, (3) AGENTS.md clean-room & scientific audit
# checklist + automated pre-flight scan, and (4) git log --stat + git diff.

set -eu

usage() {
    cat <<'EOF'
Usage:
  sh scripts/review_pack.sh [--task Txx.y] [<git_range>] [--out review_pack.md]

Options:
  -t, --task <ID>     Include atomic Task Card <ID> (e.g. T08.3, T09.4, T13.3)
                      from docs/roadmap/MASTER_ROADMAP.md.
  -o, --out <PATH>    Write Markdown review bundle to <PATH> (use '-' or omit
                      to write to stdout, e.g. for '| pbcopy' on macOS).
  -h, --help          Show this help message and exit.

Arguments:
  <git_range>         Git revision range to audit (default: origin/main..HEAD,
                      falling back to main..HEAD if origin/main is absent).

Examples:
  sh scripts/review_pack.sh --task T08.3 HEAD~1..HEAD --out /tmp/review_T08.3.md
  sh scripts/review_pack.sh --task T08.3 HEAD~1..HEAD | pbcopy
  sh scripts/review_pack.sh --task T09.4 --out review_pack.md
  sh scripts/review_pack.sh v0.3.0..HEAD --out /tmp/review_phase08.md
EOF
}

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
ROADMAP_FILE="$REPO_ROOT/docs/roadmap/MASTER_ROADMAP.md"
AGENTS_FILE="$REPO_ROOT/AGENTS.md"

TASK_ID=""
OUT_PATH=""
GIT_RANGE=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -t|--task)
            if [ "$#" -lt 2 ]; then
                echo "error: --task requires a task ID argument (e.g. T08.3)" >&2
                exit 1
            fi
            TASK_ID="$2"
            shift 2
            ;;
        --task=*)
            TASK_ID="${1#--task=}"
            shift
            ;;
        -o|--out)
            if [ "$#" -lt 2 ]; then
                echo "error: --out requires a file path argument" >&2
                exit 1
            fi
            OUT_PATH="$2"
            shift 2
            ;;
        --out=*)
            OUT_PATH="${1#--out=}"
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "error: unknown option '$1' (run with --help for usage)" >&2
            exit 1
            ;;
        *)
            if [ -n "$GIT_RANGE" ]; then
                echo "error: unexpected extra positional argument '$1' (git_range already set to '$GIT_RANGE')" >&2
                exit 1
            fi
            GIT_RANGE="$1"
            shift
            ;;
    esac
done

if [ "$#" -gt 0 ]; then
    if [ -z "$GIT_RANGE" ] && [ "$#" -eq 1 ]; then
        GIT_RANGE="$1"
    else
        echo "error: unexpected trailing arguments: $*" >&2
        exit 1
    fi
fi

if [ ! -f "$ROADMAP_FILE" ]; then
    echo "error: roadmap file not found at $ROADMAP_FILE" >&2
    exit 1
fi

if [ -z "$GIT_RANGE" ]; then
    if git -C "$REPO_ROOT" rev-parse --verify origin/main >/dev/null 2>&1; then
        GIT_RANGE="origin/main..HEAD"
    elif git -C "$REPO_ROOT" rev-parse --verify main >/dev/null 2>&1; then
        GIT_RANGE="main..HEAD"
    else
        GIT_RANGE="HEAD~1..HEAD"
    fi
fi

# Verify git range is valid in this repository.
if ! COMMIT_COUNT=$(git -C "$REPO_ROOT" rev-list --count "$GIT_RANGE" 2>/dev/null); then
    echo "error: invalid git revision range '$GIT_RANGE'" >&2
    exit 1
fi

TASK_CARD=""
NORM_TASK_ID=""
if [ -n "$TASK_ID" ]; then
    NORM_TASK_ID=$(printf '%s' "$TASK_ID" | tr '[:lower:]' '[:upper:]')
    if ! TASK_CARD=$(awk -v task_id="$NORM_TASK_ID" '
        BEGIN { found = 0; printing = 0 }
        /^### / {
            if (printing) { exit 0 }
            if (toupper($2) == task_id) {
                found = 1
                printing = 1
            }
        }
        /^## / {
            if (printing) { exit 0 }
        }
        printing { print }
        END {
            if (!found) { exit 2 }
        }
    ' "$ROADMAP_FILE"); then
        AVAILABLE_IDS=$(awk '/^### T[0-9]/ { printf "%s ", $2 }' "$ROADMAP_FILE")
        echo "error: task '$TASK_ID' not found in docs/roadmap/MASTER_ROADMAP.md" >&2
        echo "available tasks: $AVAILABLE_IDS" >&2
        exit 1
    fi
fi

TMP_OUT=$(mktemp "${TMPDIR:-/tmp}/nmrassign_review_pack.XXXXXX")
TMP_DIFF=$(mktemp "${TMPDIR:-/tmp}/nmrassign_review_diff.XXXXXX")
trap 'rm -f "$TMP_OUT" "$TMP_DIFF"' EXIT HUP INT TERM

git -C "$REPO_ROOT" diff --no-color "$GIT_RANGE" > "$TMP_DIFF"

BRANCH_NAME=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "detached")
HEAD_SHA=$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo "unknown")
UTC_NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
CHANGED_FILES=$(git -C "$REPO_ROOT" diff --name-status "$GIT_RANGE")
CHANGED_FILE_COUNT=$(git -C "$REPO_ROOT" diff --name-only "$GIT_RANGE" | wc -l | tr -d ' ')
DIFF_LINES=$(wc -l < "$TMP_DIFF" | tr -d ' ')

# Automated pre-flight checks on added diff lines & repository hygiene.
HOST_PATH_HITS=$(awk '/^\+[^+]/ && (/\/[U]sers\/[A-Za-z0-9_.-]/ || /\/[h]ome\/[A-Za-z0-9_.-]/) { c++ } END { print c + 0 }' "$TMP_DIFF")
COAUTHOR_HITS=$(git -C "$REPO_ROOT" log "$GIT_RANGE" --grep='Co-Authored-By' --oneline | wc -l | tr -d ' ')
PRIVATE_TRACKED=$(git -C "$REPO_ROOT" ls-files "data/private" | wc -l | tr -d ' ')
CARGO_TOML_TOUCHED=$(git -C "$REPO_ROOT" diff --name-only "$GIT_RANGE" | awk '/Cargo\.toml$/ { c++ } END { print c + 0 }')
THIRD_PARTY_TOUCHED=$(git -C "$REPO_ROOT" diff --name-only "$GIT_RANGE" | awk '/^THIRD_PARTY\.md$/ { c++ } END { print c + 0 }')

# Derive restricted tool-name regex from AGENTS.md Clean-room section when present.
FORBIDDEN_HITS="0"
if [ -f "$AGENTS_FILE" ]; then
    FORBIDDEN_RE=$(awk '
        /^Forbidden:/ { flag = 1; next }
        /^Provenance:/ { flag = 0 }
        flag { print }
    ' "$AGENTS_FILE" | tr -cs '[:alnum:]-' '\n' | awk '
        /^[A-Z][A-Za-z0-9-]*$/ && length($0) >= 4 && $0 !~ /^(Source|Reading|Using|Work|Compatibility|GPL|AGPL|Picker)$/ {
            printf "%s|", tolower($0)
        }
    ' | sed 's/|$//')
    if [ -n "$FORBIDDEN_RE" ]; then
        FORBIDDEN_HITS=$(awk '/^\+[^+]/ { print tolower($0) }' "$TMP_DIFF" | grep -E -w "$FORBIDDEN_RE" | wc -l | tr -d ' ' || true)
    fi
fi

{
    cat <<EOF
# nmr-assign External Adversarial Review Bundle

- **Generated (UTC)**: \`$UTC_NOW\`
- **Branch / HEAD**: \`$BRANCH_NAME\` @ \`$HEAD_SHA\`
- **Evaluated Git Range**: \`$GIT_RANGE\` (\`$COMMIT_COUNT\` commit(s), \`$CHANGED_FILE_COUNT\` file(s) changed, \`$DIFF_LINES\` diff lines)
EOF
    if [ -n "$NORM_TASK_ID" ]; then
        printf -- '- **Target Roadmap Task**: `%s`\n' "$NORM_TASK_ID"
    else
        printf -- '- **Target Roadmap Task**: *(multi-task / phase range — pass `--task Txx.y` for a single task card)*\n'
    fi

    cat <<'EOF'

---

## 1. Adversarial Review Prompt (Paste Directly into Claude)

You are an independent **biomolecular NMR spectroscopist, numerical scientific-computing engineer (Rust + WebGPU/WGSL), and clean-room IP auditor** performing an adversarial code review on `nmr-assign`.

Do **not** rubber-stamp this change. Assume subtle bugs exist until proven otherwise from the equations, types, and tests in the diff below. Specifically audit:

1. **Physical & Spectroscopic Math**:
   - Sign conventions in quadrature recombination (`States`, `States-TPPI`, `Echo-Antiecho` / Rance–Kay), digital filter group-delay removal + sub-sample linear phase, FFT direction & Nyquist `fftshift`, `REVERSE` axis storage, chemical-shift referencing (`ppm` <-> index off-by-one or `N` vs `N - 1`), scalar couplings, relaxation rates, or Bloch–McConnell / Saupe matrices.
   - **Field-strength invariance (`500–950 MHz`)**: verify physical quantities in `Hz` ($^3J$, linewidths, digital resolution `Hz/pt`, sinc-sidelobe floors) convert to `ppm` dynamically via `obs_mhz` rather than hardcoded `500 MHz` `ppm` constants.
2. **Statistical Hygiene & Zero Data Leakage**:
   - **Unlabelled-only reference offsets**: global `(dH, dN, dCA, dCB, dC)` offsets must depend *only* on unlabelled empirical shift sets vs sequence-weighted priors (never matched `(pred_k, truth_k)` pairs, and must remain invariant to arbitrary residue-label permutations).
   - **Sequence-only assignment & homolog firewall**: backbone assignment (`assign`, `pipeline`, `benchmark`, `real-benchmark`) must never consult PDB coordinates; any empirical torsion/fragment database must run strictly post-assignment and enforce `< 35%` sequence identity + zero shared 7-mer peptides; peak picking and NUS IST/FISTA reconstruction must never peek at sequence or BMRB shift priors (zero confirmation bias).
   - **Confidence calibration & holdout firewall**: verify high-confidence (`>= 0.80`) estimates account for whole-fragment register ambiguity on unanchored short stretches, and confirm zero hyperparameters were tuned on `heldout` datasets.
3. **Systems, Cross-Platform & WebGPU Engineering**:
   - Linux `ext4` case-sensitive filesystem safety when opening spectrometer files (`ACQUS`/`SER` vs `acqus`/`ser`), archive path-traversal/symlink safety, memory bounds on 3D/4D cubes, `wasm32-unknown-unknown` compatibility, numerical stability (`NaN`/`Inf` / zero-division guards), and loud `--engine gpu` errors (zero silent CPU fallbacks).
4. **Clean-Room & Scope Compliance (`AGENTS.md`)**:
   - Verify module provenance docstrings cite peer-reviewed literature / public format specs, only task-owned files are modified, no unauthorized dependencies are introduced, and compatibility statements are phrased strictly as true comparisons without forbidden tool names in modules/files/identifiers.

### Required Review Output Format
1. **Verdict**: `APPROVE`, `APPROVE WITH MINOR FOLLOW-UPS`, or `BLOCK`.
2. **Blocking Defects (`P0` / `P1`)**: For each defect, cite exact `file:line`, explain the mathematical/physical/engineering failure mode, give a concrete counterexample (e.g. `800 MHz`, `REVERSE=yes`, uppercase `ACQUS`, degenerate `CA(i) ~ CA(i-1)`, `[2Fe-2S]` bleaching), and specify the minimal clean fix.
3. **Non-Blocking Improvements & Edge Cases (`P2`)**: Calibration, performance, or test-coverage gaps worth tracking in `TODO.md`.
4. **Task Card Done-Gate Audit**: Explicitly check each requirement in Section 2 against the implementation and tests in Section 5.

---

## 2. Target Task Card (`docs/roadmap/MASTER_ROADMAP.md`)

EOF

    if [ -n "$TASK_CARD" ]; then
        printf '%s\n' "$TASK_CARD"
    else
        cat <<'EOF'
*(No `--task Txx.y` flag was provided; auditing range across tasks. Pass `--task Txx.y` to embed a specific atomic task contract.)*
EOF
    fi

    cat <<EOF

---

## 3. Automated Pre-Flight Scan & \`AGENTS.md\` Audit Checklist

### 3.1 Automated Diff Pre-Flight Summary
| Check | Result | Notes |
|---|---:|---|
| Commits in \`$GIT_RANGE\` | \`$COMMIT_COUNT\` | Evaluated revision range |
| Changed files | \`$CHANGED_FILE_COUNT\` | Compare against **Owned Files/Modules** in Section 2 |
| Hardcoded host user-home paths in \`+\` lines | \`$HOST_PATH_HITS\` | Must be \`0\` in committed code/docs |
| Restricted tool-name tokens in \`+\` lines | \`$FORBIDDEN_HITS\` | Must be \`0\` in module/file/symbol names; clean comparisons only in docs |
| Tracked files under \`data/private/\` | \`$PRIVATE_TRACKED\` | Must be \`0\` |
| \`Cargo.toml\` touched / \`THIRD_PARTY.md\` touched | \`$CARGO_TOML_TOUCHED\` / \`$THIRD_PARTY_TOUCHED\` | Any new crate requires MIT/BSD/Apache-2.0/ISC/Zlib + \`THIRD_PARTY.md\` |
| \`Co-Authored-By\` trailers in range | \`$COAUTHOR_HITS\` | Swarm convention: \`0\` |

#### Changed Files (\`git diff --name-status $GIT_RANGE\`)
\`\`\`text
$CHANGED_FILES
\`\`\`

### 3.2 Manual / LLM Scientific & Clean-Room Checklist
- [ ] **Clean-room provenance**: New algorithm modules begin with peer-reviewed paper / public format citations; zero proprietary or GPL/AGPL code/macros/tables consulted.
- [ ] **Maintainer ownership**: \`docs/specs/03-assignment.md\` §3–§4 scoring semantics untouched unless explicitly approved by the maintainer.
- [ ] **Hz vs ppm scaling**: Frequency cutoffs and couplings scale with \`obs_mhz\` across \`500–950 MHz\`.
- [ ] **Unlabelled reference offsets**: Offset estimation uses unlabelled shift distributions only (permutation-invariant; zero paired prediction-truth residual fitting).
- [ ] **Zero leakage / confirmation bias**: Sequence-only backbone assignment; zero sequence/prior inputs into peak picking or NUS reconstruction; strict \`dev\` vs \`heldout\` separation.
- [ ] **Cross-platform & WASM**: Case-insensitive spectrometer file lookup for Linux \`ext4\`; \`wasm32-unknown-unknown\` target clean; loud GPU errors without silent CPU fallback.
- [ ] **5-seed regression gates**: \`--realistic --seeds 5\` Ubiquitin and 5-seed \`CISD3\` (\`standard\` + \`evo-graph\`) verified when solver/picker behavior changes.

---

## 4. Commit History & Diffstat (\`git log --stat $GIT_RANGE\`)

\`\`\`text
EOF
    git -C "$REPO_ROOT" log --no-color --stat "$GIT_RANGE"
    cat <<EOF
\`\`\`

---

## 5. Unified Patch (\`git diff $GIT_RANGE\`)

\`\`\`diff
EOF
    cat "$TMP_DIFF"
    cat <<'EOF'
```
EOF
} > "$TMP_OUT"

LINE_COUNT=$(wc -l < "$TMP_OUT" | tr -d ' ')
BYTE_COUNT=$(wc -c < "$TMP_OUT" | tr -d ' ')

if [ -z "$OUT_PATH" ] || [ "$OUT_PATH" = "-" ]; then
    cat "$TMP_OUT"
else
    OUT_DIR=$(dirname -- "$OUT_PATH")
    if [ ! -d "$OUT_DIR" ]; then
        mkdir -p "$OUT_DIR"
    fi
    cp "$TMP_OUT" "$OUT_PATH"
    echo "[review_pack] Wrote $LINE_COUNT lines ($BYTE_COUNT bytes) for $GIT_RANGE${NORM_TASK_ID:+ ($NORM_TASK_ID)} -> $OUT_PATH" >&2
fi
