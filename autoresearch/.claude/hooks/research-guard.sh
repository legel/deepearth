#!/usr/bin/env bash
# PreToolUse guard for the /research contract.
#
# The loop's rules used to live only as prose in the slash command, where an agent can read them
# every turn and still drift past them. These are the ones that cost a real experiment when broken,
# so the harness enforces them instead of asking.
#
#   1. screen at 1000 steps   -- /research: "screen | ~24M | the loop -- every hypothesis, ~2 min warm"
#   2. instruments read-only  -- /research: main/harness/**, scoring/objective.py, tests/** are ground truth
#   3. commit before running  -- /research: "a dirty tree makes the run unreproducible"
#   4. shared prepared cache  -- /research: "Symlink, never copy" (a private copy is ~15.7 GB)
#
# deny = never legitimate. ask = legitimate at the confirmation step, but a human decides.
# Reads the PreToolUse payload on stdin, emits a permissionDecision on stdout.
set -uo pipefail

SHARED_CACHE="/workspace/lance-main-shared-cache-c69ee8c"
payload="$(cat)"
tool="$(printf '%s' "$payload" | jq -r '.tool_name // ""')"

decide() {  # $1 = allow|deny|ask, $2 = reason
  jq -nc --arg d "$1" --arg r "$2" \
    '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:$d,permissionDecisionReason:$r}}'
  exit 0
}

# ---------------------------------------------------------------- 2. instruments are read-only
# Editing these changes what a number MEANS, so every past result silently stops being comparable.
readonly_path() {
  case "$1" in
    */main/harness/*|*/scoring/objective.py|*/autoresearch/tests/*) return 0 ;;
    *) return 1 ;;
  esac
}

if [ "$tool" = "Edit" ] || [ "$tool" = "Write" ] || [ "$tool" = "NotebookEdit" ]; then
  f="$(printf '%s' "$payload" | jq -r '.tool_input.file_path // .tool_input.notebook_path // ""')"
  if [ -n "$f" ] && readonly_path "$f"; then
    decide deny "READ-ONLY INSTRUMENT: $f. /research forbids editing main/harness/**, scoring/objective.py and tests/** -- they are ground truth, and changing them changes what every recorded number means. If the instrument is wrong, publish an insight and hand it to ship-deepearth-improvement."
  fi
  decide allow ""
fi

[ "$tool" = "Bash" ] || decide allow ""
cmd="$(printf '%s' "$payload" | jq -r '.tool_input.command // ""')"

# A bash-side write to an instrument (redirect, sed -i, tee, cp) bypasses the Edit/Write check above.
if printf '%s' "$cmd" | grep -Eq '(>|>>|sed -i|tee|cp |mv )[^|;]*(main/harness/|scoring/objective\.py|autoresearch/tests/)'; then
  decide deny "READ-ONLY INSTRUMENT: this command writes to main/harness/**, scoring/objective.py or tests/**. Those are ground truth under /research. Publish an insight and hand it to ship-deepearth-improvement instead."
fi

# Everything below only concerns training runs.
printf '%s' "$cmd" | grep -Eq 'editable_files[./]train' || decide allow ""

# ---------------------------------------------------------------- 1. screen at 1000 steps
steps="$(printf '%s' "$cmd" | sed -nE 's/.*--steps[= ]+([0-9]+).*/\1/p' | head -1)"
if [ -n "$steps" ] && [ "$steps" != "1000" ]; then
  decide ask "STEP BUDGET: --steps $steps, but /research defines the screen scale as '~24M, every hypothesis, ~2 min warm' and the configs ship steps: 1000 (=130s). $steps steps is $(( steps / 1000 ))x that. Screening here costs 8x the GPU for a verdict the 1000-step screen has been measured to reach: the worst-first variable ranking is identical at both budgets. A larger budget is legitimate ONLY to confirm a candidate that already survived the 1k screen -- approve if that is what this is."
fi

# ---------------------------------------------------------------- 4. shared prepared cache
cd_arg="$(printf '%s' "$cmd" | sed -nE 's/.*--cache_dir[= ]+([^ '"'"'"]+).*/\1/p' | head -1)"
if [ -n "$cd_arg" ] && [ "$cd_arg" != "$SHARED_CACHE" ]; then
  decide deny "PREPARED CACHE: --cache_dir $cd_arg is not the shared cache $SHARED_CACHE. /research: 'Keep the prepared cache shared -- a worktree that builds its own writes ~15.7 GB and will fill the disk. Symlink, never copy.' This already killed a peer agent's run once. Drop the flag (the configs carry the shared path) or point it at the shared cache."
fi

# ---------------------------------------------------------------- 3. commit before running
# The diff IS the experiment. Runs execute in a remote worktree, so check the tree they run FROM,
# which is the PYTHONPATH root (or the cd target) in the command itself.
root="$(printf '%s' "$cmd" | sed -nE 's|.*PYTHONPATH=(/workspace/[A-Za-z0-9_.-]+).*|\1|p' | head -1)"
[ -n "$root" ] || root="$(printf '%s' "$cmd" | sed -nE 's|.*cd (/workspace/[A-Za-z0-9_.-]+).*|\1|p' | head -1)"
if [ -n "$root" ] && printf '%s' "$cmd" | grep -q '^ssh '; then
  host="$(printf '%s' "$cmd" | sed -nE 's/^ssh +([A-Za-z0-9_.-]+).*/\1/p' | head -1)"
  if [ -n "$host" ]; then
    dirty="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
              "git -C $root/deepearth status --porcelain 2>/dev/null | head -5" 2>/dev/null)"
    rc=$?
    if [ $rc -ne 0 ]; then
      # Never block on a hook malfunction -- a network blip must not stop the loop.
      printf '%s' "$(jq -nc --arg m "research-guard: could not verify $host:$root is committed (ssh rc=$rc); allowing." \
        '{systemMessage:$m,hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"allow",permissionDecisionReason:""}}')"
      exit 0
    fi
    if [ -n "$dirty" ]; then
      decide deny "DIRTY TREE at $host:$root/deepearth -- $(printf '%s' "$dirty" | tr '\n' ' '). /research: 'Commit the candidate before running it. The diff IS the experiment; a number measured against uncommitted code is unrecoverable by anyone else, including you tomorrow.' Commit, then re-run."
    fi
  fi
fi

decide allow ""
