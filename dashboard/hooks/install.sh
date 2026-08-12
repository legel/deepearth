#!/usr/bin/env bash
# Opt-in: refresh dashboard state after every commit (registry -> callgraph -> flow; audit
# stays manual/looped since it spends LLM tokens). Run once: bash dashboard/hooks/install.sh
set -e
REPO="$(git rev-parse --show-toplevel)"
HOOK="$REPO/.git/hooks/post-commit"
cat > "$HOOK" <<'EOF'
#!/usr/bin/env bash
# dashboard state refresh (installed by dashboard/hooks/install.sh)
REPO="$(git rev-parse --show-toplevel)"
(cd "$REPO" && python3 -m dashboard.refresh --graph-only >> /tmp/dashboard-refresh.log 2>&1 &)
EOF
chmod +x "$HOOK"
echo "installed: $HOOK (runs dashboard.refresh --no-audit in background after each commit)"
