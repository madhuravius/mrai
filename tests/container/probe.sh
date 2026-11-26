#!/bin/sh
# Runs inside the mrai container under ./ai's real flags: see `make test-container`.
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
writable() { f="$1/.probe.$$"; touch "$f" 2>/dev/null && rm -f "$f"; }

ws_root=/tmp/mrai-home/workspaces

check "not root" '[ "$(id -u)" != 0 ]'
check "capabilities dropped" 'grep -Eq "^CapBnd:[[:space:]]+0+$" /proc/self/status'
check "no_new_privs set" 'grep -Eq "^NoNewPrivs:[[:space:]]+1$" /proc/self/status'
check "repo read-only" '! writable "$PWD"'
check "other workspaces read-only" '[ -d "$ws_root/other-ws" ] && ! writable "$ws_root/other-ws"'
check "own workspace writable" 'writable "$MRAI_DIR"'
check "site writable" 'writable /tmp/mrai-home/site'
check "mrai.yml read-only" '[ ! -e /tmp/mrai-home/mrai.yml ] || ! (: >>/tmp/mrai-home/mrai.yml) 2>/dev/null'
check "no docker socket" '[ ! -e /var/run/docker.sock ]'
check "host env not passed through" '[ -z "${MRAI_PROBE_CANARY:-}" ]'

unexpected=$(awk -v pwd="$PWD" -v ws="$ws_root" -v own="$MRAI_DIR" '
  { split($0, p, " - "); split(p[2], t, " "); mp = $5 }
  t[1] ~ /^(proc|sysfs|tmpfs|devpts|mqueue|cgroup2?|overlay|shm)$/ { next }
  mp ~ /^\/etc\/(hosts|hostname|resolv\.conf)$/ { next }
  mp == pwd || mp == ws || mp == own || mp == "/tmp/mrai-home/site" || mp == "/tmp/mrai-home/mrai.yml" { next }
  { print mp }' /proc/self/mountinfo)
check "no unexpected mounts" '[ -z "$unexpected" ]'
[ -n "$unexpected" ] && printf '     %s\n' $unexpected

exit $fail
