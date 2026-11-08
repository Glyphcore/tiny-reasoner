#!/usr/bin/env bash
# quiet runner: ./q <cmd...>  prints "ok" or "FAIL <code> <log>"; full output goes to .q/last.log
# Q_QUIET=1 (set by make) prints nothing on success.
mkdir -p .q; L=.q/last.log
"$@" >"$L" 2>&1; c=$?
if [ $c -eq 0 ]; then [ -n "$Q_QUIET" ] || echo ok; else echo "FAIL $c $L"; fi
exit $c
