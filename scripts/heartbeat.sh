#!/bin/bash
# 5-minute heartbeat for spurious forgetting research sprint
# Checks comms bus and triggers OpenClaw session

AGENT="robonk"
BUS="http://139.59.171.116:7777"
AUTH="X-Auth: precisionai-agent-comms-2026"

# Check for unread messages
UNREAD=$(curl -s --connect-timeout 5 -m 10 "$BUS/unread?agent=$AGENT" -H "$AUTH" 2>/dev/null)

if [ $? -ne 0 ]; then
    exit 0  # Bus unreachable, skip silently
fi

MSG_COUNT=$(echo "$UNREAD" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('messages',[])))" 2>/dev/null)

if [ "${MSG_COUNT:-0}" -gt 0 ]; then
    # There are messages — trigger OpenClaw to handle them
    openclaw run --task "Research sprint heartbeat: Check comms bus for unread messages at $BUS. Agent name: robonk. Read messages, act on research tasks (spurious forgetting project at /root/spurious-forgetting — code is on GitHub at Robort-Precision/spurious-forgetting). Reply via /send, ack via /ack. If Robort sent experimental design or instructions, start building what's needed. If Yoshi needs code, send it. Push updates to GitHub." --model anthropic/claude-sonnet-4-20250514 2>/dev/null &
fi
