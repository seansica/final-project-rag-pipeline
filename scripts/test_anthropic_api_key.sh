#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <api_key>"
    echo "Example: $0 sk-ant-api03-xxxxxxxxxxxx"
    exit 1
fi

API_KEY="$1"

curl https://api.anthropic.com/v1/messages \
        --header "x-api-key: $API_KEY" \
        --header "anthropic-version: 2023-06-01" \
        --header "content-type: application/json" \
        --data \
    '{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello, world"}
        ]
    }'