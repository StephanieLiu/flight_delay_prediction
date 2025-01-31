#!/usr/bin/env bash
URL="${URL:-http://127.0.0.1:8500/predict}"
siege -c 255 -r 100 -b --content-type 'application/json' "$URL POST <payload_flask.json"