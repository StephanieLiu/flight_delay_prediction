#!/usr/bin/env bash
URL="${URL:-http://127.0.0.1:8080/v2/models/camp-accept-predictor-production/infer}"
siege -c 255 -r 100 -b --content-type 'application/json' "$URL POST <payload_mlserver.json"