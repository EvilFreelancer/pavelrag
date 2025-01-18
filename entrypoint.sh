#!/bin/bash

cd "$(dirname "$0")"

[ "x$APP_WORKERS" = "x" ] && export APP_WORKERS="1"
[ "x$APP_BIND" = "x" ] && export APP_BIND="0.0.0.0"
[ "x$APP_PORT" = "x" ] && export APP_PORT="5000"
[ "x$APP_LOG_LEVEL" = "x" ] && export APP_LOG_LEVEL="info"
[ "x$APP_TIMEOUT_SECONDS" = "x" ] && export APP_TIMEOUT_SECONDS="300"

gunicorn \
  --timeout=$APP_TIMEOUT_SECONDS \
  --workers=$APP_WORKERS \
  --bind=${APP_BIND}:${APP_PORT} \
  --log-level=${APP_LOG_LEVEL} \
  --error-logfile=/dev/stderr \
  --access-logfile=/dev/stdout \
  --log-file=/dev/stdout \
  wsgi:vector_app
