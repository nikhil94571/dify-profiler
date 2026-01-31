gcloud run deploy dify-profiler-v2 `
  --source . `
  --region australia-southeast1 `
  --max-instances=2 `
  --concurrency=1 `
  --set-env-vars "PROFILER_API_KEY=test_key_123,MAX_UPLOAD_BYTES=20971520,RATE_LIMIT_MAX=30,RATE_LIMIT_WINDOW=60,LOG_LEVEL=INFO"
