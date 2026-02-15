# deploy.ps1

$SA_EMAIL = $env:SA_EMAIL
$EXPORT_BUCKET = $env:EXPORT_BUCKET

if (-not $SA_EMAIL) { throw "Missing service account email. Set $env:SA_EMAIL" }
if (-not $EXPORT_BUCKET) { throw "Missing export bucket. Set $env:EXPORT_BUCKET" }

# (Decision) Build env vars from a hashtable to guarantee unique keys.
# Why it matters: avoids the exact duplicate-key failure you're seeing.
$envVars = @{
  "PROFILER_API_KEY"              = $(if ($env:PROFILER_API_KEY) { $env:PROFILER_API_KEY } else { "test_key_123" })
  "MAX_UPLOAD_BYTES"              = "20971520"
  "RATE_LIMIT_MAX"                = "30"
  "RATE_LIMIT_WINDOW"             = "60"
  "LOG_LEVEL"                     = "INFO"
  "EXPORT_BUCKET"                 = $EXPORT_BUCKET
  "SIGNING_SA_EMAIL"              = $SA_EMAIL
  "EXPORT_SIGNED_URL_TTL_MINUTES" = "30"
}


$envVarString = ($envVars.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ","

gcloud run deploy dify-profiler-v2 `
  --source . `
  --region australia-southeast1 `
  --max-instances=2 `
  --concurrency=1 `
  --service-account $SA_EMAIL `
  --set-env-vars $envVarString
