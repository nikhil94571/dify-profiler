# deploy.ps1

$SA_EMAIL = $env:SA_EMAIL
$EXPORT_BUCKET = $env:EXPORT_BUCKET
$PROFILER_API_KEY = $env:PROFILER_API_KEY

if (-not $SA_EMAIL) { throw "Missing service account email. Set $env:SA_EMAIL" }
if (-not $EXPORT_BUCKET) { throw "Missing export bucket. Set $env:EXPORT_BUCKET" }
if (-not $PROFILER_API_KEY) { throw "Missing profiler API key. Set $env:PROFILER_API_KEY" }

function Resolve-Gcloud {
  $cmd = Get-Command gcloud -ErrorAction SilentlyContinue
  if ($cmd) {
    return $cmd.Source
  }

  $candidates = @(
    "$HOME/google-cloud-sdk/bin/gcloud",
    "/opt/homebrew/share/google-cloud-sdk/bin/gcloud",
    "/opt/homebrew/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/bin/gcloud",
    "/usr/local/share/google-cloud-sdk/bin/gcloud"
  )

  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
      return $candidate
    }
  }

  throw "gcloud not found in PowerShell PATH or common macOS install locations."
}

$GCLOUD = Resolve-Gcloud

# (Decision) Build env vars from a hashtable to guarantee unique keys.
# Why it matters: avoids the exact duplicate-key failure you're seeing.
$envVars = @{
  "PROFILER_API_KEY"              = $PROFILER_API_KEY
  "MAX_UPLOAD_BYTES"              = "20971520"
  "RATE_LIMIT_MAX"                = "30"
  "RATE_LIMIT_WINDOW"             = "60"
  "LOG_LEVEL"                     = "INFO"
  "EXPORT_BUCKET"                 = $EXPORT_BUCKET
  "SIGNING_SA_EMAIL"              = $SA_EMAIL
  "EXPORT_SIGNED_URL_TTL_MINUTES" = "30"
  "PROFILE_SOURCE"                = "gcs"
  "PROFILE_PREFIX"                = "profiles/"
}


$envVarString = ($envVars.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ","

& $GCLOUD run deploy dify-profiler-v2 `
  --source . `
  --region australia-southeast1 `
  --max-instances=2 `
  --concurrency=1 `
  --service-account $SA_EMAIL `
  --set-env-vars $envVarString
