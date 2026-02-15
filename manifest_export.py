# manifest_export.py
import os
from datetime import timedelta

from google.cloud import storage
from google.auth import default
from google.auth.transport.requests import Request


def upload_and_sign_text(
    bucket_name: str,
    object_name: str,
    text_content: str,
    expiration_minutes: int = 30,
) -> str:
    # Upload text
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    blob.upload_from_string(
        text_content or "",
        content_type="text/plain; charset=utf-8",
    )

    # Cloud Run / IAM-compatible signing (no private key file required)
    credentials, _ = default()

    if getattr(credentials, "requires_scopes", False):
        credentials = credentials.with_scopes(["https://www.googleapis.com/auth/cloud-platform"])

    credentials.refresh(Request())

    service_account_email = os.environ.get("SIGNING_SA_EMAIL")
    if not service_account_email:
        raise ValueError("SIGNING_SA_EMAIL environment variable not set")

    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expiration_minutes),
        method="GET",
        service_account_email=service_account_email,
        access_token=credentials.token,
    )

    return signed_url
