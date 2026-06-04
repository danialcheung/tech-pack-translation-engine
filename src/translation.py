"""Google Cloud Translation API wrapper.

Uses lazy initialization so the client is only created when the first
translation is actually requested. This prevents import-time crashes when
credentials aren't configured (e.g. during local testing or linting).
"""

import logging
import os

from google.cloud import translate_v3 as translate

logger = logging.getLogger(__name__)

_client: translate.TranslationServiceClient | None = None
_parent: str | None = None


def _get_client() -> tuple[translate.TranslationServiceClient, str]:
    """Lazily initialize the translation client on first use.

    Requires GOOGLE_CLOUD_PROJECT env var and valid Application Default
    Credentials (set via GOOGLE_APPLICATION_CREDENTIALS in Docker).
    """
    global _client, _parent
    if _client is None:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise EnvironmentError(
                "GOOGLE_CLOUD_PROJECT environment variable is not set. "
                "Set it to your GCP project ID."
            )
        _client = translate.TranslationServiceClient()
        _parent = f"projects/{project_id}/locations/global"
    return _client, _parent


def translate_text(text: str, target_lang: str = 'zh') -> str:
    """Translate a single English text string to the target language.

    Falls back to returning the original text if the API call fails,
    ensuring the pipeline doesn't crash on transient network errors.
    """
    if not text.strip():
        return text

    client, parent = _get_client()

    try:
        response = client.translate_text(
            request={
                "parent": parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": "en",
                "target_language_code": target_lang,
            }
        )
        return response.translations[0].translated_text
    except Exception as e:
        logger.warning("Translation failed for '%s...': %s", text[:20], e)
        return text
