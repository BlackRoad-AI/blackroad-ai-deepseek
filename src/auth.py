"""
BlackRoad Gateway Authentication
Supports API key / Bearer token auth for the local AI gateway.

Set the BLACKROAD_API_KEY environment variable to authenticate requests.
"""
import os
from typing import Optional


def get_auth_headers(api_key: Optional[str] = None) -> dict:
    """Return Authorization headers for the BlackRoad gateway.

    Args:
        api_key: Explicit API key. Falls back to the BLACKROAD_API_KEY
                 environment variable when not provided.

    Returns:
        A dict with an ``Authorization: Bearer <key>`` header when a key is
        available, or an empty dict when no key is configured.
    """
    key = api_key or os.environ.get("BLACKROAD_API_KEY", "")
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}
