import logging

import requests

requests.packages.urllib3.disable_warnings()

logger = logging.getLogger(__name__)

_session = requests.Session()
_session.verify = False


def translate_vi_to_en(text: str) -> str:
    resp = _session.get(
        "https://translate.googleapis.com/translate_a/single",
        params={"client": "gtx", "sl": "vi", "tl": "en", "dt": "t", "q": text},
        timeout=5,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(part[0] for part in data[0] if part[0])
