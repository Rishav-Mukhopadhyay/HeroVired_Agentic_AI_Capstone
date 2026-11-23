"""ICD-10-CM diagnosis code lookup via the NLM Clinical Tables API (free, no key).
https://clinicaltables.nlm.nih.gov/apidoc/icd10cm/v3/doc.html
"""

from __future__ import annotations

import requests

ICD10_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"


def search_icd10(term: str, max_results: int = 7) -> list[dict]:
    """Return [{'code': 'J70.0', 'name': 'Acute pulmonary manifestations due to radiation'}, ...]."""
    r = requests.get(ICD10_URL, params={"sf": "code,name", "df": "code,name",
                                        "terms": term, "maxList": max_results}, timeout=20)
    r.raise_for_status()
    # Response shape: [total, [codes], extra_fields, [[code, name], ...]]
    _total, _codes, _extra, rows = r.json()
    return [{"code": code, "name": name} for code, name in (rows or [])]
