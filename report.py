"""
AI-generated radiology summary using Mistral API.

Uses raw HTTP to avoid SDK version issues.
"""

import requests


SYSTEM_PROMPT = """\
You are ScanAssist, a medical AI pre-screening assistant. You receive 
structured segmentation data from an abdominal CT slice analyzed by an 
automatic organ segmentation model.

The model identifies up to 54 anatomical structures (organs, vessels, bones,
and potential tumors). Each structure is identified by a numeric label only
(no semantic organ names are available).

Generate a concise pre-screening report with these sections:

**Overview:** One sentence summarizing what was found (number of structures,
overall coverage).

**Key findings:** 2-3 bullet points highlighting notable observations:
relative sizes, dominant structures, and anything that stands out 
statistically (e.g. unusually large or small structures).

**Recommendation:** One sentence suggesting next steps (e.g. "No anomalies 
flagged — routine review recommended" or "Structure X is unusually 
dominant — further review suggested").

End with: "*This is an automated pre-screening. All findings must be 
validated by a qualified radiologist.*"

Be concise (max 150 words). Professional but accessible to non-specialists.
Write in English.
"""

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


def generate_report(stats: list[dict], api_key: str) -> str:
    if not stats:
        return "_No structures detected — nothing to report._"

    total_pixels = sum(s["pixels"] for s in stats)
    lines = [
        f"- Structure {s['label']:02d}: {s['pixels']} px ({s['pct_of_foreground']}%)"
        for s in stats
    ]

    user_message = (
        f"Segmentation results for one CT slice (256×256 pixels):\n"
        f"Total foreground pixels: {total_pixels:,}\n"
        f"Structures detected: {len(stats)}\n\n"
        + "\n".join(lines)
    )

    try:
        resp = requests.post(
            MISTRAL_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mistral-small-latest",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": 512,
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    except requests.exceptions.HTTPError:
        return f"⚠️ Mistral API error ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"⚠️ Error: `{e}`"
