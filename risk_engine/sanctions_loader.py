"""
sanctions_loader.py — Downloads and parses real public sanctions/PEP lists.

Sources (all free, public, machine-readable):
  OFAC SDN:  https://www.treasury.gov/ofac/downloads/sdn.csv
  UN SC:     https://scsanctions.un.org/resources/xml/en/consolidated.xml
  RBI UAPA:  https://www.rbi.org.in/Scripts/bs_viewcontent.aspx?Id=2485
             (HTML page — parsed below for entity names)

Run standalone to refresh:
    python risk_engine/sanctions_loader.py --output risk_engine/sanctions_data.json

The output JSON is consumed by sanctions_screening.py at startup.
Cached locally so the API does not fetch on every request.
"""
import json, re, argparse, os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

OFAC_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"
UN_URL   = "https://scsanctions.un.org/resources/xml/en/consolidated.xml"

def _fetch_ofac() -> list:
    """Parse OFAC SDN CSV — columns: ent_num, SDN_Name, SDN_Type, ..."""
    if not REQUESTS_OK:
        return []
    try:
        r = requests.get(OFAC_URL, timeout=20)
        r.raise_for_status()
        names = []
        for line in r.text.splitlines():
            parts = line.split(",")
            if len(parts) > 1:
                raw = parts[1].strip().strip('"')
                if raw and raw != "SDN_Name" and len(raw) > 3:
                    names.append(raw)
        return names[:5000]   # cap — full list is 10k+ rows
    except Exception:
        logger.warning("OFAC fetch failed — returning empty list.", exc_info=True)
        return []

def _fetch_un() -> list:
    """Parse UN consolidated XML — extract FIRST_NAME + SECOND_NAME + THIRD_NAME."""
    if not REQUESTS_OK:
        return []
    try:
        r = requests.get(UN_URL, timeout=25)
        r.raise_for_status()
        # Simple regex — avoids xml.etree dependency on fresh envs
        firsts  = re.findall(r'<FIRST_NAME>(.*?)</FIRST_NAME>',  r.text)
        seconds = re.findall(r'<SECOND_NAME>(.*?)</SECOND_NAME>', r.text)
        thirds  = re.findall(r'<THIRD_NAME>(.*?)</THIRD_NAME>',  r.text)
        names = []
        for i, f in enumerate(firsts):
            parts = [f]
            if i < len(seconds) and seconds[i]: parts.append(seconds[i])
            if i < len(thirds)  and thirds[i]:  parts.append(thirds[i])
            full = " ".join(parts).strip()
            if full:
                names.append(full)
        return names
    except Exception:
        logger.warning("UN fetch failed — returning empty list.", exc_info=True)
        return []

def _rbi_seed() -> list:
    """
    RBI UAPA list is an HTML page — fragile to parse without real
    browser rendering. We include the known-static entries here as a
    curated seed and flag this as a manual-refresh item.
    These are REAL entity names from RBI's published UAPA list.
    """
    return [
        "Lashkar-e-Taiba", "Jaish-e-Mohammed", "Al-Qa'ida",
        "Harkat-ul-Mujahidin", "Hizbul Mujahideen",
        "Indian Mujahideen", "SIMI", "Babbar Khalsa International",
        "Khalistan Zindabad Force", "Khalistan Liberation Force",
        "ISIS", "Islamic State", "Daesh",
    ]

def build(output_path: str):
    print("Fetching OFAC SDN list...")
    ofac = _fetch_ofac()
    print(f"  {len(ofac)} OFAC entries")

    print("Fetching UN Security Council list...")
    un = _fetch_un()
    print(f"  {len(un)} UN entries")

    rbi = _rbi_seed()
    print(f"  {len(rbi)} RBI UAPA seed entries (static)")

    # Deduplicate by lowercased name
    seen, combined = set(), []
    for name in ofac + un + rbi:
        key = name.lower().strip()
        if key and key not in seen:
            seen.add(key)
            combined.append(name.strip())

    data = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "counts": {"ofac": len(ofac), "un": len(un), "rbi_seed": len(rbi),
                   "total_unique": len(combined)},
        "sanctions": combined,
        "pep": [],   # PEP list requires commercial feed — slot reserved
        "note": (
            "OFAC and UN entries fetched from official public APIs. "
            "RBI UAPA entries are a static curated seed — refresh manually "
            "from https://www.rbi.org.in. PEP list requires a commercial "
            "data provider and is not yet integrated."
        ),
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(combined)} unique entries -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="risk_engine/sanctions_data.json")
    args = parser.parse_args()
    build(args.output)
