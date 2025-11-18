# osm_scrapper.py
#
# Fetch OSM "services" (education, healthcare, public services, transport,
# culture, sustenance, retail, leisure) for Portuguese municipalities
# taken from total_average_income_by_municipality.csv.
#
# Uses Nominatim + Overpass, with tags inspired by:
#   https://wiki.openstreetmap.org/wiki/Map_features

import time
import unicodedata
from typing import Dict, List, Tuple, Any

import pandas as pd
import requests


# ==========================
# 0. API endpoints & headers
# ==========================

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# !!! EDIT THIS !!!
# Must include a way to contact you (email or project URL) per Nominatim policy
USER_AGENT = (
    "FCD-housing-prices/1.0 "
    "(https://github.com/Shushunya/FCD-housing-prices; goncamarqs@gmail.com)"
)

HEADERS = {
    "User-Agent": USER_AGENT
}


# ==========================
# 1. Service tags (based on Map Features)
# ==========================

# Structure:
#   key -> category -> list of values
SERVICE_TAGS: Dict[str, Dict[str, List[str]]] = {
    "amenity": {
        "education": [
            "kindergarten", "school", "college", "university",
        ],
        "healthcare": [
            "hospital", "pharmacy",
        ],
        "public_service": [
            "police", "fire_station", "courthouse",
            "post_office"
        ],
        "culture": [
            "library","cinema", "theatre"
        ],
    },
    "shop": {
        "retail_food": [
             "mall"
        ],
    },
    "railway": {
        "transport": [
            "station",
        ],
    },
    "public_transport": {
        "transport": [
            "station",
        ],
    },
    "tourism": {
        "culture": [
            "museum",
        ]
    },
}


# ==========================
# 2. Read municipalities from income CSV
# ==========================

def strip_accents(s: str) -> str:
    """
    Remove accents from a Unicode string.
    'Águeda' -> 'Agueda'
    """
    if not isinstance(s, str):
        return s
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


def read_income_csv(csv_path: str) -> pd.DataFrame:
    """
    Try to read the income CSV with a sensible encoding.
    """
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            print(f"Trying encoding: {enc}")
            df = pd.read_csv(csv_path, encoding=enc)
            return df
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "Could not decode CSV with utf-8 / utf-8-sig / latin-1. "
        "Please check the file encoding."
    )


def get_municipalities_from_income(
    csv_path: str = "total_average_income_by_municipality.csv",
) -> List[str]:
    """
    Reads total_average_income_by_municipality.csv and returns the list
    of municipalities from the 'Region' column with:
      - Scope == 'Município'
      - Education_Level == 'Total'
    (across all years)
    """
    df = read_income_csv(csv_path)
    print("Columns found in income CSV:", list(df.columns))

    required_cols = {"Region", "Scope", "Education_Level"}
    if not required_cols.issubset(df.columns):
        raise KeyError(
            f"CSV must contain columns {required_cols}, "
            f"but has {set(df.columns)}"
        )

    # Filter to municipality scope + total education
    muni_df = df[
        (df["Scope"] == "Município") &
        (df["Education_Level"] == "Total")
    ]

    municipalities = sorted(muni_df["Region"].dropna().unique())
    print(f"Found {len(municipalities)} municipalities in income CSV.")
    print("Example municipalities:", municipalities[:20])

    return municipalities


# ==========================
# 3. Nominatim: municipality -> bounding box
# ==========================

def get_area_bbox(name: str) -> Tuple[float, float, float, float]:
    """
    Uses Nominatim to get a bounding box for a Portuguese municipality.

    Returns: (south, west, north, east)
    """
    name_clean = name.strip()
    name_no_accents = strip_accents(name_clean)

    queries = [
        f"{name_clean}, Portugal",
        f"{name_no_accents}, Portugal",
        f"Concelho de {name_clean}, Portugal",
        f"Concelho de {name_no_accents}, Portugal",
        f"{name_clean}, Município, Portugal",
        f"{name_no_accents}, Municipio, Portugal",
    ]

    for q in queries:
        print(f"    Nominatim query: {q}")
        params = {
            "q": q,
            "format": "json",
            "limit": 1,
            "countrycodes": "pt",
        }
        resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS)
        if resp.status_code == 403:
            raise RuntimeError(
                "Nominatim returned 403 Forbidden. "
                "Check USER_AGENT and respect their usage policy."
            )
        resp.raise_for_status()
        data = resp.json()
        if data:
            bb = data[0]["boundingbox"]  # [south, north, west, east] as strings
            south, north, west, east = map(float, bb)
            return south, west, north, east

    raise ValueError(f"Could not find bbox for area '{name}' in Portugal")


# ==========================
# 4. Build Overpass query for a bbox & SERVICE_TAGS
# ==========================

def build_overpass_query(
    south: float, west: float, north: float, east: float
) -> str:
    """
    Builds a single Overpass QL query that fetches all nodes/ways/relations
    with tags defined in SERVICE_TAGS within the given bounding box.
    """
    bbox = f"{south},{west},{north},{east}"

    parts: List[str] = []

    for key, categories in SERVICE_TAGS.items():
        for category, values in categories.items():
            for val in values:
                # nodes
                parts.append(f'  node["{key}"="{val}"]({bbox});')
                # ways
                parts.append(f'  way["{key}"="{val}"]({bbox});')
                # relations
                parts.append(f'  relation["{key}"="{val}"]({bbox});')

    body = "\n".join(parts)

    query = f"""
    [out:json][timeout:180];
    (
{body}
    );
    out center;
    """
    return query


# ==========================
# 5. Fetch services for one municipality
# ==========================

def fetch_services_for_area(area_name: str) -> pd.DataFrame:
    """
    Fetches OSM services for a single area (municipality) and returns a DataFrame
    with one row per OSM object.
    """
    print(f"Fetching bbox for area: {area_name}")
    south, west, north, east = get_area_bbox(area_name)
    print(f"  Bbox: S={south}, W={west}, N={north}, E={east}")

    query = build_overpass_query(south, west, north, east)

    print(f"  Querying Overpass for {area_name}...")
    resp = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS)
    if resp.status_code == 429:
        raise RuntimeError("Overpass rate limit (429). Try increasing sleep_seconds.")
    resp.raise_for_status()
    data = resp.json()
    elements = data.get("elements", [])

    rows: List[Dict[str, Any]] = []

    for el in elements:
        el_type = el.get("type")  # node, way, relation
        osm_id = el.get("id")
        tags = el.get("tags", {})

        # Coordinates:
        if el_type == "node":
            lat = el.get("lat")
            lon = el.get("lon")
        else:
            center = el.get("center")
            if center:
                lat = center.get("lat")
                lon = center.get("lon")
            else:
                continue  # skip features without a center

        name = tags.get("name")

        # Identify which (key, value) pair and category this element matches
        main_key = None
        main_value = None
        main_category = None

        for key, categories in SERVICE_TAGS.items():
            if key in tags:
                val = tags[key]
                for category, values in categories.items():
                    if val in values:
                        main_key = key
                        main_value = val
                        main_category = category
                        break
                if main_key is not None:
                    break

        if main_key is None:
            continue

        rows.append(
            {
                "area": area_name,
                "osm_type": el_type,
                "osm_id": osm_id,
                "name": name,
                "key": main_key,
                "value": main_value,
                "category": main_category,
                "lat": lat,
                "lon": lon,
            }
        )

    df = pd.DataFrame(rows)
    print(f"  Found {len(df)} services in {area_name}")
    return df


# ==========================
# 6. Main: fetch for all municipalities & save
# ==========================

def fetch_all_area_services(
    income_csv: str = "total_average_income_by_municipality.csv",
    out_raw_csv: str = "osm_services_raw.csv",
    out_counts_csv: str = "osm_services_counts.csv",
    sleep_seconds: float = 3.0,
) -> None:
    """
    High-level pipeline:
      1. Read income data, get municipality names from 'Region'
      2. For each municipality, query OSM and collect requested services
      3. Save raw services + aggregated counts
    """

    areas = get_municipalities_from_income(income_csv)
    print(f"Number of municipalities: {len(areas)}")

    all_dfs: List[pd.DataFrame] = []

    for i, area_name in enumerate(areas, start=1):
        print(f"\n=== ({i}/{len(areas)}) Area: {area_name} ===")
        try:
            df_area = fetch_services_for_area(area_name)
            all_dfs.append(df_area)
        except Exception as e:
            print(f"  ERROR fetching area '{area_name}': {e}")
        time.sleep(sleep_seconds)  # be nice to the APIs

    if not all_dfs:
        print("No data fetched. Exiting.")
        return

    all_services = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal services collected: {len(all_services)}")

    # Save raw data (one row per OSM feature)
    all_services.to_csv(out_raw_csv, index=False, encoding="utf-8")
    print(f"Raw OSM services saved to: {out_raw_csv}")

    # Aggregate counts per area x value (wide format)
    counts = (
        all_services
        .groupby(["area", "category", "value"])
        .size()
        .reset_index(name="count")
    )

    counts_pivot = (
        counts
        .pivot_table(
            index="area",
            columns="value",
            values="count",
            fill_value=0,
        )
        .reset_index()
    )

    counts_pivot.to_csv(out_counts_csv, index=False, encoding="utf-8")
    print(f"Aggregated counts (wide) saved to: {out_counts_csv}")


if __name__ == "__main__":
    fetch_all_area_services(
        income_csv="total_average_income_by_municipality.csv",
        out_raw_csv="osm_services_raw.csv",
        out_counts_csv="osm_services_counts.csv",
        sleep_seconds=3.0,
    )
