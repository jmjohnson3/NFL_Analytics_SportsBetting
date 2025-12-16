"""Standalone OddsPortal diagnostics helper.

This script performs a targeted OddsPortal scrape for one or more slugs and
surfaces the signals the main pipeline relies on (HTML size, bot wall tokens,
count of rows/participants/scripts, and the parsed closing-odds frame). Use it
when live scraping is returning empty results to quickly spot SSL issues,
bot-protection pages, or markup changes.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional
from urllib.parse import urljoin

try:
    import requests
except ModuleNotFoundError as exc:  # pragma: no cover - defensive import guard
    requests = None
    _REQUESTS_ERROR = exc
else:
    _REQUESTS_ERROR = None

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError as exc:  # pragma: no cover - defensive import guard
    BeautifulSoup = None
    _BS4_ERROR = exc
else:
    _BS4_ERROR = None

def _analyze_html(html: str) -> dict:
    """Return quick structural signals about an OddsPortal results page."""

    if BeautifulSoup is None:
        raise RuntimeError(
            "The 'beautifulsoup4' package is required for HTML analysis. Install it with 'pip install beautifulsoup4'."
        )

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find(class_=re.compile(r"\btable-main\b"))
    modern_rows = soup.select("[data-id], [data-event-id]")
    legacy_nodes = soup.select("td.table-matches__participant")
    participant_nodes = soup.select(".table-participant")

    next_data_scripts: List[str] = []
    for script in soup.find_all("script"):
        content = script.get_text(" ", strip=True)
        if not content:
            continue
        if "__NEXT_DATA__" in (script.get("id") or ""):
            next_data_scripts.append("next-data-id")
        elif "nextData" in content or "__NEXT_DATA__" in content:
            next_data_scripts.append(content[:80])

    return {
        "has_table_main": table is not None,
        "modern_rows": len(modern_rows),
        "legacy_nodes": len(legacy_nodes),
        "participant_nodes": len(participant_nodes),
        "next_data_scripts": len(next_data_scripts),
    }


def _load_html(fetcher: Any, slug: str, html_file: Optional[Path]) -> tuple[str | None, str]:
    """Return HTML for a slug, preferring a local file when provided."""

    if html_file:
        html = html_file.read_text(encoding="utf-8", errors="ignore")
        return html, str(html_file)

    url = urljoin(fetcher.base_url, slug)
    html = fetcher._request(url)
    return html, url


def inspect_slug(fetcher: Any, slug: str, season: str, html_file: Optional[Path]) -> None:
    html, source = _load_html(fetcher, slug, html_file)
    if not html:
        logging.error("No HTML returned for %s", source)
        return

    logging.info("Loaded %d bytes from %s", len(html.encode("utf-8")), source)

    # Surface bot-wall signals immediately.
    fetcher._detect_bot_wall(html, url=source)

    signals = _analyze_html(html)
    logging.info(
        "Signals -> table_main=%s, modern_rows=%d, legacy_nodes=%d, participants=%d, next_data_scripts=%d",
        signals["has_table_main"],
        signals["modern_rows"],
        signals["legacy_nodes"],
        signals["participant_nodes"],
        signals["next_data_scripts"],
    )

    try:
        frame = fetcher._parse_results_page(html, season, slug)
    except Exception:
        logging.exception("Parser raised an exception for %s", source)
        return

    if frame.empty:
        logging.warning("Parser returned an empty frame for %s", source)
    else:
        logging.info("Parser returned %d rows and columns: %s", len(frame), list(frame.columns))
        logging.debug("First rows:\n%s", frame.head().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose OddsPortal closing-odds scraping.")
    parser.add_argument("--season", default="2024-regular", help="Season label to attribute parsed rows to.")
    parser.add_argument(
        "--slug",
        action="append",
        help="Specific OddsPortal results slug(s) to request. Defaults to the generated slugs for the season.",
    )
    parser.add_argument("--html-file", type=Path, help="Optional local HTML file to analyze instead of live requests.")
    parser.add_argument("--base-url", default="https://www.oddsportal.com/american-football/usa/", help="OddsPortal base URL")
    parser.add_argument("--results-path", default="nfl/results/", help="Primary results path slug")
    parser.add_argument(
        "--season-template",
        default="nfl-{season}/results/",
        help="Template used to build fallback slugs when --slug is not provided.",
    )
    parser.add_argument("--timeout", type=int, default=45, help="Request timeout in seconds")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if requests is None:
        raise SystemExit(
            "The 'requests' package is required for oddsportal_debug.py. Install it with 'pip install requests'."
        )

    if BeautifulSoup is None:
        raise SystemExit(
            "The 'beautifulsoup4' package is required for oddsportal_debug.py. Install it with 'pip install beautifulsoup4'."
        )

    try:
        from NFL_SPORTSBETTING import OddsPortalFetcher
    except ImportError as exc:  # pragma: no cover - surface missing optional deps early
        raise SystemExit(
            f"Unable to import OddsPortalFetcher ({exc}). Install the missing dependency (see README requirements)."
        ) from exc

    session = requests.Session()
    fetcher = OddsPortalFetcher(
        session,
        base_url=args.base_url,
        results_path=args.results_path,
        season_path_template=args.season_template,
        timeout=args.timeout,
    )

    # Ensure overrides from the main pipeline do not mask live debugging behaviour.
    fetcher._html_override_path = None
    fetcher._override_only = False
    fetcher._debug_dump_enabled = False

    slugs: Iterable[str]
    if args.slug:
        slugs = args.slug
    else:
        slugs = fetcher._season_slugs(args.season)

    for slug in slugs:
        inspect_slug(fetcher, slug, args.season, args.html_file)


if __name__ == "__main__":
    main()
