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


def _looks_like_oddsportal(html: str) -> tuple[bool, str]:
    """Heuristically determine whether the HTML came from OddsPortal."""

    if "oddsportal" not in html.lower():
        return False, "HTML does not mention 'oddsportal'; did you save the page source?"

    if BeautifulSoup is None:
        return True, "Skipped deep detection because BeautifulSoup is unavailable"

    soup = BeautifulSoup(html, "html.parser")
    canonical = soup.find("link", rel="canonical")
    og_site_name = soup.find("meta", property="og:site_name")
    if canonical and "oddsportal.com" in (canonical.get("href") or ""):
        return True, "Canonical link is OddsPortal"
    if og_site_name and "oddsportal" in (og_site_name.get("content") or "").lower():
        return True, "OpenGraph site name is OddsPortal"

    return False, "Could not find canonical/OG markers for OddsPortal"


def _load_html(fetcher: Any, slug: str, html_file: Optional[Path]) -> tuple[str | None, str]:
    """Return HTML for a slug, preferring a local file when provided."""

    if html_file:
        html = html_file.read_text(encoding="utf-8", errors="ignore")
        return html, str(html_file)

    url = urljoin(fetcher.base_url, slug)
    html = fetcher._request(url)
    return html, url


def inspect_slug(fetcher: Any, slug: str, season: str, html_file: Optional[Path]) -> bool:
    html, source = _load_html(fetcher, slug, html_file)
    if not html:
        logging.error("No HTML returned for %s", source)
        return

    logging.info("Loaded %d bytes from %s", len(html.encode("utf-8")), source)

    looks_like_ops, ops_reason = _looks_like_oddsportal(html)
    if not looks_like_ops:
        logging.warning(
            "Input HTML does not look like an OddsPortal results page (%s). If you passed --html-file, make sure it is a saved OddsPortal page source.",
            ops_reason,
        )
        if html_file:
            logging.error(
                "Provided --html-file %s does not resemble an OddsPortal page; supply a saved OddsPortal results HTML instead.",
                html_file,
            )
            return False

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

    if not any(signals.values()):
        logging.warning(
            "No OddsPortal markers were detected in the HTML. This usually means the page is a bot wall, an unrelated HTML file, or that OddsPortal changed markup.",
        )

    try:
        frame = fetcher._parse_results_page(html, season, slug)
    except Exception:
        logging.exception("Parser raised an exception for %s", source)
        return False

    if frame.empty:
        logging.warning("Parser returned an empty frame for %s", source)
        return False

    logging.info("Parser returned %d rows and columns: %s", len(frame), list(frame.columns))
    logging.debug("First rows:\n%s", frame.head().to_string())
    return True


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
        "--cookie",
        help="Optional Cookie header copied from a browser session to bypass bot walls.",
    )
    parser.add_argument(
        "--header",
        action="append",
        help="Extra request header(s) in Key:Value format (can be repeated).",
    )
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
    extra_headers = {}
    if args.cookie:
        extra_headers["Cookie"] = args.cookie.strip()
    for header in args.header or []:
        if not header or ":" not in header:
            logging.warning("Ignoring malformed --header value (expected Key:Value): %s", header)
            continue
        key, value = header.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            extra_headers[key] = value

    fetcher = OddsPortalFetcher(
        session,
        base_url=args.base_url,
        results_path=args.results_path,
        season_path_template=args.season_template,
        timeout=args.timeout,
        extra_headers=extra_headers or None,
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

    any_success = False
    for slug in slugs:
        parsed = inspect_slug(fetcher, slug, args.season, args.html_file)
        any_success = any_success or parsed

        # If a provided HTML file is not a valid OddsPortal page or parsed empty,
        # fall back to a live fetch so users can quickly compare local vs live.
        if args.html_file and not parsed:
            logging.warning(
                "Falling back to live OddsPortal request for %s because the provided --html-file did not yield closing odds.",
                slug,
            )
            parsed_live = inspect_slug(fetcher, slug, args.season, None)
            any_success = any_success or parsed_live

    if args.html_file and not any_success:
        raise SystemExit(
            "No closing odds could be parsed from the provided --html-file or the live fallback. Pass a browser-saved OddsPortal results HTML page (Ctrl+S page source) or omit --html-file to fetch live only.",
        )

    if not args.html_file and not any_success:
        raise SystemExit(
            "No closing odds were parsed from live OddsPortal requests. Check the logs above for bot-wall guidance and confirm the slugs are correct.",
        )


if __name__ == "__main__":
    main()
