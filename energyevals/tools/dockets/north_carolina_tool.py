import json
import os
import re
from typing import Any

import requests
from bs4 import BeautifulSoup, Tag
from loguru import logger

from energyevals.utils import generate_timestamp, request_with_retry

from energyevals.tools.base_tool import tool_method
from energyevals.tools.dockets._base import DocketBaseTool


class NorthCarolinaDocketTool(DocketBaseTool):
    """Search North Carolina Utilities Commission dockets."""

    def __init__(self) -> None:
        super().__init__(
            name="north_carolina_dockets",
            description="Search North Carolina Utilities Commission dockets",
        )

    @tool_method(name="search_north_carolina_dockets")
    def search_north_carolina(
        self,
        date_from: str,
        date_to: str,
        docket_number: str | None = None,
        company_name: str | None = None,
        exclude_closed: bool = False,
        limit_to_filing_type_labels: list[str] | None = None,
        storage_state_path: str | None = None,
        max_pages: int = 5,
        timeout: int = 30,
    ) -> str:
        """Search North Carolina Utilities Commission dockets.

        Args:
            date_from: Start date for the search in MM/DD/YYYY format.
            date_to: End date for the search in MM/DD/YYYY format.
            docket_number: Docket number to filter by.
            company_name: Company name to filter by.
            exclude_closed: Whether to exclude closed dockets. Defaults to False.
            limit_to_filing_type_labels: List of filing type labels to restrict results.
            storage_state_path: Optional Playwright storage state path.
            max_pages: Maximum number of pages to fetch. Defaults to 5.
            timeout: Timeout in seconds. Defaults to 30.

        Returns:
            JSON string with the search results.
        """
        try:
            portal_url = "https://starw1.ncuc.gov/NCUC/page/Dockets/portal.aspx"
            timestamp = generate_timestamp()
            save_csv_path = f"north_carolina_dockets_{timestamp}.csv"

            def resolve_filing_type_values(
                soup: BeautifulSoup, desired_labels: list[str]
            ) -> list[str]:
                desired_set = {lbl.strip().lower() for lbl in desired_labels}
                values: list[str] = []
                sel = soup.find("select", id=re.compile(r"_filingTypesList$"))
                if not sel or not isinstance(sel, Tag):
                    return values
                for opt in sel.find_all("option"):
                    label = (opt.text or "").strip().lower()
                    if label in desired_set:
                        values.append(str(opt.get("value", "")))
                return values

            def parse_results_page(html: str) -> dict[str, Any]:
                soup = BeautifulSoup(html, "html.parser")
                results: list[dict[str, Any]] = []

                item_count = None
                count_span = soup.find(id=re.compile(r"_itemCountLabel$"))
                if count_span and count_span.text:
                    match = re.search(r"Items Count:(\d+)", count_span.text)
                    if match:
                        item_count = int(match.group(1))

                rss_link = None
                rss_a = soup.find(
                    "a", id=re.compile(r"RssButtonControl1_rssButtonHyperLink$")
                )
                if rss_a and isinstance(rss_a, Tag) and rss_a.has_attr("href"):
                    rss_link = str(rss_a["href"])

                for row in soup.select(
                    "tr.SearchResultsItem, tr.SearchResultsAlternatingItem"
                ):
                    a = row.select_one("a[href]")
                    if not a:
                        continue
                    docket_number_value = (a.get_text() or "").strip()
                    docket_link = a["href"]

                    tds = row.select("td.width-full")
                    description = None
                    if len(tds) >= 2:
                        description = (tds[1].get_text() or "").strip()

                    date_td = row.find("td", class_="text-left width-full")
                    date_filed = None
                    if date_td:
                        txt = " ".join(date_td.stripped_strings)
                        match = re.search(
                            r"Date Filed:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{4})", txt
                        )
                        if match:
                            date_filed = match.group(1)

                    results.append(
                        {
                            "docket_number": docket_number_value,
                            "date_filed": date_filed,
                            "description": description,
                            "docket_link": docket_link,
                        }
                    )

                pager_targets: list[str] = []
                for a in soup.select(
                    "tr.SearchResultsFooter a[href^='javascript:__doPostBack(']"
                ):
                    href = str(a.get("href", ""))
                    match = re.search(r"__doPostBack\('([^']+)'", href)
                    if match:
                        pager_targets.append(match.group(1))

                return {
                    "items": results,
                    "item_count": item_count,
                    "rss_link": rss_link,
                    "pager_targets": pager_targets,
                }

            # --- Form field name prefix ---
            base = (
                "ctl00$ContentPlaceHolder1$PortalPageControl1"
                "$ctl86$DocketSearchControlNCUC1$"
            )
            fld_from = base + "filedOnOrAfterTextBox"
            fld_to = base + "filedOnOrBeforeTextBox"
            fld_dkt = base + "docketNumberTextBox"
            fld_co = base + "companyNameTextBox"
            fld_excl = base + "filterByDocketTypeOpenClosed"
            fld_chk = base + "filterByDocketTypeCheckBox"
            fld_types = base + "filingTypesList"
            btn_search = base + "searchButton"

            def run_playwright_search(state_path: str) -> dict[str, Any]:
                try:
                    from playwright.sync_api import sync_playwright
                except ImportError as exc:
                    raise RuntimeError(
                        "Playwright is required for North Carolina UC scraping. "
                        "Install it with `pip install playwright` and run "
                        "`python -m playwright install`."
                    ) from exc

                with sync_playwright() as p:
                    has_state = os.path.exists(state_path)
                    browser = p.chromium.launch(headless=has_state)
                    context = (
                        browser.new_context(storage_state=state_path)
                        if has_state
                        else browser.new_context()
                    )
                    page = context.new_page()
                    page.set_default_timeout(timeout * 1000)
                    try:
                        page.goto(
                            portal_url, wait_until="networkidle", timeout=timeout * 1000
                        )
                    except Exception:
                        page.goto(
                            portal_url,
                            wait_until="domcontentloaded",
                            timeout=timeout * 1000,
                        )

                    if not has_state:
                        logger.info(
                            "Complete the Cloudflare challenge in the opened browser, "
                            "then press Enter here to continue."
                        )
                        try:
                            input()
                        except EOFError:
                            raise RuntimeError(
                                "Playwright requires manual confirmation "
                                "for the Cloudflare challenge."
                            ) from None

                        os.makedirs(os.path.dirname(state_path), exist_ok=True)
                        context.storage_state(path=state_path)

                    html = page.content()
                    landing = BeautifulSoup(html, "html.parser")

                    if date_from:
                        page.fill(f'input[name="{fld_from}"]', date_from)
                    if date_to:
                        page.fill(f'input[name="{fld_to}"]', date_to)
                    if docket_number:
                        page.fill(f'input[name="{fld_dkt}"]', docket_number)
                    if company_name:
                        page.fill(f'input[name="{fld_co}"]', company_name)

                    if exclude_closed:
                        page.check(f'input[name="{fld_excl}"]')

                    if limit_to_filing_type_labels:
                        values = resolve_filing_type_values(
                            landing, limit_to_filing_type_labels
                        )
                        if values:
                            page.check(f'input[name="{fld_chk}"]')
                            page.select_option(
                                f'select[name="{fld_types}"]', values
                            )

                    page.click(f'input[name="{btn_search}"]')
                    try:
                        page.wait_for_load_state("networkidle")
                    except Exception:
                        page.wait_for_load_state("domcontentloaded")

                    page_html = page.content()
                    parsed = parse_results_page(page_html)
                    all_items = parsed["items"]
                    pages_fetched = 1

                    pager_targets = parsed["pager_targets"]
                    while pages_fetched < max_pages and pager_targets:
                        target = pager_targets.pop(0)
                        page.evaluate(f"__doPostBack('{target}', '')")
                        try:
                            page.wait_for_load_state("networkidle")
                        except Exception:
                            page.wait_for_load_state("domcontentloaded")
                        html_n = page.content()
                        parsed_n = parse_results_page(html_n)
                        all_items.extend(parsed_n["items"])
                        pages_fetched += 1
                        pager_targets = parsed_n["pager_targets"]

                    browser.close()

                return {
                    "items": all_items,
                    "item_count": parsed["item_count"],
                    "rss_link": parsed["rss_link"],
                    "pages_fetched": pages_fetched,
                }

            # --- Main requests-based flow ---
            session = requests.Session()
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,image/apng,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": portal_url,
                "Origin": "https://starw1.ncuc.gov",
                "Upgrade-Insecure-Requests": "1",
                "Connection": "keep-alive",
            }
            try:
                r = request_with_retry("get", portal_url, session=session, headers=headers, timeout=timeout)
                r.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 403:
                    state_path = storage_state_path or os.path.expanduser(
                        "~/.config/energyevals/ncuc_storage_state.json"
                    )
                    payload = run_playwright_search(state_path)
                    saved_csv = self._save_csv(payload["items"], save_csv_path)
                    payload["saved_csv"] = saved_csv
                    return self._result_json(payload)
                raise

            landing = BeautifulSoup(r.text, "html.parser")
            data: dict[str, Any] = self._collect_hidden_fields(landing)

            if date_from:
                data[fld_from] = date_from
            if date_to:
                data[fld_to] = date_to
            if docket_number:
                data[fld_dkt] = docket_number
            if company_name:
                data[fld_co] = company_name

            if exclude_closed:
                data[fld_excl] = "on"
            else:
                data.pop(fld_excl, None)

            if limit_to_filing_type_labels:
                values = resolve_filing_type_values(landing, limit_to_filing_type_labels)
                if values:
                    data[fld_chk] = "on"
                    data[fld_types] = values

            data[btn_search] = "Search"

            r2 = request_with_retry("post", portal_url, session=session, headers=headers, data=data, timeout=timeout)
            r2.raise_for_status()
            page_html = r2.text
            parsed = parse_results_page(page_html)

            all_items = parsed["items"]
            raw_pages = [page_html]
            pages_fetched = 1

            pager_targets = parsed["pager_targets"]
            while pages_fetched < max_pages and pager_targets:
                target = pager_targets.pop(0)
                soup_prev = BeautifulSoup(raw_pages[-1], "html.parser")
                post_data = self._collect_hidden_fields(soup_prev)
                post_data["__EVENTTARGET"] = target
                post_data["__EVENTARGUMENT"] = ""

                response_n = request_with_retry(
                    "post", portal_url, session=session, headers=headers, data=post_data, timeout=timeout
                )
                response_n.raise_for_status()
                html_n = response_n.text
                raw_pages.append(html_n)
                parsed_n = parse_results_page(html_n)
                all_items.extend(parsed_n["items"])
                pages_fetched += 1
                pager_targets = parsed_n["pager_targets"]

            saved_csv = self._save_csv(all_items, save_csv_path)
            return self._result_json(
                {
                    "items": all_items,
                    "item_count": parsed["item_count"],
                    "rss_link": parsed["rss_link"],
                    "pages_fetched": pages_fetched,
                    "saved_csv": saved_csv,
                }
            )
        except Exception as e:
            logger.error(f"North Carolina UC search failed: {e}")
            return json.dumps({"error": str(e), "source": "North Carolina UC"})
