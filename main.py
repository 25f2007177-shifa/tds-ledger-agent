import json
import os
import re
import requests

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="TDS Ledger Agent")

IST = ZoneInfo("Asia/Kolkata")
TIMEOUT = 10


class QuestionRequest(BaseModel):
    question: str


def fetch_json(url):
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def page_url(url, page_number):
    """Change only the page parameter, preserving the other query parameters."""
    parts = urlsplit(url)
    params = parse_qsl(parts.query, keep_blank_values=True)

    found = False
    updated_params = []

    for key, value in params:
        if key == "page":
            updated_params.append((key, str(page_number)))
            found = True
        else:
            updated_params.append((key, value))

    if not found:
        updated_params.append(("page", str(page_number)))

    return urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path,
        urlencode(updated_params),
        parts.fragment,
    ))


@lru_cache(maxsize=1)
def load_ledger():
    root_url = os.environ.get("QUESTION_DATA_URL")
    if not root_url:
        raise RuntimeError("QUESTION_DATA_URL is not configured")

    root = fetch_json(root_url)
    links = root.get("links", {})

    if "rates" not in links or "orders" not in links:
        raise RuntimeError("Missing orders or rates API link")

    rates = fetch_json(links["rates"]).get("usd_per_unit", {})
    rows = None

    # Prefer the export endpoint if it returns a usable response.
    if links.get("export"):
        try:
            response = requests.get(links["export"], timeout=8)
            response.raise_for_status()

            try:
                payload = response.json()
                if isinstance(payload, list):
                    rows = payload
                elif isinstance(payload, dict):
                    rows = payload.get("orders", payload.get("data"))
                    if isinstance(rows, dict):
                        rows = rows.get("orders")
            except ValueError:
                # Also support newline-delimited JSON exports.
                rows = [
                    json.loads(line)
                    for line in response.text.splitlines()
                    if line.strip()
                ]

            if not isinstance(rows, list) or not rows:
                rows = None
        except Exception:
            rows = None

    # Fall back to fetching pages concurrently.
    if rows is None:
        orders_url = links["orders"]
        first_page = fetch_json(orders_url)
        page_count = int(first_page.get("pages", 1))

        urls = [
            page_url(orders_url, n)
            for n in range(2, page_count + 1)
        ]

        with ThreadPoolExecutor(max_workers=32) as pool:
            pages = list(pool.map(fetch_json, urls))

        rows = list(first_page.get("orders", []))
        for page in pages:
            rows.extend(page.get("orders", []))

    # Keep only the latest revision of each order ID.
    current = {}
    for row in rows:
        order_id = row.get("id")
        updated = parse_date(row.get("updated_at"))

        if order_id is None or updated is None:
            continue

        if (
            order_id not in current
            or updated > current[order_id][0]
        ):
            current[order_id] = (updated, row)

    return [item[1] for item in current.values()], rates


def parse_date(value):
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=IST)
        return parsed
    except (ValueError, TypeError):
        return None


def order_date(order, field="created_at"):
    value = parse_date(order.get(field))
    if value is None:
        return None
    return value.astimezone(IST)


def money(value):
    return Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )


def amount_usd(order, rates):
    currency = order.get("currency", "USD")
    rate = rates.get(currency)

    if rate is None:
        raise ValueError(f"Unknown currency: {currency}")

    return (
        Decimal(str(order.get("amount", 0)))
        * Decimal(str(rate))
    )


def parse_month(question):
    q = question.lower()
    months = {
        "january": 1, "february": 2, "march": 3,
        "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9,
        "october": 10, "november": 11, "december": 12,
    }

    for name, number in months.items():
        match = re.search(rf"\b{name}\s+(20\d{{2}})\b", q)
        if match:
            return int(match.group(1)), number

    match = re.search(r"\b(20\d{2})-(0?[1-9]|1[0-2])\b", q)
    if match:
        return int(match.group(1)), int(match.group(2))

    return None


def answer_question(question, orders, rates):
    q = question.lower().strip()

    if not q:
        raise ValueError("Question cannot be empty")

    selected = list(orders)

    # Region filter.
    regions = ("north", "south", "east", "west", "central")
    region = next(
        (r for r in regions if re.search(rf"\b{r}\b", q)),
        None,
    )
    if region:
        selected = [
            o for o in selected
            if str(o.get("region", "")).lower() == region
        ]

    # Product filter. Match the longest product name first.
    products = sorted(
        {
            str(o.get("product", ""))
            for o in orders
            if o.get("product")
        },
        key=len,
        reverse=True,
    )
    product = next(
        (p for p in products if p.lower() in q),
        None,
    )
    if product:
        selected = [
            o for o in selected
            if str(o.get("product", "")).lower() == product.lower()
        ]

    # Customer filter, such as C1221.
    customer_match = re.search(r"\bc\d+\b", q, re.I)
    if customer_match:
        customer = customer_match.group(0).upper()
        selected = [
            o for o in selected
            if str(o.get("customer", "")).upper() == customer
        ]

    # Month filter uses business-local time in Asia/Kolkata.
    month = parse_month(q)
    if month:
        year, month_number = month
        selected = [
            o for o in selected
            if (d := order_date(o)) is not None
            and d.year == year
            and d.month == month_number
        ]

    # Optional year-only filter.
    if not month:
        year_match = re.search(r"\b(20\d{2})\b", q)
        if year_match:
            year = int(year_match.group(1))
            selected = [
                o for o in selected
                if (d := order_date(o)) is not None
                and d.year == year
            ]

    # Status filter. "Revenue" always means paid orders.
    asks_paid = "paid" in q or any(
        phrase in q
        for phrase in ("revenue", "sales", "turnover")
    )
    asks_refunded = any(
        phrase in q
        for phrase in ("refund", "refunded")
    )
    asks_void = any(
        phrase in q
        for phrase in ("void", "voided", "cancelled", "canceled")
    )

    if asks_refunded:
        selected = [
            o for o in selected if o.get("status") == "refunded"
        ]
    elif asks_void:
        selected = [
            o for o in selected if o.get("status") == "void"
        ]
    elif asks_paid:
        selected = [
            o for o in selected if o.get("status") == "paid"
        ]

    # Revenue / sales totals.
    if any(word in q for word in ("revenue", "sales", "turnover")):
        total = sum(
            (amount_usd(o, rates) for o in selected),
            Decimal("0"),
        )
        return float(money(total))

    # Refund totals or refund counts.
    if asks_refunded:
        if any(word in q for word in ("how many", "count", "number of")):
            return len(selected)

        total = sum(
            (amount_usd(o, rates) for o in selected),
            Decimal("0"),
        )
        return float(money(total))

    # Distinct customer questions.
    if "customer" in q and any(
        word in q for word in ("how many", "count", "number of", "distinct")
    ):
        return len({
            o.get("customer")
            for o in selected
            if o.get("customer")
        })

    # Order counts.
    if "order" in q and any(
        word in q for word in ("how many", "count", "number of", "total")
    ):
        return len(selected)

    # Quantity sold, rather than number of order rows.
    if any(word in q for word in ("units sold", "items sold", "quantity sold")):
        return sum(int(o.get("qty", 0)) for o in selected)

    # Product ranking and product order counts.
    if (
        "product" in q
        or "best-selling" in q
        or "most popular" in q
        or "most orders" in q
    ):
        counts = Counter(
            o.get("product", "")
            for o in selected
            if o.get("product")
        )

        if counts:
            if any(word in q for word in ("least", "fewest")):
                return min(counts, key=counts.get)

            if any(word in q for word in ("most", "top", "best-selling")):
                return max(counts, key=counts.get)

            return dict(counts)

    # Customer revenue rankings and individual customer totals.
    if "customer" in q:
        totals = defaultdict(lambda: Decimal("0"))

        for order in selected:
            name = order.get("customer")
            if name and order.get("status") == "paid":
                totals[name] += amount_usd(order, rates)

        if totals and any(
            word in q for word in ("highest", "top", "most", "largest")
        ):
            return max(totals, key=totals.get)

        if customer_match:
            return float(money(totals.get(
                customer_match.group(0).upper(), Decimal("0")
            )))

    raise ValueError(
        "I couldn't confidently interpret this question."
    )


@app.get("/")
def home():
    return {"status": "Ledger Agent is running"}


@app.post("/")
def answer_endpoint(request: QuestionRequest):
    try:
        orders, rates = load_ledger()
        answer = answer_question(request.question, orders, rates)
        return {"answer": answer}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve or process ledger data.",
        )
