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
TIMEOUT = 12


class QuestionRequest(BaseModel):
    question: str


def fetch_json(url):
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def page_url(url, page_number):
    parts = urlsplit(url)
    params = parse_qsl(parts.query, keep_blank_values=True)

    updated = []
    found = False

    for key, value in params:
        if key == "page":
            updated.append((key, str(page_number)))
            found = True
        else:
            updated.append((key, value))

    if not found:
        updated.append(("page", str(page_number)))

    return urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path,
        urlencode(updated),
        parts.fragment,
    ))


def parse_date(value):
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
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


@lru_cache(maxsize=1)
def load_ledger():
    root_url = os.environ.get("QUESTION_DATA_URL")
    if not root_url:
        raise RuntimeError("QUESTION_DATA_URL is not configured")

    root = fetch_json(root_url)
    links = root.get("links", {})

    if "orders" not in links or "rates" not in links:
        raise RuntimeError("Missing orders or rates API link")

    rates_data = fetch_json(links["rates"])
    rates = rates_data.get("usd_per_unit", {})

    rows = None

    # Try the full export first to reduce loading time.
    if links.get("export"):
        try:
            response = requests.get(links["export"], timeout=TIMEOUT)
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
                # Support newline-delimited JSON exports.
                rows = [
                    json.loads(line)
                    for line in response.text.splitlines()
                    if line.strip()
                ]

            if not isinstance(rows, list) or not rows:
                rows = None

        except Exception:
            rows = None

    # If export is unavailable, retrieve the paginated order data.
    if rows is None:
        orders_url = links["orders"]
        first_page = fetch_json(orders_url)

        rows = list(first_page.get("orders", []))
        page_count = int(first_page.get("pages", 1))

        urls = [
            page_url(orders_url, page_number)
            for page_number in range(2, page_count + 1)
        ]

        with ThreadPoolExecutor(max_workers=16) as pool:
            for page in pool.map(fetch_json, urls):
                rows.extend(page.get("orders", []))

    # Keep the latest revision of each order ID.
    current = {}

    for order in rows:
        order_id = order.get("id")
        updated = parse_date(order.get("updated_at"))

        if order_id is None or updated is None:
            continue

        if (
            order_id not in current
            or updated > current[order_id][0]
        ):
            current[order_id] = (updated, order)

    return [entry[1] for entry in current.values()], rates


def money(value):
    return Decimal(str(value)).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )


def amount_usd(order, rates):
    currency = str(order.get("currency", "USD")).upper()
    rate = rates.get(currency)

    if rate is None:
        raise ValueError(f"Unknown currency: {currency}")

    amount = Decimal(str(order.get("amount", 0)))
    return amount * Decimal(str(rate))


def parse_month(question):
    q = question.lower()

    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
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
            order for order in selected
            if str(order.get("region", "")).lower() == region
        ]

    # Product filter. Match the longest product name first.
    products = sorted(
        {
            str(order.get("product", ""))
            for order in orders
            if order.get("product")
        },
        key=len,
        reverse=True,
    )

    product = next(
        (name for name in products if name.lower() in q),
        None,
    )

    if product:
        selected = [
            order for order in selected
            if str(order.get("product", "")).lower()
            == product.lower()
        ]

    # Customer filter, such as C1221.
    customer_match = re.search(r"\bc\d+\b", q, re.I)

    if customer_match:
        customer_id = customer_match.group(0).upper()
        selected = [
            order for order in selected
            if str(order.get("customer", "")).upper() == customer_id
        ]

    # Month/year filters use Asia/Kolkata business dates.
    month = parse_month(q)

    if month:
        year, month_number = month
        selected = [
            order for order in selected
            if (date := order_date(order)) is not None
            and date.year == year
            and date.month == month_number
        ]
    else:
        year_match = re.search(r"\b(20\d{2})\b", q)

        if year_match:
            year = int(year_match.group(1))
            selected = [
                order for order in selected
                if (date := order_date(order)) is not None
                and date.year == year
            ]

    # Status rules.
    asks_refunded = any(
        word in q for word in ("refund", "refunded", "refunds")
    )
    asks_void = any(
        word in q
        for word in ("void", "voided", "cancelled", "canceled")
    )
    asks_paid = (
        "paid" in q
        or any(
            word in q
            for word in (
                "revenue", "sales", "turnover", "earned"
            )
        )
    )

    if asks_refunded:
        selected = [
            order for order in selected
            if order.get("status") == "refunded"
        ]
    elif asks_void:
        selected = [
            order for order in selected
            if order.get("status") == "void"
        ]
    elif asks_paid:
        selected = [
            order for order in selected
            if order.get("status") == "paid"
        ]

    # Average paid order value in USD.
    if any(word in q for word in ("average", "avg", "mean")):
        paid_orders = [
            order for order in selected
            if order.get("status") == "paid"
        ]

        if not paid_orders:
            return 0.0

        total = sum(
            (amount_usd(order, rates) for order in paid_orders),
            Decimal("0"),
        )

        average = total / Decimal(len(paid_orders))
        return float(money(average))

        # Find the top-selling product by revenue.
    if (
        "product" in q
        and any(word in q for word in ("revenue", "sales", "turnover", "earned"))
        and any(word in q for word in ("top", "most", "highest", "best"))
    ):
        product_revenue = defaultdict(lambda: Decimal("0"))

        for order in selected:
            if order.get("status") == "paid":
                name = order.get("product")
                if name:
                    product_revenue[name] += amount_usd(order, rates)

        if product_revenue:
            return max(product_revenue, key=product_revenue.get)
    
    # Revenue/sales totals: paid orders only.
    if any(
        word in q
        for word in ("revenue", "sales", "turnover", "earned")
    ):
        paid_orders = [
            order for order in selected
            if order.get("status") == "paid"
        ]

        total = sum(
            (amount_usd(order, rates) for order in paid_orders),
            Decimal("0"),
        )
        return float(money(total))

    # Refund totals or refund counts.
    if asks_refunded:
        if any(word in q for word in ("how many", "count", "number of")):
            return len(selected)

        total = sum(
            (amount_usd(order, rates) for order in selected),
            Decimal("0"),
        )

        return float(money(total))

    # Distinct customer counts.
    if "customer" in q and any(
        word in q
        for word in ("how many", "count", "number of", "distinct")
    ):
        return len({
            order.get("customer")
            for order in selected
            if order.get("customer")
        })

    # Order counts.
    if "order" in q and any(
        word in q
        for word in ("how many", "count", "number of", "total")
    ):
        return len(selected)

    # Quantity sold.
    if any(
        phrase in q
        for phrase in ("units sold", "items sold", "quantity sold")
    ):
        return sum(int(order.get("qty", 0)) for order in selected)

    # Product ranking and product order counts.
    if (
        "product" in q
        or "best-selling" in q
        or "most popular" in q
        or "most orders" in q
    ):
        counts = Counter(
            order.get("product", "")
            for order in selected
            if order.get("product")
        )

        if counts:
            if any(word in q for word in ("least", "fewest")):
                return min(counts, key=counts.get)

            if any(word in q for word in ("most", "top", "best-selling")):
                return max(counts, key=counts.get)

            return dict(counts)

    # Customer revenue totals and rankings.
    if "customer" in q:
        totals = defaultdict(lambda: Decimal("0"))

        for order in selected:
            customer = order.get("customer")

            if customer and order.get("status") == "paid":
                totals[customer] += amount_usd(order, rates)

        if totals and any(
            word in q for word in ("highest", "top", "most", "largest")
        ):
            return max(totals, key=totals.get)

        if customer_match:
            return float(money(
                totals.get(
                    customer_match.group(0).upper(),
                    Decimal("0"),
                )
            ))

    raise ValueError("I couldn't confidently interpret this question.")


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

