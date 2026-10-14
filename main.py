
import os
import re
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="TDS Ledger Agent")

IST = ZoneInfo("Asia/Kolkata")
TIMEOUT = 8


class QuestionRequest(BaseModel):
    question: str


def fetch_json(url):
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=1)
def load_ledger():
    root_url = os.environ.get("QUESTION_DATA_URL")
    if not root_url:
        raise RuntimeError("QUESTION_DATA_URL is not configured")

    root = fetch_json(root_url)
    links = root.get("links", {})

    if "orders" not in links or "rates" not in links:
        raise RuntimeError("API root is missing orders or rates links")

    first_page = fetch_json(links["orders"])
    pages = int(first_page.get("pages", 1))
    next_url = first_page.get("next")

    urls = []
    if next_url:
        urls.append(next_url)

    # Follow the API's pagination links.
    while len(urls) < pages - 1 and next_url:
        page = fetch_json(next_url)
        next_url = page.get("next")
        if next_url and next_url not in urls:
            urls.append(next_url)
        else:
            break

    with ThreadPoolExecutor(max_workers=16) as pool:
        page_results = list(pool.map(fetch_json, urls))

    rows = list(first_page.get("orders", []))
    for page in page_results:
        rows.extend(page.get("orders", []))

    # Keep the latest version of each order ID.
    current = {}
    for row in rows:
        order_id = row.get("id")
        updated = datetime.fromisoformat(
            row["updated_at"].replace("Z", "+00:00")
        )
        if order_id not in current:
            current[order_id] = (updated, row)
        elif updated > current[order_id][0]:
            current[order_id] = (updated, row)

    orders = [item[1] for item in current.values()]
    rates = fetch_json(links["rates"]).get("usd_per_unit", {})
    return orders, rates


def order_date(order, field="created_at"):
    value = order.get(field)
    if not value:
        return None
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    ).astimezone(IST)


def amount_usd(order, rates):
    currency = order.get("currency", "USD")
    rate = rates.get(currency)
    if rate is None:
        raise ValueError(f"Unknown currency: {currency}")
    return float(order.get("amount", 0)) * float(rate)


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
    return None


def answer_question(question, orders, rates):
    q = question.lower().strip()

    # Basic filters: region, product, and customer.
    selected = list(orders)

    regions = ("north", "south", "east", "west", "central")
    region = next((r for r in regions if re.search(rf"\b{r}\b", q)), None)
    if region:
        selected = [
            o for o in selected
            if o.get("region", "").lower() == region
        ]

    products = sorted(
        {o.get("product", "") for o in orders},
        key=len,
        reverse=True,
    )
    product = next(
        (p for p in products if p and p.lower() in q),
        None,
    )
    if product:
        selected = [
            o for o in selected
            if o.get("product", "").lower() == product.lower()
        ]

    customer_match = re.search(r"\bc\d+\b", q, re.I)
    if customer_match:
        customer = customer_match.group(0).upper()
        selected = [
            o for o in selected
            if o.get("customer", "").upper() == customer
        ]

    month = parse_month(q)
    if month:
        year, month_number = month
        selected = [
            o for o in selected
            if (d := order_date(o)) is not None
            and d.year == year
            and d.month == month_number
        ]

    # Revenue means paid orders only.
    if any(word in q for word in ("revenue", "sales", "turnover")):
        paid = [o for o in selected if o.get("status") == "paid"]
        total = round(sum(amount_usd(o, rates) for o in paid), 2)
        return total

    # Refund questions.
    if "refund" in q:
        refunded = [
            o for o in selected if o.get("status") == "refunded"
        ]
        if any(word in q for word in ("how many", "number of", "count")):
            return len(refunded)
        return round(sum(amount_usd(o, rates) for o in refunded), 2)

    # Counts.
    if any(word in q for word in ("how many", "count", "number of")):
        if "customer" in q and not customer_match:
            return len({o.get("customer") for o in selected})
        if "order" in q:
            return len(selected)
        return len(selected)

    # Product questions.
    if "product" in q or "best-selling" in q or "most popular" in q:
        counts = {}
        for o in selected:
            counts[o.get("product", "")] = (
                counts.get(o.get("product", ""), 0) + 1
            )
        if counts:
            if any(word in q for word in ("most", "top", "best-selling")):
                return max(counts, key=counts.get)
            if any(word in q for word in ("least", "fewest")):
                return min(counts, key=counts.get)
            return counts

    # Customer questions.
    if "customer" in q:
        totals = {}
        for o in selected:
            name = o.get("customer", "")
            totals[name] = totals.get(name, 0) + (
                amount_usd(o, rates) if o.get("status") == "paid" else 0
            )
        if totals and any(word in q for word in ("highest", "top", "most")):
            return max(totals, key=totals.get)
        if customer_match:
            return round(totals.get(customer_match.group(0).upper(), 0), 2)

    raise ValueError(
        "I couldn't confidently interpret this question yet."
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
