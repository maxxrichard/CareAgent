"""SQLite-backed order store with a deterministic synthetic seed.

The public customer-support dataset we use for intents and phrasing does not include
structured order records, so orders are synthesised with a fixed seed. Every order is
reproducible from (seed, index), which lets the evaluation set reference real order IDs.
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from careagent.config import get_settings
from careagent.models import Order, OrderItem, OrderStatus

CATALOGUE: list[tuple[str, str, str, float]] = [
    ("ELX-1001", "Noise-cancelling headphones", "electronics", 149.00),
    ("ELX-1002", "USB-C charging hub", "electronics", 39.90),
    ("ELX-1003", "Smart speaker", "electronics", 89.00),
    ("CLO-2001", "Merino wool sweater", "clothing", 79.00),
    ("CLO-2002", "Running shoes", "clothing", 119.00),
    ("CLO-2003", "Rain jacket", "clothing", 99.00),
    ("HOM-3001", "Cast-iron skillet", "home", 59.00),
    ("HOM-3002", "Bedside lamp", "home", 34.50),
    ("HOM-3003", "Espresso machine", "home", 249.00),
    ("BEA-4001", "Vitamin C serum", "beauty", 24.90),
    ("BEA-4002", "Electric toothbrush", "beauty", 69.00),
    ("BOK-5001", "Hardcover cookbook", "books", 29.99),
    ("BOK-5002", "Paperback novel", "books", 12.99),
    ("PER-6001", "Artisan coffee beans 1 kg", "perishable", 22.00),
    ("PER-6002", "Indoor plant (monstera)", "perishable", 31.00),
]
CARRIERS = ["DHL", "DPD", "Hermes", "UPS", "GLS"]
FIRST_NAMES = [
    "anna",
    "ben",
    "clara",
    "david",
    "emma",
    "felix",
    "greta",
    "hannes",
    "ines",
    "jonas",
    "katrin",
    "lukas",
    "mia",
    "noah",
    "olivia",
    "paul",
    "quinn",
    "rosa",
    "sam",
    "tara",
]
DOMAINS = ["example.com", "example.org", "mail.example.net"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_email TEXT NOT NULL,
    status TEXT NOT NULL,
    placed_on TEXT NOT NULL,
    estimated_delivery TEXT,
    delivered_on TEXT,
    carrier TEXT,
    tracking_number TEXT,
    items_json TEXT NOT NULL,
    total_eur REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_email ON orders(customer_email);
"""


def synthesize_order(i: int, rng: random.Random, today: date | None = None) -> Order:
    today = today or date.today()
    placed = today - timedelta(days=rng.randint(1, 60))
    n_items = rng.choice([1, 1, 1, 2, 2, 3])
    items = []
    for sku, name, cat, price in rng.sample(CATALOGUE, n_items):
        items.append(
            OrderItem(
                sku=sku,
                name=name,
                category=cat,
                quantity=rng.choice([1, 1, 2]),
                unit_price_eur=price,
            )
        )
    total = round(sum(it.quantity * it.unit_price_eur for it in items), 2)
    status = rng.choices(
        list(OrderStatus),
        weights=[10, 20, 8, 40, 7, 5, 6, 4],
    )[0]
    eta = placed + timedelta(days=rng.randint(2, 6))
    delivered = None
    carrier = None
    tracking = None
    if status in {
        OrderStatus.SHIPPED,
        OrderStatus.OUT_FOR_DELIVERY,
        OrderStatus.DELIVERED,
        OrderStatus.DELAYED,
        OrderStatus.RETURN_REQUESTED,
        OrderStatus.REFUNDED,
    }:
        carrier = rng.choice(CARRIERS)
        tracking = f"{carrier[:2].upper()}{rng.randint(10**9, 10**10 - 1)}"
    if status in {OrderStatus.DELIVERED, OrderStatus.RETURN_REQUESTED, OrderStatus.REFUNDED}:
        delivered = min(eta + timedelta(days=rng.randint(-1, 2)), today)
    if status == OrderStatus.DELAYED:
        eta = max(eta, today - timedelta(days=rng.randint(1, 4)))
    email = f"{rng.choice(FIRST_NAMES)}.{rng.choice(FIRST_NAMES)}{rng.randint(1, 99)}@{rng.choice(DOMAINS)}"
    return Order(
        order_id=f"ORD-{100000 + i}",
        customer_email=email,
        status=status,
        placed_on=placed,
        estimated_delivery=eta,
        delivered_on=delivered,
        carrier=carrier,
        tracking_number=tracking,
        items=items,
        total_eur=total,
    )


class OrderStore:
    """Thin repository over SQLite. Use `OrderStore.seeded()` for a reproducible demo DB."""

    def __init__(self, db_path: Path | str = ":memory:"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)

    @classmethod
    def seeded(
        cls,
        db_path: Path | str | None = None,
        n: int | None = None,
        seed: int | None = None,
        today: date | None = None,
    ) -> OrderStore:
        settings = get_settings()
        store = cls(db_path if db_path is not None else settings.orders_db)
        if store.count() == 0:
            rng = random.Random(seed if seed is not None else settings.seed)
            for i in range(n or settings.seed_orders):
                store.upsert(synthesize_order(i, rng, today))
        return store

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]

    def upsert(self, order: Order) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                order.order_id,
                order.customer_email,
                order.status.value,
                order.placed_on.isoformat(),
                order.estimated_delivery.isoformat() if order.estimated_delivery else None,
                order.delivered_on.isoformat() if order.delivered_on else None,
                order.carrier,
                order.tracking_number,
                json.dumps([it.model_dump() for it in order.items]),
                order.total_eur,
            ),
        )
        self._conn.commit()

    def get(self, order_id: str) -> Order | None:
        row = self._conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id.strip().upper(),)
        ).fetchone()
        return self._row_to_order(row) if row else None

    def find_by_email(self, email: str, limit: int = 5) -> list[Order]:
        rows = self._conn.execute(
            "SELECT * FROM orders WHERE lower(customer_email) = ? ORDER BY placed_on DESC LIMIT ?",
            (email.strip().lower(), limit),
        ).fetchall()
        return [self._row_to_order(r) for r in rows]

    def all(self) -> list[Order]:
        return [
            self._row_to_order(r) for r in self._conn.execute("SELECT * FROM orders").fetchall()
        ]

    def update_status(self, order_id: str, status: OrderStatus) -> bool:
        cur = self._conn.execute(
            "UPDATE orders SET status = ? WHERE order_id = ?", (status.value, order_id)
        )
        self._conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _row_to_order(row: tuple) -> Order:
        (oid, email, status, placed, eta, delivered, carrier, tracking, items_json, total) = row
        return Order(
            order_id=oid,
            customer_email=email,
            status=OrderStatus(status),
            placed_on=date.fromisoformat(placed),
            estimated_delivery=date.fromisoformat(eta) if eta else None,
            delivered_on=date.fromisoformat(delivered) if delivered else None,
            carrier=carrier,
            tracking_number=tracking,
            items=[OrderItem(**it) for it in json.loads(items_json)],
            total_eur=total,
        )
