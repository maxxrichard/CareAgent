import random

from careagent.data.orders import OrderStore, synthesize_order
from careagent.models import OrderStatus
from tests.conftest import TODAY


def test_seed_is_deterministic():
    a = OrderStore.seeded(":memory:", n=20, seed=1, today=TODAY).all()
    b = OrderStore.seeded(":memory:", n=20, seed=1, today=TODAY).all()
    assert [o.order_id for o in a] == [o.order_id for o in b]
    assert [o.status for o in a] == [o.status for o in b]


def test_lookup_and_email_scoping(orders: OrderStore):
    first = orders.all()[0]
    assert orders.get(first.order_id) == first
    assert orders.get(first.order_id.lower()) == first
    assert orders.get("ORD-000000") is None
    assert first in orders.find_by_email(first.customer_email.upper())


def test_update_status(orders: OrderStore):
    oid = orders.all()[0].order_id
    assert orders.update_status(oid, OrderStatus.CANCELLED)
    assert orders.get(oid).status == OrderStatus.CANCELLED
    assert not orders.update_status("ORD-1", OrderStatus.CANCELLED)


def test_synthesized_order_consistency():
    rng = random.Random(3)
    for i in range(200):
        o = synthesize_order(i, rng, TODAY)
        assert o.total_eur == round(sum(it.quantity * it.unit_price_eur for it in o.items), 2)
        if o.delivered_on:
            assert o.delivered_on <= TODAY
            assert o.status in {
                OrderStatus.DELIVERED,
                OrderStatus.RETURN_REQUESTED,
                OrderStatus.REFUNDED,
            }
        if o.tracking_number:
            assert o.carrier
