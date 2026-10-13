"""Order-status lookup tool."""

from __future__ import annotations

from pydantic_ai import RunContext

from careagent.models import OrderLookupResult
from careagent.tools.deps import AgentDeps


def lookup_order(ctx: RunContext[AgentDeps], order_id: str) -> OrderLookupResult:
    """Look up the status, items, carrier, tracking number and delivery dates of an order.

    Args:
        order_id: The public order number the customer provided, e.g. "ORD-100042".
    """
    ctx.deps.record("lookup_order")
    order = ctx.deps.orders.get(order_id)
    if order is None:
        return OrderLookupResult(
            found=False,
            message=f"No order with number {order_id!r} was found. Ask the customer to double-check it.",
        )
    if ctx.deps.customer_email and order.customer_email.lower() != ctx.deps.customer_email.lower():
        # Never leak another customer's order details.
        return OrderLookupResult(
            found=False,
            message="This order does not belong to the authenticated customer. Do not reveal its details.",
        )
    ctx.deps.evidence.add_order(order)
    return OrderLookupResult(found=True, order=order, message="Order found.")


def list_recent_orders(ctx: RunContext[AgentDeps]) -> list[str]:
    """List the order numbers of the authenticated customer's most recent orders.

    Use this when the customer asks about "my order" without giving an order number.
    """
    ctx.deps.record("list_recent_orders")
    if not ctx.deps.customer_email:
        return []
    orders = ctx.deps.orders.find_by_email(ctx.deps.customer_email)
    for o in orders:
        ctx.deps.evidence.add_order(o)
    return [f"{o.order_id} ({o.status.value}, placed {o.placed_on.isoformat()})" for o in orders]
