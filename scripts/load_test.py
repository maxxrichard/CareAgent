#!/usr/bin/env python
"""Fire concurrent requests at a running API and print the live /stats afterwards."""

import argparse
import asyncio
import random

import httpx

from careagent.evaluation.dataset import load_cases


async def main(url: str, n: int, concurrency: int) -> None:
    cases = load_cases()
    sem = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, case) -> None:
        async with sem:
            r = await client.post(
                f"{url}/chat", json={"message": case.query, "customer_email": case.customer_email}
            )
            r.raise_for_status()

    async with httpx.AsyncClient(timeout=120) as client:
        await asyncio.gather(*(one(client, random.choice(cases)) for _ in range(n)))
        print((await client.get(f"{url}/stats")).json())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=5)
    a = ap.parse_args()
    asyncio.run(main(a.url, a.n, a.concurrency))
