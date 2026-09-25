"""End-to-end regression tests for extremely small (tiny) request deltas.

A request delta such as 1e-309 makes the naive ``1/delta`` overflow to
``inf``, which used to poison the shared RDP curve and wrongly reject the
submission as over budget.  The curve computation is numerically stable now,
so single requests, atomic batches, reservations and the read-only trials all
admit such requests when the finite projection fits the cap.
"""
from __future__ import annotations

import itertools
import math
import os
import tempfile

import pytest
import pytest_asyncio

# Point the ledger at an isolated database before importing the app.
_tmp = tempfile.mkdtemp(prefix="dpledger-test-tiny-delta-")
os.environ["DATABASE_PATH"] = os.path.join(_tmp, "tiny-delta.db")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app import db, rdp  # noqa: E402
from app.main import app  # noqa: E402

_counter = itertools.count()

TINY_DELTA = 1e-309


def _new_dataset_id(name: str) -> str:
    return f"dstiny-{name}-{next(_counter)}"


@pytest_asyncio.fixture()
async def client():
    await db.init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _register(client, dataset_id: str, eps: float = 1.0, delta: float = 1e-6):
    return await client.post(
        f"/datasets/{dataset_id}",
        json={"epsilon_cap": eps, "delta_cap": delta,
              "description": "test dataset"},
    )


@pytest.mark.asyncio
async def test_single_request_with_tiny_delta_is_granted(client):
    ds = _new_dataset_id("single")
    await _register(client, ds, eps=1.0, delta=1e-6)

    r = await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": "r1", "query_type": "gaussian_mean",
              "epsilon": 0.1, "delta": TINY_DELTA},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["allowed"] is True
    assert body["status"] == "granted"
    assert math.isfinite(body["epsilon_after"])
    assert body["epsilon_after"] == pytest.approx(
        rdp.evaluate_entries([(0.1, TINY_DELTA)], 1e-6)[1]
    )

    # The budget view recomputes with the corrected occupancy and stays finite.
    budget = (await client.get(f"/datasets/{ds}/budget")).json()
    assert math.isfinite(budget["epsilon_consumed"])
    assert budget["epsilon_consumed"] == pytest.approx(body["epsilon_after"])
    assert all(math.isfinite(v) for v in budget["rdp_curve"].values())

    # Same-key replay returns the recorded verdict unchanged.
    replay = await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": "r1", "query_type": "gaussian_mean",
              "epsilon": 0.1, "delta": TINY_DELTA},
    )
    assert replay.status_code == 200
    assert replay.json()["idempotent_replay"] is True
    assert replay.json()["epsilon_after"] == body["epsilon_after"]


@pytest.mark.asyncio
async def test_tiny_delta_still_rejected_when_projection_exceeds_cap(client):
    ds = _new_dataset_id("toobig")
    await _register(client, ds, eps=1.0, delta=1e-6)

    r = await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": "r1", "query_type": "gaussian_mean",
              "epsilon": 2.0, "delta": TINY_DELTA},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["allowed"] is False
    assert body["status"] == "rejected"
    assert "budget exceeded" in body["reason"]
    # A rejection consumes nothing.
    budget = (await client.get(f"/datasets/{ds}/budget")).json()
    assert budget["epsilon_consumed"] == 0.0


@pytest.mark.asyncio
async def test_batch_with_tiny_delta_items_is_granted(client):
    ds = _new_dataset_id("batch")
    await _register(client, ds, eps=1.0, delta=1e-6)

    r = await client.post(
        f"/datasets/{ds}/request-batches",
        json={
            "batch_id": "batch-1",
            "items": [
                {"request_id": "b1", "query_type": "gaussian_mean",
                 "epsilon": 0.1, "delta": TINY_DELTA},
                {"request_id": "b2", "query_type": "histogram",
                 "epsilon": 0.1, "delta": 1e-8},
            ],
        },
    )
    assert r.status_code == 201
    body = r.json()
    assert body["allowed"] is True
    assert body["status"] == "granted"
    assert math.isfinite(body["epsilon_after"])
    expected = rdp.evaluate_entries(
        [(0.1, TINY_DELTA), (0.1, 1e-8)], 1e-6
    )[1]
    assert body["epsilon_after"] == pytest.approx(expected)


@pytest.mark.asyncio
async def test_reservation_with_tiny_delta_is_granted_and_confirmable(client):
    ds = _new_dataset_id("reservation")
    await _register(client, ds, eps=1.0, delta=1e-6)

    r = await client.post(
        f"/datasets/{ds}/reservations",
        json={"reservation_id": "res-1", "epsilon": 0.1,
              "delta": TINY_DELTA, "ttl_seconds": 600},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["allowed"] is True
    assert body["status"] == "granted"
    assert math.isfinite(body["epsilon_after"])

    confirm = await client.post(
        f"/datasets/{ds}/reservations/res-1/confirm",
        json={"request_id": "req-from-res", "query_type": "gaussian_mean"},
    )
    assert confirm.status_code == 201
    confirmed = confirm.json()
    assert confirmed["allowed"] is True
    assert math.isfinite(confirmed["epsilon_after"])
    # Confirming does not charge the budget twice.
    assert confirmed["epsilon_after"] == pytest.approx(body["epsilon_after"])


@pytest.mark.asyncio
async def test_read_only_trials_with_tiny_delta_are_finite(client):
    ds = _new_dataset_id("trials")
    await _register(client, ds, eps=1.0, delta=1e-6)

    plan = await client.post(
        f"/datasets/{ds}/selection-plans",
        json={"candidates": [
            {"candidate_id": "c1", "epsilon": 0.1, "delta": TINY_DELTA,
             "value": 5},
        ]},
    )
    assert plan.status_code == 200
    plan_body = plan.json()
    assert plan_body["selected_candidate_ids"] == ["c1"]
    assert math.isfinite(plan_body["projected_epsilon"])

    trial = await client.post(
        f"/datasets/{ds}/repeat-run-trials",
        json={"items": [{"epsilon": 0.1, "delta": TINY_DELTA}],
              "max_runs": 10},
    )
    assert trial.status_code == 200
    trial_body = trial.json()
    assert trial_body["max_complete_runs"] >= 1
    assert math.isfinite(trial_body["projection"]["epsilon_projected"])
