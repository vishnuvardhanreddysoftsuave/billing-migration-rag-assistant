---
article_id: HC-4001
title: Billing Migration Overview and Cutover Timeline
product_area: billing-migration
last_updated: 2026-07-30
owner: Billing Platform Support
audience: tier-1-support
---

# Billing Migration Overview and Cutover Timeline

## What is changing

We are moving every account off the legacy billing service (internally `billing-v1`) and
onto the new metering and invoicing platform (`billing-v2`). The migration changes where
invoices are generated, how payment methods are stored, and which API endpoints accept
writes. It does **not** change contract terms, plan pricing, or tax registration numbers.

Customers see three visible differences after their account is cut over: invoice numbers
gain a `B2-` prefix, the billing portal shows a new "Payment methods" tab backed by the
new card vault, and statements are issued in the account's registered billing currency
rather than the workspace currency.

## Cutover timeline

The migration runs in four phases. All timestamps are UTC. Support agents should quote
these dates verbatim; do not estimate a customer's phase from their invoice history.

| Phase | Window | What happens | Customer impact |
|---|---|---|---|
| Phase 1 — Shadow | 2026-06-01 to 2026-07-14 | `billing-v2` computes invoices in parallel; nothing is issued | None |
| Phase 2 — Opt-in | 2026-07-15 to 2026-08-13 | Accounts may self-migrate from the billing portal | Optional |
| Phase 3 — Cutover | 2026-08-14 | The legacy billing endpoint stops accepting writes at 23:59 UTC on 2026-08-14 | Writes fail with ERR-4001 |
| Phase 4 — Decommission | 2026-09-30 | `billing-v1` is switched off and its data is archived | Read access ends |

Between Phase 3 and Phase 4 the legacy service remains **read-only**: historical invoices
and statements can still be downloaded, but no new charges, refunds, or payment-method
updates are accepted there.

## Troubleshooting: general migration errors

| Error code | Cause | Fix |
|---|---|---|
| ERR-4001 | A write was sent to the legacy billing endpoint after the Phase 3 cutover | Re-send the request to `https://api.example.com/v2/billing`; the legacy host is read-only after 2026-08-14 |
| ERR-4002 | The account is mid-migration and is locked for the duration of the copy | Wait for the lock to clear; migration locks are released automatically within 30 minutes |
| ERR-4003 | The account has no billing currency set, which `billing-v2` requires | Set the billing currency in Account → Billing → Preferences, then retry the migration |
| ERR-4004 | Two migration jobs were queued for the same account | Cancel the newer job in the migration console; running both corrupts the invoice sequence |
| ERR-4005 | The account's legacy data failed the pre-migration integrity check | Open a Billing Platform ticket with the account ID; do not retry the migration manually |
| ERR-4006 | A migration was attempted before the account's Phase 2 opt-in window opened | Wait for the account's scheduled window; windows are listed in the migration console |
| ERR-4007 | The workspace owner's email is unverified, so migration notices cannot be delivered | Ask the owner to verify their email, then re-queue the migration |
| ERR-4008 | The account holds an open dispute that blocks the ledger copy | Resolve or withdraw the dispute first; the ledger cannot be copied while amounts are contested |

## Migration locks

While an account is being copied it is locked. A locked account rejects writes with
ERR-4002 and shows a banner in the billing portal. Locks are released automatically
within 30 minutes. If a lock is still present after 30 minutes, the migration job has
stalled and needs a Billing Platform ticket — support agents cannot clear locks.

## What to tell customers

Tell customers that no action is required unless they have a stored card issued before
2026-05-01 (see HC-4002) or a VAT ID on file (see HC-4005). Everything else is copied
automatically. If a customer asks whether they can stay on `billing-v1`, the answer is
no: Phase 4 decommissions the service on 2026-09-30 for all accounts.
