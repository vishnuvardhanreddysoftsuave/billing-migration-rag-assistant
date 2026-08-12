---
article_id: HC-4006
title: API and Webhook Changes for the Billing Migration
product_area: developer-api
last_updated: 2026-07-31
owner: Developer Support
audience: tier-2-support
---

# API and Webhook Changes for the Billing Migration

## Required API version

Clients calling the billing API after migration must send the `2026-06-01` API version in
the `X-Api-Version` header. Requests that omit the header are pinned to the caller's
account-default version, which for migrated accounts is also `2026-06-01`. Requests that
send an older version against a migrated account are rejected with ERR-4200 — there is no
compatibility shim for pre-migration versions.

The breaking changes in `2026-06-01` are: invoice identifiers are strings rather than
integers, amounts are minor units rather than decimal strings, and the `plan` field is
replaced by `tier`.

## Webhook delivery and the retry window

Webhook endpoints are unchanged, but delivery guarantees are not. A failed webhook
delivery is retried with exponential backoff for up to 24 hours, across 6 attempts, after
which the event is dropped and recorded as undeliverable in the developer console. The
legacy service retried for 72 hours; integrations that relied on the longer window need to
consume the replay endpoint instead.

Events are signed with a per-endpoint v2 signing secret. The legacy shared secret is not
carried over.

## Troubleshooting: API and webhook errors

| Error code | Cause | Fix |
|---|---|---|
| ERR-4200 | The request sent an API version older than `2026-06-01` against a migrated account | Send `X-Api-Version: 2026-06-01`; older versions are not served for migrated accounts |
| ERR-4201 | The request used a legacy invoice identifier as an integer | Send the identifier as a string, including the `B2-` prefix |
| ERR-4202 | The request sent an amount as a decimal string rather than minor units | Send minor units as an integer (for example 1050 for 10.50) |
| ERR-4203 | The webhook payload was signed with the legacy shared secret, so the signature did not match | Rotate to the v2 signing secret in Developer → Webhooks and verify with HMAC-SHA256 over the raw request body |
| ERR-4204 | The webhook endpoint responded with a non-2xx status six times, exhausting the retry window | Fix the endpoint, then replay the events from Developer → Webhooks → Replay; retries are not resumed automatically |
| ERR-4205 | The request referenced a `plan` field that no longer exists in `2026-06-01` | Use the `tier` field; `plan` was removed, not deprecated |
| ERR-4206 | The webhook signature was verified against a parsed and re-serialised body | Verify against the raw body bytes; re-serialising changes whitespace and breaks the HMAC |
| ERR-4207 | The API key was issued by the legacy service and has no `billing-v2` scope | Issue a new key in Developer → API keys; legacy keys are not re-scoped automatically |
| ERR-4208 | The endpoint acknowledged the delivery after the 30-second timeout | Acknowledge within 30 seconds and process asynchronously; slow acknowledgements are counted as failures |

## Replaying events

The replay endpoint holds undeliverable events for 30 days. Replay is per-event, not
per-endpoint: selecting an endpoint and a time range replays every undeliverable event in
that range, including ones the integration may have already processed out of band, so
consumers must be idempotent.

## Sandbox differences

Sandbox accounts were migrated on 2026-06-15, ahead of production. A sandbox integration
that works today can still fail in production if the production account has not reached
its Phase 3 cutover — check the account's phase before debugging the integration.
