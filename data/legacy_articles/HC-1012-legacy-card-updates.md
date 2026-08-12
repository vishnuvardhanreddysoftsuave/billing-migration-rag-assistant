---
article_id: HC-1012
title: Updating a Card on the Legacy Billing Service
product_area: payments
last_updated: 2025-12-18
owner: Payments Support
audience: tier-1-support
---

# Updating a Card on the Legacy Billing Service

## Updating a stored card

On `billing-v1` a customer updates a stored card from Account → Billing → Card on file.
The card is tokenised at entry and the token replaces any previous one immediately, so
there is never more than one active card per account.

## Failed charges

A failed charge on the legacy service is retried nightly for 7 nights before the
subscription is suspended. There is no configurable retry schedule and agents cannot
trigger an out-of-band retry.

## Expiring cards

The legacy service emails the billing contact 30 days and 7 days before a stored card
expires. It does not use account updater services, so an expired card always requires
the customer to enter new details.
