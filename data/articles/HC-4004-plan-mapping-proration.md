---
article_id: HC-4004
title: Subscription Plan Mapping and Proration
product_area: subscriptions
last_updated: 2026-07-26
owner: Subscriptions Support
audience: tier-1-support
---

# Subscription Plan Mapping and Proration

## How legacy plans map to new tiers

Every `billing-v1` plan is mapped to exactly one `billing-v2` tier. Mappings are defined
per price book, not per account, so two customers on the same legacy plan always land on
the same new tier. Custom-priced contracts keep their negotiated amount: the tier changes,
the price does not.

Where a legacy plan has no equivalent (typically retired add-ons), the migration creates a
"legacy compatibility" tier that preserves the price and entitlements for the remainder of
the contract term. Compatibility tiers cannot be renewed — at renewal the account must
move to a current tier.

## How proration is calculated

When a subscription is mapped to a new tier part-way through a billing cycle, the unused
portion of the legacy plan is credited at its daily rate and the new tier is charged pro
rata from the mapping date. Proration is always calculated on a 30-day month regardless of
the calendar length of the month, so a mid-February mapping and a mid-March mapping produce
the same daily rate.

Two consequences follow, and both generate support contacts:

- A customer mapped on the 31st of a 31-day month sees 30 days billed, not 31.
- A customer mapped in February sees a daily rate lower than `monthly / 28`.

Neither is a defect. If a customer disputes the arithmetic, quote the 30-day rule and show
the credit line and the charge line separately on the invoice.

## Troubleshooting: plan mapping errors

| Error code | Cause | Fix |
|---|---|---|
| ERR-4150 | The legacy plan has no mapping in the account's price book | Add the mapping in the migration console, then re-run the subscription step only |
| ERR-4151 | The legacy plan maps to two tiers because the price book was edited mid-migration | Freeze the price book, delete the newer mapping, and re-run the subscription step |
| ERR-4152 | The subscription is in a trial that ends after the cutover date | Let the trial finish; trialling subscriptions are migrated on the trial end date, not at cutover |
| ERR-4153 | The subscription has a scheduled plan change queued in the legacy service | Cancel the scheduled change, migrate, then re-schedule it in `billing-v2` |
| ERR-4154 | Proration produced a credit larger than the remaining contract value | Cap the credit at the remaining contract value and raise a Billing Platform ticket with the account ID |
| ERR-4155 | The target tier is deprecated and closed to new subscriptions | Map to the current replacement tier listed in the price book, not to the deprecated tier |
| ERR-4156 | The subscription quantity exceeds the seat ceiling of the target tier | Map to the next tier up, or reduce seats before migrating; `billing-v2` refuses to over-subscribe a tier |
| ERR-4157 | A compatibility tier was created for an account whose contract has already expired | Move the account to a current tier; expired contracts are not eligible for compatibility tiers |

## Cancellations during migration

A subscription cancelled before cutover is not migrated at all — it stays in `billing-v1`
history. A subscription cancelled after cutover follows the normal `billing-v2` rules:
service continues to the end of the paid period and no proration credit is issued for a
voluntary cancellation.
