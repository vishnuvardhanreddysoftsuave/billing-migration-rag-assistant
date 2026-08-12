---
article_id: HC-4003
title: Invoice and Statement Discrepancies After Migration
product_area: invoicing
last_updated: 2026-07-27
owner: Invoicing Support
audience: tier-1-support
---

# Invoice and Statement Discrepancies After Migration

## Why the first post-migration invoice looks different

The first invoice issued by `billing-v2` covers the period from the account's cutover
timestamp to the end of the billing cycle, so it is almost always shorter than a full
month. It carries a `B2-` prefix, restarts the invoice sequence at `B2-000001` per
account, and lists any migrated credit as a separate line item.

Legacy credit balances are not paid out. A remaining balance is carried over as a single
line item named "migration credit" and applied to the first invoice issued after cutover.
Balances below 1.00 in the account's billing currency are written off rather than
carried, because the new ledger does not track sub-unit residuals.

## Troubleshooting: invoice errors

| Error code | Cause | Fix |
|---|---|---|
| ERR-4085 | The invoice period overlaps a period already invoiced by the legacy service | Void the `billing-v2` invoice and re-issue with the cutover timestamp as the period start |
| ERR-4086 | The account's billing cycle anchor day does not exist in the target month (for example, the 31st) | Move the anchor to the 28th in Account → Billing → Preferences; `billing-v2` does not clamp anchors automatically |
| ERR-4087 | A line item references a plan that was not mapped during migration | Map the plan in the migration console (see HC-4004), then re-generate the invoice |
| ERR-4088 | A duplicate invoice number was generated because the legacy sequence was replayed into the new ledger | Void the duplicate and re-issue; the `B2-` sequence must be contiguous per account and never reuses a number |
| ERR-4089 | The migration credit line item exceeds the invoice total, producing a negative amount due | Split the credit: apply it up to the invoice total and carry the remainder to the next invoice |
| ERR-4090 | The invoice PDF failed to render because the legacy template referenced a retired logo asset | Re-render with the `b2-default` template; legacy templates are not supported after cutover |
| ERR-4091 | The billing contact on the invoice is an address that no longer exists in the workspace | Set a current billing contact in Account → Billing → Contacts, then re-send the invoice |
| ERR-4092 | Two invoices claim the same period because a migration was run twice for the account | Void both, run the ledger repair job, then issue a single corrected invoice |
| ERR-4093 | The statement total disagrees with the sum of its line items after currency conversion | Re-run the statement; conversion uses the rate at issue time and must not be recalculated later |

## Reading a migrated statement

A migrated statement has three sections: charges accrued in `billing-v1` before cutover,
charges accrued in `billing-v2` after cutover, and adjustments. The adjustments section is
where the migration credit appears. If a customer says their statement "double counts"
usage, check the cutover timestamp first — an overlap of even one hour produces ERR-4085
and is the most common cause of an inflated total.

## What agents may and may not do

Agents may void and re-issue an invoice, re-render a PDF, and correct a billing contact.
Agents may not edit a line item amount, change a period start, or delete an invoice: the
new ledger is append-only and every correction must exist as a void plus a re-issue so the
audit trail stays intact.
