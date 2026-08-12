---
article_id: HC-4005
title: Tax and VAT ID Validation Errors
product_area: tax-compliance
last_updated: 2026-07-25
owner: Tax Compliance Support
audience: tier-1-support
---

# Tax and VAT ID Validation Errors

## What the migration revalidates

`billing-v1` stored tax identifiers as free text and validated them once, at entry.
`billing-v2` revalidates every identifier against the issuing authority at migration time
and then re-checks it every 90 days. Identifiers that were accepted years ago can therefore
fail during migration — usually because the registration lapsed, not because the number was
typed wrongly.

EU VAT IDs are checked against VIES. UK VAT numbers are checked against HMRC. US sales-tax
exemption certificates are not machine-validated: they are copied as-is and flagged for
manual review if their expiry date has passed.

## Troubleshooting: tax validation errors

| Error code | Cause | Fix |
|---|---|---|
| ERR-4115 | The account has no tax identifier but its billing country requires one | Ask the customer to add a tax ID in Account → Billing → Tax; invoices are blocked until one is present |
| ERR-4116 | The tax identifier's country prefix does not match the billing country | Correct whichever field is wrong; the prefix and the billing country must agree |
| ERR-4117 | The VAT ID was rejected by VIES because the registration is no longer active for cross-border trade | Ask the customer to confirm the number with their tax authority and re-enter it; if VIES is down, mark the account "validation pending" and re-run the check within 5 business days |
| ERR-4118 | The VAT ID is syntactically valid but belongs to a different legal entity than the billing name | Update the billing name to the registered entity name; VIES matches on entity, not on trading name |
| ERR-4119 | A US exemption certificate expired before the cutover date | Request a current certificate; expired certificates are not carried into `billing-v2` |
| ERR-4120 | The account is in a reverse-charge jurisdiction but has no VAT ID, so reverse charge cannot be applied | Add the VAT ID, or bill with local VAT applied; reverse charge without an ID is not permitted |
| ERR-4121 | Two accounts in the same workspace claim the same VAT ID | Consolidate the accounts or correct the duplicate; a VAT ID may back one billing account only |
| ERR-4122 | The VIES service returned a transient error three times in a row | Mark the account "validation pending" and retry after 24 hours; do not delete the identifier |

## Validation pending status

"Validation pending" is a holding state, not a failure. An account in that state is
invoiced normally with tax applied under the rules of its billing country, and the
identifier is re-checked automatically. If validation still fails after 5 business days,
the account moves to "validation failed" and reverse charge is switched off.

## Escalation

Escalate to Tax Compliance only when a customer insists their registration is active and
VIES disagrees for more than 5 business days. Include the VAT ID, the billing name as
registered, and the timestamp of the last failed check. Do not promise a tax credit for a
period already invoiced: adjustments to issued invoices need a tax review first.
