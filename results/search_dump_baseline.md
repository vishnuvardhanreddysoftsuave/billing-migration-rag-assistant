# Search-only dump — `baseline`

Index namespace: `baseline__cs800_ov120` · 46 chunks · mean 690.1 chars

No generation is involved here: this is the raw top-5 for each of the eight questions.

## Q1 — What does ERR-4032 mean and what is the fix?

Gold: **HC-4002** / Troubleshooting: payment method errors · must contain ['ERR-4032', 're-authorise the card'] · **HIT at rank 2**

Diagnosis: answer-bearing chunk retrieved at rank 2

| Rank | Score | chunk_id | product_area | Article | Section | Preview |
|---|---|---|---|---|---|---|
| 1 | 0.0373 | `HC-4002#baseline-0007` | payments | HC-4002 | Dunning and the retry window | Dunning emails are sent after each failed attempt to the billing contact, not to the workspace owner, unless the two are the same person. ## Re-author… |
| 2 | 0.0341 | `HC-4002#baseline-0002` | payments | HC-4002 | Troubleshooting: payment method errors | \| ERR-4031 \| The card's expiry date has already passed, so the token cannot be re-tokenised \| Ask the customer to add a current card; expired cards… |
| 3 | 0.0247 | `HC-4003#baseline-0001` | invoicing | HC-4003 | Why the first post-migration invoice looks different | Legacy credit balances are not paid out. A remaining balance is carried over as a single line item named "migration credit" and applied to the first i… |
| 4 | 0.0241 | `HC-4001#baseline-0004` | billing-migration | HC-4001 | Cutover timeline | Between Phase 3 and Phase 4 the legacy service remains **read-only**: historical invoices and statements can still be downloaded, but no new charges, … |
| 5 | 0.0230 | `HC-4002#baseline-0001` | payments | HC-4002 | How payment methods are migrated | Bank debit mandates (SEPA, BACS, ACH) are migrated in full regardless of age, because the mandate reference, not a token, is the stored credential. Cu… |

## Q2 — A customer's VAT ID was rejected with ERR-4117. What causes it and how do we fix it?

Gold: **HC-4005** / Troubleshooting: tax validation errors · must contain ['ERR-4117', 'VIES'] · **HIT at rank 1**

Diagnosis: answer-bearing chunk retrieved at rank 1

| Rank | Score | chunk_id | product_area | Article | Section | Preview |
|---|---|---|---|---|---|---|
| 1 | 0.1630 | `HC-4005#baseline-0002` | tax-compliance | HC-4005 | Troubleshooting: tax validation errors | \| ERR-4116 \| The tax identifier's country prefix does not match the billing country \| Correct whichever field is wrong; the prefix and the billing … |
| 2 | 0.1221 | `HC-4005#baseline-0003` | tax-compliance | HC-4005 | Troubleshooting: tax validation errors | \| ERR-4118 \| The VAT ID is syntactically valid but belongs to a different legal entity than the billing name \| Update the billing name to the regis… |
| 3 | 0.0803 | `HC-4005#baseline-0004` | tax-compliance | HC-4005 | Troubleshooting: tax validation errors | \| ERR-4121 \| Two accounts in the same workspace claim the same VAT ID \| Consolidate the accounts or correct the duplicate; a VAT ID may back one bi… |
| 4 | 0.0795 | `HC-4005#baseline-0000` | tax-compliance | HC-4005 | Tax and VAT ID Validation Errors | # Tax and VAT ID Validation Errors ## What the migration revalidates `billing-v1` stored tax identifiers as free text and validated them once, at entr… |
| 5 | 0.0684 | `HC-4005#baseline-0001` | tax-compliance | HC-4005 | What the migration revalidates | EU VAT IDs are checked against VIES. UK VAT numbers are checked against HMRC. US sales-tax exemption certificates are not machine-validated: they are … |

## Q3 — What causes ERR-4203 on webhook delivery and how do we resolve it?

Gold: **HC-4006** / Troubleshooting: API and webhook errors · must contain ['ERR-4203', 'HMAC-SHA256'] · **HIT at rank 2**

Diagnosis: answer-bearing chunk retrieved at rank 2

| Rank | Score | chunk_id | product_area | Article | Section | Preview |
|---|---|---|---|---|---|---|
| 1 | 0.1310 | `HC-4006#baseline-0001` | developer-api | HC-4006 | Required API version | The breaking changes in `2026-06-01` are: invoice identifiers are strings rather than integers, amounts are minor units rather than decimal strings, a… |
| 2 | 0.0882 | `HC-4006#baseline-0003` | developer-api | HC-4006 | Troubleshooting: API and webhook errors | \| ERR-4202 \| The request sent an amount as a decimal string rather than minor units \| Send minor units as an integer (for example 1050 for 10.50) \… |
| 3 | 0.0825 | `HC-4006#baseline-0000` | developer-api | HC-4006 | API and Webhook Changes for the Billing Migration | # API and Webhook Changes for the Billing Migration ## Required API version Clients calling the billing API after migration must send the `2026-06-01`… |
| 4 | 0.0435 | `HC-4006#baseline-0004` | developer-api | HC-4006 | Troubleshooting: API and webhook errors | \| ERR-4205 \| The request referenced a `plan` field that no longer exists in `2026-06-01` \| Use the `tier` field; `plan` was removed, not deprecated… |
| 5 | 0.0280 | `HC-4001#baseline-0007` | billing-migration | HC-4001 | Troubleshooting: general migration errors | \| ERR-4008 \| The account holds an open dispute that blocks the ledger copy \| Resolve or withdraw the dispute first; the ledger cannot be copied whi… |

## Q4 — What does ERR-4088 mean on a migrated invoice and what should an agent do?

Gold: **HC-4003** / Troubleshooting: invoice errors · must contain ['ERR-4088', 'Void the duplicate'] · **HIT at rank 1**

Diagnosis: answer-bearing chunk retrieved at rank 1

| Rank | Score | chunk_id | product_area | Article | Section | Preview |
|---|---|---|---|---|---|---|
| 1 | 0.0571 | `HC-4003#baseline-0003` | invoicing | HC-4003 | Troubleshooting: invoice errors | \| ERR-4087 \| A line item references a plan that was not mapped during migration \| Map the plan in the migration console (see HC-4004), then re-gene… |
| 2 | 0.0454 | `HC-4003#baseline-0000` | invoicing | HC-4003 | Invoice and Statement Discrepancies After Migration | # Invoice and Statement Discrepancies After Migration ## Why the first post-migration invoice looks different The first invoice issued by `billing-v2`… |
| 3 | 0.0413 | `HC-4003#baseline-0004` | invoicing | HC-4003 | Troubleshooting: invoice errors | \| ERR-4090 \| The invoice PDF failed to render because the legacy template referenced a retired logo asset \| Re-render with the `b2-default` templat… |
| 4 | 0.0367 | `HC-4006#baseline-0002` | developer-api | HC-4006 | Webhook delivery and the retry window | Events are signed with a per-endpoint v2 signing secret. The legacy shared secret is not carried over. ## Troubleshooting: API and webhook errors \| E… |
| 5 | 0.0367 | `HC-4006#baseline-0000` | developer-api | HC-4006 | API and Webhook Changes for the Billing Migration | # API and Webhook Changes for the Billing Migration ## Required API version Clients calling the billing API after migration must send the `2026-06-01`… |

## Q5 — Between the Phase 3 cutover and Phase 4, what can still be done on the legacy billing service?

Gold: **HC-4001** / Cutover timeline · must contain ['read-only', 'no new charges, refunds, or payment-method updates'] · **HIT at rank 1**

Diagnosis: answer-bearing chunk retrieved at rank 1

| Rank | Score | chunk_id | product_area | Article | Section | Preview |
|---|---|---|---|---|---|---|
| 1 | 0.1928 | `HC-4001#baseline-0003` | billing-migration | HC-4001 | Cutover timeline | \| Phase \| Window \| What happens \| Customer impact \| \|---\|---\|---\|---\| \| Phase 1 — Shadow \| 2026-06-01 to 2026-07-14 \| `billing-v2` comput… |
| 2 | 0.1868 | `HC-4001#baseline-0004` | billing-migration | HC-4001 | Cutover timeline | Between Phase 3 and Phase 4 the legacy service remains **read-only**: historical invoices and statements can still be downloaded, but no new charges, … |
| 3 | 0.1523 | `HC-4001#baseline-0002` | billing-migration | HC-4001 | Cutover timeline | The migration runs in four phases. All timestamps are UTC. Support agents should quote these dates verbatim; do not estimate a customer's phase from t… |
| 4 | 0.0973 | `HC-1012#baseline-0000` | payments | HC-1012 | Updating a Card on the Legacy Billing Service | # Updating a Card on the Legacy Billing Service ## Updating a stored card On `billing-v1` a customer updates a stored card from Account → Billing → Ca… |
| 5 | 0.0958 | `HC-4001#baseline-0000` | billing-migration | HC-4001 | Billing Migration Overview and Cutover Timeline | # Billing Migration Overview and Cutover Timeline ## What is changing We are moving every account off the legacy billing service (internally `billing-… |

## Q6 — How is proration calculated when a subscription is mapped to a new tier mid-cycle?

Gold: **HC-4004** / How proration is calculated · must contain ['30-day month'] · **HIT at rank 1**

Diagnosis: answer-bearing chunk retrieved at rank 1

| Rank | Score | chunk_id | product_area | Article | Section | Preview |
|---|---|---|---|---|---|---|
| 1 | 0.2923 | `HC-4004#baseline-0001` | subscriptions | HC-4004 | How legacy plans map to new tiers | Where a legacy plan has no equivalent (typically retired add-ons), the migration creates a "legacy compatibility" tier that preserves the price and en… |
| 2 | 0.2096 | `HC-4004#baseline-0002` | subscriptions | HC-4004 | How proration is calculated | When a subscription is mapped to a new tier part-way through a billing cycle, the unused portion of the legacy plan is credited at its daily rate and … |
| 3 | 0.1970 | `HC-4004#baseline-0000` | subscriptions | HC-4004 | Subscription Plan Mapping and Proration | # Subscription Plan Mapping and Proration ## How legacy plans map to new tiers Every `billing-v1` plan is mapped to exactly one `billing-v2` tier. Map… |
| 4 | 0.0895 | `HC-4004#baseline-0004` | subscriptions | HC-4004 | Troubleshooting: plan mapping errors | \| ERR-4152 \| The subscription is in a trial that ends after the cutover date \| Let the trial finish; trialling subscriptions are migrated on the tr… |
| 5 | 0.0871 | `HC-4004#baseline-0006` | subscriptions | HC-4004 | Troubleshooting: plan mapping errors | \| ERR-4157 \| A compatibility tier was created for an account whose contract has already expired \| Move the account to a current tier; expired contr… |

## Q7 — How long are failed webhook deliveries retried, and how many attempts are made?

Gold: **HC-4006** / Webhook delivery and the retry window · must contain ['24 hours', '6 attempts'] · **HIT at rank 1**

Diagnosis: answer-bearing chunk retrieved at rank 1

| Rank | Score | chunk_id | product_area | Article | Section | Preview |
|---|---|---|---|---|---|---|
| 1 | 0.1045 | `HC-4006#baseline-0001` | developer-api | HC-4006 | Required API version | The breaking changes in `2026-06-01` are: invoice identifiers are strings rather than integers, amounts are minor units rather than decimal strings, a… |
| 2 | 0.0555 | `HC-4002#baseline-0006` | payments | HC-4002 | Dunning and the retry window | When a charge fails after migration, `billing-v2` retries it on a fixed schedule. Failed payments are retried three times over a 10-day retry window: … |
| 3 | 0.0448 | `HC-4002#baseline-0005` | payments | HC-4002 | Troubleshooting: payment method errors | \| ERR-4038 \| The zero-amount verification timed out because the issuer did not respond within 30 seconds \| Retry once; if it fails twice, treat it … |
| 4 | 0.0387 | `HC-4002#baseline-0007` | payments | HC-4002 | Dunning and the retry window | Dunning emails are sent after each failed attempt to the billing contact, not to the workspace owner, unless the two are the same person. ## Re-author… |
| 5 | 0.0320 | `HC-1012#baseline-0000` | payments | HC-1012 | Updating a Card on the Legacy Billing Service | # Updating a Card on the Legacy Billing Service ## Updating a stored card On `billing-v1` a customer updates a stored card from Account → Billing → Ca… |

## Q8 — What happens to a customer's legacy credit balance after migration?

Gold: **HC-4003** / Why the first post-migration invoice looks different · must contain ['migration credit', 'written off'] · **HIT at rank 1**

Diagnosis: answer-bearing chunk retrieved at rank 1

| Rank | Score | chunk_id | product_area | Article | Section | Preview |
|---|---|---|---|---|---|---|
| 1 | 0.1238 | `HC-4003#baseline-0001` | invoicing | HC-4003 | Why the first post-migration invoice looks different | Legacy credit balances are not paid out. A remaining balance is carried over as a single line item named "migration credit" and applied to the first i… |
| 2 | 0.1099 | `HC-4003#baseline-0000` | invoicing | HC-4003 | Invoice and Statement Discrepancies After Migration | # Invoice and Statement Discrepancies After Migration ## Why the first post-migration invoice looks different The first invoice issued by `billing-v2`… |
| 3 | 0.0994 | `HC-4001#baseline-0002` | billing-migration | HC-4001 | Cutover timeline | The migration runs in four phases. All timestamps are UTC. Support agents should quote these dates verbatim; do not estimate a customer's phase from t… |
| 4 | 0.0864 | `HC-4001#baseline-0003` | billing-migration | HC-4001 | Cutover timeline | \| Phase \| Window \| What happens \| Customer impact \| \|---\|---\|---\|---\| \| Phase 1 — Shadow \| 2026-06-01 to 2026-07-14 \| `billing-v2` comput… |
| 5 | 0.0705 | `HC-4003#baseline-0003` | invoicing | HC-4003 | Troubleshooting: invoice errors | \| ERR-4087 \| A line item references a plan that was not mapped during migration \| Map the plan in the migration console (see HC-4004), then re-gene… |
