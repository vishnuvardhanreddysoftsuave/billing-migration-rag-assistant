---
article_id: HC-4002
title: Payment Method Errors During the Billing Migration
product_area: payments
last_updated: 2026-07-29
owner: Payments Support
audience: tier-1-support
---

# Payment Method Errors During the Billing Migration

## How payment methods are migrated

Stored cards live in a card vault, not in the billing service itself. The migration
re-points each account at the new vault (`vault-v2`) and copies the token that represents
the card. Tokens issued on or after 2026-05-01 carry the network transaction identifier
required by `vault-v2` and are copied automatically. Tokens issued **before 2026-05-01**
predate that identifier and cannot be re-tokenised without the card number, which we do
not store — those customers must re-enter the card.

Bank debit mandates (SEPA, BACS, ACH) are migrated in full regardless of age, because the
mandate reference, not a token, is the stored credential. Customers do not need to
re-authorise a debit mandate.

## Troubleshooting: payment method errors

| Error code | Cause | Fix |
|---|---|---|
| ERR-4030 | The card vault rejected the copy because the account had no default payment method | Ask the customer to add a payment method, then re-queue the migration |
| ERR-4031 | The card's expiry date has already passed, so the token cannot be re-tokenised | Ask the customer to add a current card; expired cards are never migrated |
| ERR-4032 | The stored card token was issued before 2026-05-01 and has no network transaction identifier, so it was not migrated to the new vault | Ask the customer to re-authorise the card under Billing → Payment methods; pre-2026-05-01 tokens cannot be migrated automatically and must be re-entered by the cardholder |
| ERR-4033 | The card issuer declined the zero-amount verification used to confirm the copied token | Ask the customer to contact their issuer, then retry the verification from the billing portal |
| ERR-4034 | Two accounts reference the same vault token after a workspace merge | Detach the token from the non-billing account in the vault console; a token may belong to one account only |
| ERR-4035 | The payment method is a debit mandate whose reference failed checksum validation | Re-collect the mandate; a failed checksum means the stored reference is corrupt |
| ERR-4036 | The card is a virtual or single-use card that the issuer will not re-authorise | Ask the customer for a durable card; single-use cards cannot back a subscription |
| ERR-4037 | The customer's billing country does not match the card's issuing country under the new vault rules | Update the billing address to match the card, or use a card issued in the billing country |
| ERR-4038 | The zero-amount verification timed out because the issuer did not respond within 30 seconds | Retry once; if it fails twice, treat it as ERR-4033 and refer the customer to their issuer |

## Dunning and the retry window

When a charge fails after migration, `billing-v2` retries it on a fixed schedule. Failed
payments are retried three times over a 10-day retry window: on day 1, day 5, and day 10
after the original failure. The subscription is suspended only after the day-10 attempt
fails. Agents cannot shorten the retry window, but they can trigger one immediate
out-of-band retry from the billing portal, which does not consume one of the three
scheduled attempts.

Dunning emails are sent after each failed attempt to the billing contact, not to the
workspace owner, unless the two are the same person.

## Re-authorisation walkthrough

Direct customers hitting ERR-4032 to Billing → Payment methods → "Re-authorise card".
They will need the physical card: the flow asks for the full card number, expiry, and
CVC, because the pre-2026-05-01 token cannot supply them. Once re-authorised, the new
token is created directly in `vault-v2` and no further action is needed. Any invoice that
failed while the card was unusable is retried automatically at the next dunning attempt.
