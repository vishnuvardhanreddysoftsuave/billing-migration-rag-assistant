---
article_id: HC-1007
title: Reading a Legacy Invoice
product_area: invoicing
last_updated: 2025-11-04
owner: Invoicing Support
audience: tier-1-support
---

# Reading a Legacy Invoice

## Invoice layout

A legacy (`billing-v1`) invoice has a header block with the invoice number, the issue
date, and the billing period, followed by line items grouped by subscription. Legacy
invoice numbers are integers with no prefix and are allocated from a single global
sequence, which is why two accounts never share a number but an account's numbers are
not contiguous.

## Common questions

Customers most often ask why the billing period does not match a calendar month. Legacy
billing anchors the period to the subscription start date, so an account created on the
9th is invoiced from the 9th to the 8th. This is unchanged behaviour and is not a defect.

Credit balances appear as a negative "account credit" line at the bottom of the invoice
and are consumed automatically by the next invoice.
