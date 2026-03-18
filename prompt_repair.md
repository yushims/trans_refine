## JSON REPAIR PROTOCOL

Return **ONLY** one strict JSON value (no markdown/prose) that is a corrected version of `previous_output`.

### 1. GUIDING PRINCIPLES

* **Semantic Integrity:** Do **NOT** change semantic content except as required to make it valid JSON and satisfy the schema below.
* **Format Strictness:** Output **MUST** be exactly one JSON value (object, array, string, number, boolean, or null).
* **Cleaning:** Remove any non-JSON text (including markdown or code fences) outside the JSON value.

---

### 2. REPAIR & NORMALIZATION RULES

* **Syntax Correction:** * Quote keys and strings properly using double quotes.
* Escape internal quotes and newlines.
* Remove trailing commas and fix mismatched brackets/braces.
* Ensure proper separators and valid UTF-8 encoding.


* **Structural Alignment:** * **Missing Fields:** If required fields are missing, add them using schema-consistent safe defaults:
* `object`: Add property with schema-defined default (else: `{}`, `0`, `false`, or `null`).
* `array`: `[]`
* `string`: `""`
* `number`/`integer`: `0`
* `boolean`: `false`
* `null`: `null`
* **Type Coercion:** If a field has the wrong type, coerce to the required type only if unambiguous; otherwise, replace with a safe default.
* **Schema Pruning:** Drop any keys/properties not allowed by the schema (e.g., when `additionalProperties=false`).
* **Constraint Satisfaction:** Modify values only as strictly needed to satisfy schema constraints (`min/max`, `enum/const`, `required`, `minItems`, `patterns`).



---

### 3. CONTEXTUAL DATA

**Target Schema (Authoritative):**

```json
{target_schema}

```

**Validation Error:**

> {validation_error}

**Previous Output (Corrupted):**

```text
{previous_output}

```

---
