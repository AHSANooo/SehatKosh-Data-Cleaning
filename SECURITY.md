# Security Policy

## Healthcare Data Privacy & Security

**SehatKosh De-Identification Pipeline** is designed specifically to protect patient privacy by removing Personally Identifiable Information (PII) and Protected Health Information (PHI) from clinical documents, prescriptions, and medical receipts.

### Fundamental Guarantees:
1. **Fully Air-Gapped & Offline**: This software does not make external network requests during OCR inference or NER processing. All models run 100% locally on your machine.
2. **Local-Only Processing**: No patient data or extracted metadata is sent to third-party APIs or cloud endpoints.
3. **Data Immutability**: Input files in `Data/` are read-only and never modified or overwritten in place.

---

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

---

## Reporting a Vulnerability or PII Leakage Edge Case

If you discover a security vulnerability, an insecure dependency, or an algorithmic bypass that regularly fails to redact critical PII patterns (such as unhandled national ID formats or phone number variations):

1. **Do NOT open a public GitHub issue containing real patient records or unredacted sensitive documents.**
2. Send an email to **ahsankhetran72@gmail.com** with:
   - A synthetic (fake/mocked) sample document reproducing the issue.
   - The version of the pipeline and models used.
   - Any proposed regular expressions or anchor label enhancements.
3. We will acknowledge receipt within 48 hours and work with you to test and release a patch.
