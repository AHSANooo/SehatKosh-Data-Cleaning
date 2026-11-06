# Contributing to SehatKosh De-Identification Pipeline

Thank you for your interest in contributing to the **SehatKosh Medical Document De-Identification Pipeline**! We welcome community contributions to help improve clinical de-identification, regex patterns for regional medical forms, and audit tooling.

---

## ⚠️ Critical Rule: No Real Patient Data in Issues or PRs

**Never upload, commit, or link to documents containing real patient names, CNICs, phone numbers, addresses, or clinical records in issues, discussions, or pull requests.**
- Use **synthetic or mocked data** when reporting issues or creating test cases.
- Any pull request or issue containing real Protected Health Information (PHI) will be deleted immediately.

---

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/<your-username>/SehatKosh-Data-Cleaning.git
   cd SehatKosh-Data-Cleaning
   ```
3. **Set up the virtual environment**:
   ```bash
   python3 -m venv redaction_env
   source redaction_env/bin/activate  # Or on Windows: .\redaction_env\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

---

## Areas Where Contributions Are Needed

- **Regional Regex Patterns**: Expanding support for medical records across South Asia and developing healthcare systems (e.g., custom MRNO formats, hospital admission codes, national health card IDs).
- **Anchor Keywords**: Additional multilingual or regional anchor phrases (`urdu`, `roman urdu`, etc.) in clinical notes.
- **Audit Tool Features**: Enhancing the Streamlit audit application (e.g., multi-annotator support, bounding box drawing/correction tool).
- **Inference Speed**: Optimizing OCR batch inference with ONNX Runtime or TensorRT.

---

## Development & Code Style

- Write clean, documented Python adhering to PEP 8 standards.
- Maintain backwards compatibility with Python 3.10+.
- Keep inference strictly offline and air-gapped — do not introduce external network dependencies during document processing.
- Verify syntax and linting before submitting:
  ```bash
  python3 -m py_compile batch_redact.py streamlit_audit.py
  ```

---

## Pull Request Guidelines

1. Create a dedicated branch for your feature or bug fix:
   ```bash
   git checkout -b feat/add-custom-mrno-pattern
   ```
2. Commit your changes with clear, semantic commit messages (e.g., `feat: ...`, `fix: ...`, `docs: ...`).
3. Ensure no local test logs or documents are committed (`git status`).
4. Push your branch and open a Pull Request against `main`.
5. Provide a clear description in your PR of what was changed and why.
