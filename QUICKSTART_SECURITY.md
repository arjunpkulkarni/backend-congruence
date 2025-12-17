# Quick Start: Adding Security (1-2 Hours)

This guide helps you add basic encryption and de-identification to your system TODAY.
For full HIPAA compliance, see `DEPLOYMENT_HIPAA_GUIDE.md`.

---

## ⚡ 5-Minute Security Boost

### Step 1: Install Security Dependencies

```bash
cd /Users/arjunkulkarni/Desktop/CONGRUENCE
pip install cryptography passlib[bcrypt] python-jose python-dotenv spacy
python -m spacy download en_core_web_sm
```

### Step 2: Generate Security Keys

```bash
# Generate encryption key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Generate pepper for pseudonymization
openssl rand -hex 32

# Generate JWT secret
openssl rand -hex 32
```

### Step 3: Create .env File

```bash
# Copy example to .env
cp env.example.txt .env

# Edit .env and paste your generated keys
nano .env
# (or use your preferred editor)
```

Paste the keys you generated in Step 2:
```bash
MASTER_ENCRYPTION_KEY=your-generated-key-here
PEPPER_SECRET=your-generated-pepper-here
JWT_SECRET_KEY=your-generated-jwt-secret-here
ENVIRONMENT=development
```

**IMPORTANT:** Never commit `.env` to git!

```bash
# Ensure .env is in .gitignore
echo ".env" >> .gitignore
```

---

## 🔐 Use Encryption in Your Code

### Update main.py

Add these imports at the top:

```python
from app.utils.encryption import setup_encryption
from app.utils.deidentification import setup_deidentifier
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize security utilities (once at startup)
cipher = setup_encryption()
deidentifier = setup_deidentifier()
```

### Pseudonymize Patient IDs

Replace this:
```python
@app.post("/process_session")
def process_session(payload: ProcessSessionRequest):
    patient_id = payload.patient_id  # ❌ Don't use real ID
```

With this:
```python
@app.post("/process_session")
def process_session(payload: ProcessSessionRequest):
    # Pseudonymize immediately - never store real ID
    pseudonymous_id = deidentifier.pseudonymize(payload.patient_id)
    
    # Use pseudonym everywhere
    session_dir, media_dir, frames_dir, outputs_dir = create_session_directories(
        workspace_root=workspace_root,
        patient_id=pseudonymous_id,  # ✅ Use pseudonym
        session_ts=session_ts,
    )
```

### Encrypt Files After Processing

Add this after video download:
```python
# Download video
download_video_file(video_url=payload.video_url, destination_path=video_path)

# ✅ Encrypt immediately after download
video_encrypted_path = video_path + ".encrypted"
cipher.encrypt_file(video_path, video_encrypted_path)
os.remove(video_path)  # Delete plaintext

# ✅ Decrypt only when needed for processing
video_temp_path = video_path + ".temp"
cipher.decrypt_file(video_encrypted_path, video_temp_path)

# Process the video
extract_audio_with_ffmpeg(input_video_path=video_temp_path, ...)

# ✅ Delete decrypted temp file immediately
os.remove(video_temp_path)
```

### Encrypt JSON Outputs

Replace this:
```python
with open(session_summary_path, "w") as f:
    json.dump(session_summary, f)  # ❌ Plaintext PHI
```

With this:
```python
# Encrypt JSON before saving
encrypted_summary = cipher.encrypt_json(session_summary)
with open(session_summary_path + ".encrypted", "wb") as f:
    f.write(encrypted_summary)  # ✅ Encrypted PHI
```

### Anonymize Transcripts Before Logging

Replace this:
```python
logger.info("Transcript: %s", transcript_text)  # ❌ May contain PII
```

With this:
```python
safe_transcript = deidentifier.anonymize_text(transcript_text)
logger.info("Transcript: %s", safe_transcript)  # ✅ PII removed
```

---

## ✅ Test Your Security

Create `test_security.py`:

```python
#!/usr/bin/env python3
"""Test encryption and de-identification"""

import os
from app.utils.encryption import PHIEncryption
from app.utils.deidentification import PatientDeidentifier

# Generate test keys
encryption_key = PHIEncryption.generate_key()
pepper = os.urandom(32)

# Test encryption
cipher = PHIEncryption(encryption_key)

# Test string encryption
original = "John Doe, DOB: 1990-01-01"
encrypted = cipher.encrypt_string(original)
decrypted = cipher.decrypt_string(encrypted)

print(f"Original: {original}")
print(f"Encrypted: {encrypted}")
print(f"Decrypted: {decrypted}")
assert decrypted == original, "Decryption failed!"
print("✅ Encryption test passed\n")

# Test JSON encryption
data = {"patient": "John Doe", "diagnosis": "Anxiety"}
encrypted_json = cipher.encrypt_json(data)
decrypted_json = cipher.decrypt_json(encrypted_json)

print(f"Original JSON: {data}")
print(f"Encrypted: {encrypted_json[:50]}...")
print(f"Decrypted JSON: {decrypted_json}")
assert decrypted_json == data, "JSON encryption failed!"
print("✅ JSON encryption test passed\n")

# Test pseudonymization
deidentifier = PatientDeidentifier(pepper)

patient_id = "patient@example.com"
pseudonym1 = deidentifier.pseudonymize(patient_id)
pseudonym2 = deidentifier.pseudonymize(patient_id)

print(f"Patient ID: {patient_id}")
print(f"Pseudonym 1: {pseudonym1}")
print(f"Pseudonym 2: {pseudonym2}")
assert pseudonym1 == pseudonym2, "Pseudonymization not deterministic!"
assert pseudonym1 != patient_id, "Pseudonymization failed!"
print("✅ Pseudonymization test passed\n")

# Test text anonymization
text = "My name is John Doe and I live in New York. Call me at 555-123-4567."
anonymized = deidentifier.anonymize_text(text)

print(f"Original: {text}")
print(f"Anonymized: {anonymized}")
assert "John Doe" not in anonymized, "Name not anonymized!"
assert "555-123-4567" not in anonymized, "Phone not anonymized!"
print("✅ Text anonymization test passed\n")

print("=" * 60)
print("🎉 ALL SECURITY TESTS PASSED!")
print("=" * 60)
```

Run the test:
```bash
python test_security.py
```

Expected output:
```
✅ Encryption test passed
✅ JSON encryption test passed
✅ Pseudonymization test passed
✅ Text anonymization test passed
🎉 ALL SECURITY TESTS PASSED!
```

---

## 🚨 What This DOES NOT Cover

This quick security boost does **NOT** make you HIPAA compliant! Still needed:

- [ ] Authentication & authorization
- [ ] Audit logging
- [ ] Secure database (not filesystem)
- [ ] HTTPS/TLS
- [ ] Business Associate Agreements with vendors
- [ ] Policies and procedures
- [ ] Staff training
- [ ] Risk assessment
- [ ] Incident response plan

See `DEPLOYMENT_HIPAA_GUIDE.md` for full compliance.

---

## 🎯 Next Steps

1. ✅ You've added encryption and de-identification (today)
2. ⏭️ Read `DEPLOYMENT_HIPAA_GUIDE.md` (30 minutes)
3. ⏭️ Add authentication (see guide, 1-2 days)
4. ⏭️ Add audit logging (see guide, 1 day)
5. ⏭️ Hire HIPAA compliance consultant (strongly recommended)
6. ⏭️ Deploy to secure infrastructure (1-2 weeks)

---

## ⚠️ Important Reminders

- **NEVER commit `.env` to git** - Contains encryption keys
- **NEVER log decrypted PHI** - Use anonymized versions only
- **NEVER store real patient IDs** - Use pseudonyms everywhere
- **DELETE temp decrypted files** - Encrypt immediately after use
- **ROTATE keys regularly** - At least annually
- **TEST disaster recovery** - Ensure you can restore from backups

---

## 🆘 Troubleshooting

### "Invalid encryption key" error
```bash
# Regenerate key with correct format
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Copy to .env as MASTER_ENCRYPTION_KEY
```

### "Pepper must be at least 32 bytes"
```bash
# Generate longer pepper
openssl rand -hex 32
# Copy to .env as PEPPER_SECRET
```

### spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

### Environment variables not loading
```bash
# Ensure .env file is in project root
ls -la .env

# Check it's being loaded in code
from dotenv import load_dotenv
load_dotenv()
```

---

## 📞 Need Help?

- Review `DEPLOYMENT_HIPAA_GUIDE.md` for comprehensive guidance
- Hire a HIPAA compliance consultant
- Don't hesitate to invest in security - a breach costs far more!

**Remember: This is healthcare data. When in doubt, err on the side of caution.**

