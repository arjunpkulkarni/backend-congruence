# Deployment & HIPAA Compliance Guide

## 🚨 Current Status: NOT HIPAA COMPLIANT

Your system currently stores Protected Health Information (PHI) insecurely:
- ❌ Unencrypted files on filesystem
- ❌ No access controls or authentication
- ❌ Patient IDs in plaintext
- ❌ No audit logging
- ❌ Third-party APIs without BAAs
- ❌ Video/audio stored unencrypted

**DO NOT use this system with real patient data until these issues are resolved.**

---

## 🎯 Roadmap to HIPAA Compliance

### Phase 1: Critical Security (2-4 weeks)
1. Encryption at rest and in transit
2. Authentication & authorization
3. Secure third-party API usage
4. Audit logging

### Phase 2: Infrastructure (2-3 weeks)
5. Production deployment
6. Database migration
7. Backup and disaster recovery

### Phase 3: Compliance Documentation (2-3 weeks)
8. Policies and procedures
9. Risk assessment
10. Staff training

**Total estimated time: 6-10 weeks**

---

## 📋 HIPAA Requirements Checklist

### Administrative Safeguards
- [ ] Security Management Process
  - [ ] Risk analysis completed
  - [ ] Risk management plan
  - [ ] Sanction policy for violations
  - [ ] Information system activity review
- [ ] Workforce Security
  - [ ] Authorization and supervision procedures
  - [ ] Workforce clearance procedures
  - [ ] Termination procedures
- [ ] Information Access Management
  - [ ] Isolating health care clearinghouse functions
  - [ ] Access authorization policies
  - [ ] Access establishment and modification
- [ ] Security Awareness and Training
  - [ ] Security reminders
  - [ ] Protection from malicious software
  - [ ] Log-in monitoring
  - [ ] Password management
- [ ] Security Incident Procedures
  - [ ] Response and reporting
- [ ] Contingency Plan
  - [ ] Data backup plan
  - [ ] Disaster recovery plan
  - [ ] Emergency mode operation plan
- [ ] Business Associate Agreements
  - [ ] Contracts with all vendors handling PHI

### Physical Safeguards
- [ ] Facility Access Controls
  - [ ] Contingency operations
  - [ ] Facility security plan
  - [ ] Access control and validation procedures
- [ ] Workstation Use and Security
  - [ ] Workstation use policies
  - [ ] Workstation security policies
- [ ] Device and Media Controls
  - [ ] Disposal procedures
  - [ ] Media re-use procedures
  - [ ] Accountability tracking
  - [ ] Data backup and storage

### Technical Safeguards
- [ ] Access Control
  - [ ] Unique user identification (Required)
  - [ ] Emergency access procedure (Required)
  - [ ] Automatic logoff
  - [ ] Encryption and decryption
- [ ] Audit Controls (Required)
  - [ ] Hardware, software, and/or procedural mechanisms to record and examine activity
- [ ] Integrity Controls
  - [ ] Mechanisms to authenticate electronic PHI
- [ ] Person or Entity Authentication (Required)
  - [ ] Procedures to verify person/entity seeking access
- [ ] Transmission Security
  - [ ] Integrity controls
  - [ ] Encryption

---

## 🔐 Phase 1: Critical Security Implementation

### 1.1 Encryption at Rest

**File System Encryption:**

```python
# app/utils/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64
import os

class PHIEncryption:
    """Encrypt/decrypt PHI using AES-256 via Fernet."""
    
    def __init__(self, master_key: bytes):
        """
        master_key should be stored in a secure key management system
        (AWS KMS, Azure Key Vault, HashiCorp Vault, etc.)
        """
        self.cipher = Fernet(master_key)
    
    def encrypt_file(self, input_path: str, output_path: str) -> None:
        """Encrypt a file (video, audio, JSON, etc.)"""
        with open(input_path, 'rb') as f:
            data = f.read()
        
        encrypted = self.cipher.encrypt(data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted)
    
    def decrypt_file(self, input_path: str, output_path: str) -> None:
        """Decrypt a file for processing"""
        with open(input_path, 'rb') as f:
            encrypted = f.read()
        
        decrypted = self.cipher.decrypt(encrypted)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted)
    
    def encrypt_json(self, data: dict) -> bytes:
        """Encrypt JSON data (session summaries, etc.)"""
        import json
        json_str = json.dumps(data).encode('utf-8')
        return self.cipher.encrypt(json_str)
    
    def decrypt_json(self, encrypted: bytes) -> dict:
        """Decrypt JSON data"""
        import json
        decrypted = self.cipher.decrypt(encrypted)
        return json.loads(decrypted.decode('utf-8'))

# Usage:
# cipher = PHIEncryption(master_key=os.environ['MASTER_KEY'])
# cipher.encrypt_file('video.mp4', 'video.mp4.encrypted')
```

**Key Management:**
- **Option A (Best):** AWS KMS, Azure Key Vault, Google Cloud KMS
- **Option B (Good):** HashiCorp Vault (self-hosted)
- **Option C (Minimum):** Environment variables with restricted access

### 1.2 De-identification / Pseudonymization

**Replace patient IDs with non-reversible hashes:**

```python
# app/utils/deidentification.py
import hashlib
import hmac

class PatientDeidentifier:
    """Convert patient IDs to pseudonymous identifiers."""
    
    def __init__(self, pepper: bytes):
        """
        pepper: secret value stored separately from database
        Never store pepper with data - use key management system
        """
        self.pepper = pepper
    
    def pseudonymize(self, patient_id: str) -> str:
        """
        Convert patient_id to non-reversible pseudonym.
        Same patient_id always produces same pseudonym (for cross-session analysis).
        """
        h = hmac.new(self.pepper, patient_id.encode('utf-8'), hashlib.sha256)
        return h.hexdigest()[:32]  # 128-bit identifier
    
    def is_valid_pseudonym(self, pseudonym: str) -> bool:
        """Check if string looks like a valid pseudonym"""
        return len(pseudonym) == 32 and all(c in '0123456789abcdef' for c in pseudonym)

# Usage in main.py:
# deidentifier = PatientDeidentifier(pepper=os.environ['PEPPER_SECRET'])
# pseudonymous_id = deidentifier.pseudonymize(payload.patient_id)
# # Use pseudonymous_id for all file storage, never store real patient_id
```

### 1.3 Authentication & Authorization

**Add JWT-based authentication:**

```python
# requirements.txt additions:
# python-jose[cryptography]==3.3.0
# passlib[bcrypt]==1.7.4
# python-multipart==0.0.6

# app/auth.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.environ.get("JWT_SECRET_KEY")  # Store in key management
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class User:
    def __init__(self, username: str, role: str, organization: str):
        self.username = username
        self.role = role  # "therapist", "admin", "analyst"
        self.organization = organization

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Extract and validate JWT token from request."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        organization: str = payload.get("org")
        
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        
        return User(username=username, role=role, organization=organization)
    
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

# Update main.py endpoints:
@app.post("/process_session", response_model=ProcessSessionResponse)
async def process_session(
    payload: ProcessSessionRequest,
    current_user: User = Depends(get_current_user)  # Add this
) -> ProcessSessionResponse:
    # Log access for audit trail
    audit_log(
        user=current_user.username,
        action="process_session",
        patient_id=payload.patient_id,
        timestamp=datetime.utcnow()
    )
    # ... rest of existing code
```

### 1.4 Audit Logging

**Comprehensive audit trail:**

```python
# app/utils/audit.py
import logging
from datetime import datetime
from typing import Optional
import json

class AuditLogger:
    """HIPAA-compliant audit logging."""
    
    def __init__(self, log_file: str = "/var/log/congruence/audit.log"):
        self.logger = logging.getLogger("audit")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_access(
        self,
        user: str,
        action: str,
        patient_id: str,
        resource: Optional[str] = None,
        success: bool = True,
        ip_address: Optional[str] = None
    ):
        """Log PHI access event."""
        event = {
            "event_type": "phi_access",
            "user": user,
            "action": action,
            "patient_id": patient_id,
            "resource": resource,
            "success": success,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(event))
    
    def log_security_event(
        self,
        event_type: str,
        user: Optional[str],
        details: dict
    ):
        """Log security-related events (failed logins, etc.)"""
        event = {
            "event_type": event_type,
            "user": user,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.warning(json.dumps(event))

# Usage:
audit = AuditLogger()
audit.log_access(
    user="therapist@example.com",
    action="view_session",
    patient_id="pseudonym_abc123",
    ip_address=request.client.host
)
```

### 1.5 Secure Third-Party APIs

**Current issues:**
- OpenAI API - need BAA (Business Associate Agreement)
- DeepFace - runs locally (OK)
- Vesper - runs locally (OK)
- faster-whisper - runs locally (OK)

**Solutions:**

**Option A: Get BAAs from vendors**
- OpenAI offers BAA for Enterprise customers
- Azure OpenAI Service has HIPAA-compliant offering
- AWS Bedrock (Claude/Titan) has BAA available

**Option B: Self-host everything**
```python
# Use local LLMs instead of OpenAI
# app/services/llm.py

# Replace OpenAI calls with:
from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalLLM:
    """HIPAA-compliant local LLM (no data leaves your infrastructure)"""
    
    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def analyze_text_emotion(self, text: str) -> dict:
        # Run inference locally - no API calls
        prompt = f"Analyze the emotional valence of this text: {text}"
        # ... inference logic
```

**Option C: Anonymize before API calls**
```python
def anonymize_text(text: str) -> str:
    """Remove all identifying information before sending to API"""
    # Replace names, dates, locations, etc.
    # Use NER (Named Entity Recognition) to detect and redact
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    anonymized = text
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "DATE", "GPE", "LOC"]:
            anonymized = anonymized.replace(ent.text, f"[{ent.label_}]")
    
    return anonymized
```

---

## 🏗️ Phase 2: Infrastructure

### 2.1 Production Architecture

**Recommended Stack:**

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer (HTTPS only)              │
│                     - TLS 1.3                                │
│                     - Certificate from trusted CA            │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
┌────────▼────────┐            ┌────────▼────────┐
│   API Server 1  │            │   API Server 2  │
│   (FastAPI)     │            │   (FastAPI)     │
│   - Gunicorn    │            │   - Gunicorn    │
│   - Workers: 4  │            │   - Workers: 4  │
└────────┬────────┘            └────────┬────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
         ┌───────────────┴────────────────────────┐
         │                                        │
┌────────▼──────────┐                  ┌─────────▼──────────┐
│   PostgreSQL      │                  │   S3-compatible     │
│   (Encrypted)     │                  │   Storage           │
│   - Patient data  │                  │   (Encrypted)       │
│   - Session meta  │                  │   - Videos          │
│   - Audit logs    │                  │   - Audio           │
└───────────────────┘                  │   - Frames          │
                                       └────────────────────┘
         │
┌────────▼──────────┐
│   Redis/Memcached │
│   (Session cache) │
└───────────────────┘
```

### 2.2 Environment Setup

**Docker Compose for local development:**

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/congruence
      - REDIS_URL=redis://redis:6379
      - S3_BUCKET=congruence-media
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - MASTER_ENCRYPTION_KEY=${MASTER_ENCRYPTION_KEY}
    volumes:
      - ./app:/app
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=changeme
      - POSTGRES_DB=congruence
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    # Enable encryption at rest
    command: postgres -c ssl=on -c ssl_cert_file=/etc/ssl/certs/server.crt

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass changeme

volumes:
  postgres_data:
```

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY vesper/ ./vesper/

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "app.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### 2.3 Database Migration

**Replace filesystem with PostgreSQL:**

```sql
-- init.sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Patients table (minimal identifying info)
CREATE TABLE patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pseudonymous_id VARCHAR(64) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Sessions table
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID REFERENCES patients(id),
    session_timestamp BIGINT NOT NULL,
    duration FLOAT,
    video_url TEXT,  -- Encrypted S3 URL
    status VARCHAR(50) DEFAULT 'processing',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Analysis results (encrypted JSON blobs)
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id),
    result_type VARCHAR(50),  -- 'intensity_timeline', 'incongruence_markers', etc.
    encrypted_data BYTEA NOT NULL,  -- Encrypted JSON
    created_at TIMESTAMP DEFAULT NOW()
);

-- Audit log table
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    user_id VARCHAR(255),
    action VARCHAR(100),
    patient_id UUID,
    session_id UUID,
    ip_address INET,
    details JSONB,
    success BOOLEAN DEFAULT TRUE
);

-- Create indexes for performance
CREATE INDEX idx_sessions_patient ON sessions(patient_id);
CREATE INDEX idx_sessions_timestamp ON sessions(session_timestamp);
CREATE INDEX idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_user ON audit_log(user_id);
CREATE INDEX idx_audit_patient ON audit_log(patient_id);

-- Row-level security (optional but recommended)
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY session_access ON sessions
    USING (patient_id IN (SELECT patient_id FROM user_access WHERE user_id = current_user));
```

**Database access layer:**

```python
# app/db.py
from sqlalchemy import create_engine, Column, String, Float, BigInteger, LargeBinary, TIMESTAMP, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Patient(Base):
    __tablename__ = "patients"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pseudonymous_id = Column(String(64), unique=True, nullable=False)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

class Session(Base):
    __tablename__ = "sessions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True))
    session_timestamp = Column(BigInteger, nullable=False)
    duration = Column(Float)
    video_url = Column(String)
    status = Column(String(50))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True))
    result_type = Column(String(50))
    encrypted_data = Column(LargeBinary)
    created_at = Column(TIMESTAMP)

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(BigInteger, primary_key=True)
    timestamp = Column(TIMESTAMP)
    user_id = Column(String(255))
    action = Column(String(100))
    patient_id = Column(UUID(as_uuid=True))
    session_id = Column(UUID(as_uuid=True))
    ip_address = Column(String(45))
    details = Column(JSONB)
    success = Column(Boolean)
```

### 2.4 Cloud Deployment Options

**Option A: AWS (Most Common)**
- **Compute:** ECS Fargate or EKS
- **Database:** RDS PostgreSQL with encryption
- **Storage:** S3 with server-side encryption (SSE-KMS)
- **Key Management:** AWS KMS
- **Load Balancer:** ALB with AWS Certificate Manager
- **Monitoring:** CloudWatch + CloudTrail
- **Compliance:** AWS has HIPAA eligibility, sign BAA

**Option B: Azure**
- **Compute:** Azure Container Instances or AKS
- **Database:** Azure Database for PostgreSQL
- **Storage:** Azure Blob Storage with encryption
- **Key Management:** Azure Key Vault
- **Load Balancer:** Azure Load Balancer
- **Monitoring:** Azure Monitor
- **Compliance:** Azure has HIPAA compliance, sign BAA

**Option C: Self-Hosted (Maximum Control)**
- **Pros:** Complete control, no cloud BAA needed
- **Cons:** You handle ALL security, backups, disaster recovery
- **Requirements:** 
  - Secure data center with physical access controls
  - Dedicated IT security team
  - Disaster recovery site
  - Network security infrastructure

**Recommendation: Start with AWS or Azure**
- Easier to achieve HIPAA compliance
- Built-in encryption, backup, monitoring
- Shared responsibility model
- BAAs readily available

---

## 📄 Phase 3: Compliance Documentation

### 3.1 Required Policies

Create these policy documents:

1. **Security Management Process**
   - Risk assessment procedures
   - Risk mitigation strategies
   - Sanction policy for violations

2. **Access Control Policy**
   - User access procedures
   - Role-based access control (RBAC)
   - Minimum necessary access principle

3. **Incident Response Plan**
   - Breach detection procedures
   - Breach notification procedures (60 days)
   - Remediation steps

4. **Data Backup and Disaster Recovery**
   - Backup frequency (daily recommended)
   - Recovery time objective (RTO)
   - Recovery point objective (RPO)

5. **Employee Training Program**
   - Initial HIPAA training for all staff
   - Annual refresher training
   - Training documentation

### 3.2 Business Associate Agreements (BAAs)

Required BAAs with:
- [ ] Cloud provider (AWS/Azure/GCP)
- [ ] LLM provider (OpenAI/Anthropic if used)
- [ ] Any subcontractors
- [ ] Backup service providers

### 3.3 Risk Assessment

Annual risk assessment covering:
- [ ] Potential threats and vulnerabilities
- [ ] Current security measures
- [ ] Gaps in security
- [ ] Action plan to address gaps

---

## 🚀 Deployment Checklist

### Pre-Launch

- [ ] All PHI encrypted at rest and in transit
- [ ] Authentication and authorization implemented
- [ ] Audit logging active and tested
- [ ] Database configured with encryption
- [ ] S3/blob storage configured with encryption
- [ ] Key management system configured
- [ ] BAAs signed with all vendors
- [ ] Policies and procedures documented
- [ ] Staff trained on HIPAA requirements
- [ ] Penetration testing completed
- [ ] Disaster recovery plan tested

### Launch Day

- [ ] SSL/TLS certificates installed (A+ rating on SSL Labs)
- [ ] Firewall rules configured (deny all, allow specific)
- [ ] Monitoring and alerting configured
- [ ] Backup systems verified
- [ ] Incident response team on standby
- [ ] Legal compliance officer approval

### Post-Launch

- [ ] Daily backup verification
- [ ] Weekly security log review
- [ ] Monthly access review (remove unused accounts)
- [ ] Quarterly vulnerability scans
- [ ] Annual risk assessment
- [ ] Annual HIPAA training for staff

---

## 💰 Estimated Costs

### Development (One-Time)
- Security implementation: $20,000 - $40,000
- HIPAA compliance consulting: $10,000 - $25,000
- Penetration testing: $5,000 - $15,000
- Legal review: $5,000 - $10,000
- **Total: $40,000 - $90,000**

### Infrastructure (Monthly)
- AWS/Azure hosting: $500 - $2,000
- Database: $200 - $800
- Storage (S3/Blob): $100 - $500
- Key management: $50 - $200
- Monitoring/logging: $100 - $300
- **Total: $950 - $3,800/month**

### Ongoing (Annual)
- Compliance audits: $10,000 - $30,000
- Security updates: $10,000 - $25,000
- Staff training: $2,000 - $5,000
- Insurance (cyber liability): $5,000 - $15,000
- **Total: $27,000 - $75,000/year**

---

## 🆘 Quick Start (Minimum Viable Compliance)

If you need to launch quickly:

### Week 1-2: Immediate Security
```bash
# 1. Add encryption
pip install cryptography

# 2. Add authentication  
pip install python-jose[cryptography] passlib[bcrypt]

# 3. Set up environment variables (don't commit!)
export JWT_SECRET_KEY=$(openssl rand -hex 32)
export MASTER_ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
export PEPPER_SECRET=$(openssl rand -hex 32)

# 4. Pseudonymize patient IDs immediately
# Never store real patient IDs
```

### Week 3-4: Deploy Securely
```bash
# 1. Set up managed database
# AWS RDS PostgreSQL or Azure Database for PostgreSQL

# 2. Enable encryption at rest
# Both AWS and Azure have checkbox options

# 3. Deploy with HTTPS only
# Use AWS Certificate Manager or Let's Encrypt

# 4. Enable audit logging
# CloudWatch (AWS) or Azure Monitor
```

### Week 5-6: Documentation
- Sign BAAs with cloud provider
- Write incident response plan
- Document access control procedures
- Train staff on HIPAA basics

---

## ⚠️ Common Mistakes to Avoid

1. **"We'll add security later"** - NO. Security must be built in from day 1
2. **Storing unencrypted videos** - Videos contain faces (biometric data under HIPAA)
3. **No audit logging** - Required by HIPAA, essential for breach investigation
4. **Weak access controls** - "Everyone can access everything" is not acceptable
5. **Forgetting about backups** - Encryption without backups = data loss risk
6. **DIY encryption** - Use proven libraries (cryptography, Fernet), not homebrew
7. **Skipping BAAs** - Using any third-party service without BAA is a violation

---

## 📚 Resources

### HIPAA Official Resources
- HHS HIPAA Security Rule: https://www.hhs.gov/hipaa/for-professionals/security/index.html
- HIPAA Breach Notification Rule: https://www.hhs.gov/hipaa/for-professionals/breach-notification/index.html

### Cloud Provider HIPAA Guides
- AWS HIPAA Compliance: https://aws.amazon.com/compliance/hipaa-compliance/
- Azure HIPAA/HITECH: https://docs.microsoft.com/en-us/azure/compliance/offerings/offering-hipaa-us
- GCP HIPAA Compliance: https://cloud.google.com/security/compliance/hipaa

### Security Tools
- SSL Labs SSL Test: https://www.ssllabs.com/ssltest/
- OWASP Top 10: https://owasp.org/www-project-top-ten/
- CIS Benchmarks: https://www.cisecurity.org/cis-benchmarks/

### Legal
- Consider hiring a healthcare IT attorney
- HIPAA compliance consultants (HITRUST, Coalfire, etc.)

---

## 🎯 Next Steps

1. **Hire a HIPAA compliance consultant** (strongly recommended)
2. **Implement Phase 1 (Critical Security)** before processing any real PHI
3. **Sign BAA with cloud provider** before deployment
4. **Get legal review** of your compliance approach
5. **Start with pilot group** (friendly therapists, test patients)
6. **Iterate based on feedback** and audit findings

---

## ⚖️ Legal Disclaimer

This guide is for informational purposes only and does not constitute legal advice. 
HIPAA compliance is complex and requirements vary by organization size, business model, 
and jurisdiction. Consult with a qualified healthcare IT attorney and HIPAA compliance 
expert before handling protected health information.

**The authors of this software are not liable for any HIPAA violations or data breaches 
resulting from use of this system.**

