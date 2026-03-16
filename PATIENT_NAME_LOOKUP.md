# Patient Name Lookup Feature

## Overview

Users can now reference patients by **name** instead of UUID. The agent automatically resolves names to patient IDs.

## What Changed

### 1. New Data File: `data/patients.json`
Stores patient metadata:
```json
{
  "4e3c1260-9e27-4cc8-9720-114e068d03f1": {
    "name": "Rob Wazowski",
    "mrn": "MRN-001",
    "dob": "2010-01-01",
    "created_at": "2025-11-30T00:00:00Z"
  }
}
```

### 2. New Data Access Functions
- `find_patient_by_name(name)` - Find patient by name (partial match)
- `search_patients(query)` - Search by name, MRN, or patient_id
- `list_patients()` - Now includes patient names and MRNs

### 3. New Agent Tool: `find_patient`
Resolves patient names to IDs before calling other tools.

### 4. Updated System Prompt
Agent now knows to call `find_patient` first when users mention names.

## Usage Examples

### Before (Required UUID)
```
"Show me notes for patient 4e3c1260-9e27-4cc8-9720-114e068d03f1"
```

### After (Natural Language)
```
"Show me notes for Rob Wazowski"
"What was Rob's congruence score?"
"Give me transcript for the demo patient"
"Tell me about patient MRN-001"
```

## Test Results

All tests passing:

✅ Full name lookup: "Rob Wazowski" → finds patient  
✅ Partial name: "Rob" → finds patient  
✅ MRN lookup: "MRN-001" → finds patient  
✅ Common references: "demo patient" → finds patient  
✅ Conversation memory: "Tell me about Rob" → "What was his score?" (remembers context)  
✅ Multi-tool workflows: Agent calls `find_patient` first, then uses patient_id for other tools

## API Examples

### Find Patient
```bash
curl -X POST http://127.0.0.1:8001/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me clinical notes for Rob Wazowski",
    "user_id": "user-1",
    "role": "clinician"
  }'
```

**Agent workflow:**
1. Calls `find_patient("Rob Wazowski")` → gets `patient_id`
2. Calls `generate_clinical_note(patient_id)` → returns notes
3. Responds with formatted clinical information

### Get Transcript by Name
```bash
curl -X POST http://127.0.0.1:8001/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Give me the transcript for Rob",
    "user_id": "user-2",
    "role": "clinician"
  }'
```

## Adding New Patients

To add a new patient to the name lookup system, edit `data/patients.json`:

```json
{
  "new-patient-uuid": {
    "name": "Jane Doe",
    "mrn": "MRN-123",
    "dob": "1990-01-01",
    "created_at": "2026-03-16T00:00:00Z"
  }
}
```

The agent will automatically find them by name on the next request.

## Benefits

1. **Better UX**: Clinicians use names, not UUIDs
2. **Flexible matching**: Partial names, MRNs, or IDs all work
3. **Conversation memory**: "Tell me about Rob" → "What was his score?" works
4. **Backward compatible**: Old UUID-based queries still work
5. **Multi-tool support**: Agent chains `find_patient` with other tools automatically

## Technical Details

- **Tool**: `find_patient` (available to all roles)
- **Data source**: `data/patients.json`
- **Matching**: Case-insensitive partial match on name, MRN, or patient_id
- **Fallback**: If no match, suggests using `list_all_patients`
- **Disambiguation**: If multiple matches, returns all with suggestion to be more specific
