# Test Prompts - Natural Language (With Patient Names)

## ✅ Now You Can Use These Natural Prompts!

### **Basic Patient Lookup**

```
"Who is Rob Wazowski?"
```

```
"Tell me about the demo patient"
```

```
"Show me all information for patient MRN-001"
```

### **Clinical Notes**

```
"Show me the clinical notes for Rob Wazowski"
```

```
"What were the key themes in Rob's latest session?"
```

```
"Give me the risk assessment for the demo patient"
```

### **Transcripts**

```
"What did Rob Wazowski say in his latest session?"
```

```
"Show me the transcript for Rob"
```

```
"Give me the full transcript for the demo patient"
```

### **Congruence Scores**

```
"What was Rob's congruence score?"
```

```
"How is Rob Wazowski's emotional congruence trending?"
```

```
"Compare the congruence scores across Rob's last 3 sessions"
```

### **ICD-10 Codes**

```
"Suggest ICD-10 codes for Rob Wazowski based on his latest session"
```

```
"What diagnostic codes would you recommend for the demo patient?"
```

### **Multi-Step Queries**

```
"Give me a complete overview of Rob Wazowski: his history, latest notes, and congruence trend"
```

```
"Show me the transcript and clinical notes for Rob's latest session, then suggest ICD-10 codes"
```

### **Conversation Memory (Multi-Turn)**

```
Turn 1: "Tell me about Rob Wazowski"
Turn 2: "What was his congruence score?"
Turn 3: "Show me his transcript"
```

```
Turn 1: "Who is the demo patient?"
Turn 2: "What did they talk about in their latest session?"
Turn 3: "Were there any incongruent moments?"
```

### **Practice Analytics**

```
"How many patients do we have?"
```

```
"What's our practice's average congruence score?"
```

```
"Show me practice-wide analytics"
```

### **Insurance/Admin (Admin Role)**

```
"Generate an insurance packet for Rob Wazowski"
```

```
"Create a prior authorization for the demo patient"
```

## 🧪 Quick Test Script

```bash
#!/bin/bash
BASE_URL="http://127.0.0.1:8001/agent/chat"

# Test with patient name
curl -X POST "$BASE_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me clinical notes for Rob Wazowski",
    "user_id": "test-user",
    "role": "clinician"
  }' | jq '.response, .tools_used'
```

## 📋 Comparison: Before vs After

### Before (UUID Required)
```
❌ "Show me notes for 4e3c1260-9e27-4cc8-9720-114e068d03f1"
❌ "What was patient 4e3c1260-9e27-4cc8-9720-114e068d03f1's score?"
```

### After (Natural Language)
```
✅ "Show me notes for Rob Wazowski"
✅ "What was Rob's congruence score?"
✅ "Tell me about the demo patient"
```

## 🎯 Best Practices

1. **Use names**: "Rob Wazowski" instead of UUIDs
2. **Partial names work**: "Rob" will find "Rob Wazowski"
3. **MRNs work**: "MRN-001" will find the patient
4. **Common references**: "demo patient", "test patient" work
5. **Follow-up questions**: Agent remembers context across turns

## 🔧 Available Patients

Based on `data/patients.json`:

- **Rob Wazowski** (MRN-001)
- **Demo Patient** (MRN-DEMO)
- **Dev Test Patient** (MRN-DEV)
- **Test Patient** (MRN-TEST)
- And 7 more...

## 💡 Pro Tips

### Tip 1: Let the agent find patients
```
"Show me notes for Rob"  ← Agent calls find_patient first
```

### Tip 2: Use conversation memory
```
"Tell me about Rob Wazowski"
"What was his score?"  ← Agent remembers we're talking about Rob
"Show me his transcript"  ← Still remembers!
```

### Tip 3: Be specific if needed
```
"Show me notes for the patient with MRN-001"  ← Unambiguous
```

## ✅ Verified Working

All these prompts have been tested and work:

- ✅ Full name lookup
- ✅ Partial name matching
- ✅ MRN lookup
- ✅ Common patient references
- ✅ Multi-turn conversations
- ✅ Tool chaining (find_patient → other tools)
- ✅ Role-based permissions still enforced
- ✅ Backward compatible with UUIDs
