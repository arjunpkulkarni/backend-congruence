# Congruence Ops Agent - Frontend Requirements

## Overview

Build a chat-based AI assistant interface for the **Congruence Ops Agent** that allows clinicians, administrators, and practice owners to manage therapy practice operations through natural language conversation.

The backend API is **fully implemented and working** at `http://127.0.0.1:8001`. Your job is to create a modern, intuitive frontend that integrates with these endpoints.

---

## 🎯 Core Requirements

### **1. Chat Interface**
Create a conversational UI similar to ChatGPT but optimized for clinical workflows.

**Key Features:**
- Real-time chat with the AI agent
- Message history with timestamps
- Typing indicators and loading states
- Support for multi-line messages
- Auto-scroll to latest messages
- Message status indicators (sent, delivered, error)

### **2. Role-Based Access**
The system supports 3 user roles with different permissions:

- **Clinician**: Patient records, clinical notes, ICD-10 codes
- **Admin**: Insurance, intake forms, billing, claims
- **Practice Owner**: All tools + analytics and scheduling

**Implementation:**
- Role selector/indicator in the UI
- Different starter prompts per role
- Role-appropriate action buttons

### **3. Action Cards**
The agent returns actionable buttons that users can click for quick actions.

**Example Actions:**
- "Select Patient" 
- "Generate Clinical Note"
- "View Practice Analytics"
- "Manage Intake Forms"

**Behavior:**
- Render action buttons below agent responses
- Clicking an action should either:
  - Send a follow-up message to the agent, OR
  - Navigate to a relevant page/modal

### **4. Context Panel**
Show relevant context information alongside the chat.

**Context Items:**
- Current user role
- Selected patient (if any)
- Current session (if any)
- Recent activity

---

## 🔌 Backend API Integration

### **Base URL:** `http://127.0.0.1:8001`

### **Endpoints to Integrate:**

#### **1. Agent Status**
```http
GET /agent/status
```

**Response:**
```json
{
  "status": "ready",
  "model": "gpt-4o-mini", 
  "tools_count": 9,
  "message": "Congruence Ops Agent is ready"
}
```

**Use Case:** Check if agent is available before showing chat interface.

#### **2. Agent Chat**
```http
POST /agent/chat
Content-Type: application/json
```

**Request:**
```json
{
  "message": "Generate a SOAP note for today's session with Sarah",
  "user_id": "dr_smith",
  "role": "clinician",
  "context": {
    "selected_patient": "sarah_123",
    "selected_session": null
  }
}
```

**Response:**
```json
{
  "response": "I need to use the generate_clinical_note tool for that. This tool will be implemented in a later iteration.\n\nHowever, I can guide you on how to structure a SOAP note if you'd like. Would you like assistance with that?",
  "actions": [
    {
      "type": "generate_note",
      "label": "Generate Clinical Note", 
      "data": {}
    }
  ],
  "tools_used": [],
  "context": {
    "selected_patient": "sarah_123"
  },
  "metadata": {
    "model_used": "gpt-4o-mini"
  }
}
```

---

## 🎨 UI/UX Specifications

### **Layout Structure**
```
┌─────────────────────────────────────────────────────────┐
│ Header: "Congruence Ops Agent" + Role Indicator        │
├─────────────────────────────────┬───────────────────────┤
│                                 │ Context Panel         │
│                                 │ ┌─────────────────────┤
│                                 │ │ Role: Clinician     │
│ Chat Messages Area              │ │ Patient: Sarah M.   │
│ ┌─────────────────────────────┐ │ │ Session: None       │
│ │ User: Generate SOAP note    │ │ │                     │
│ │ for Sarah                   │ │ └─────────────────────┤
│ └─────────────────────────────┘ │                       │
│ ┌─────────────────────────────┐ │ Starter Prompts       │
│ │ Agent: I need to use the    │ │ ┌─────────────────────┤
│ │ generate_clinical_note...   │ │ │ • Generate SOAP     │
│ │                             │ │ │ • Show schedule     │
│ │ [Generate Clinical Note]    │ │ │ • Check claims      │
│ └─────────────────────────────┘ │ └─────────────────────┤
├─────────────────────────────────┤                       │
│ Input: Type your message...     │                       │
│                          [Send] │                       │
└─────────────────────────────────┴───────────────────────┘
```

### **Message Types**

#### **User Messages**
```jsx
<div className="user-message">
  <div className="message-content">
    Generate a SOAP note for today's session with Sarah
  </div>
  <div className="message-time">2:34 PM</div>
</div>
```

#### **Agent Messages**
```jsx
<div className="agent-message">
  <div className="message-content">
    I need to use the generate_clinical_note tool for that. 
    This tool will be implemented in a later iteration.
    
    However, I can guide you on how to structure a SOAP note 
    if you'd like. Would you like assistance with that?
  </div>
  <div className="action-buttons">
    <button className="action-btn" data-type="generate_note">
      Generate Clinical Note
    </button>
  </div>
  <div className="message-time">2:34 PM</div>
</div>
```

### **Loading States**
```jsx
<div className="agent-message loading">
  <div className="typing-indicator">
    <span></span><span></span><span></span>
  </div>
</div>
```

---

## 📱 Responsive Design

### **Desktop (1200px+)**
- Side-by-side chat and context panel
- Full-width message bubbles
- Spacious action buttons

### **Tablet (768px - 1199px)**  
- Collapsible context panel
- Slightly narrower messages
- Touch-friendly buttons

### **Mobile (< 768px)**
- Full-width chat interface
- Context panel as slide-over/modal
- Mobile-optimized input area
- Swipe gestures for actions

---

## 🎭 Role-Specific Features

### **Clinician Role**
**Starter Prompts:**
- "Generate a SOAP note for today's session"
- "Suggest ICD-10 codes for anxiety symptoms"
- "Show patient history for [patient name]"
- "Create treatment plan recommendations"

**Available Actions:**
- Select Patient
- Generate Clinical Note  
- View Patient Record
- Get Session Transcript

### **Admin Role**
**Starter Prompts:**
- "Which patients have incomplete intake forms?"
- "Generate insurance packet for [patient]"
- "Check claim status for recent submissions"
- "Send intake forms to new patients"

**Available Actions:**
- Manage Intake Forms
- Generate Insurance Packet
- Check Claim Status
- Send Intake Form

### **Practice Owner Role**
**Starter Prompts:**
- "Show practice analytics for this month"
- "Which clinicians are underbooked?"
- "What's the revenue trend this quarter?"
- "Schedule appointments for tomorrow"

**Available Actions:**
- View Practice Analytics
- Schedule Appointment
- All clinician/admin actions

---

## 🔧 Technical Implementation

### **State Management**
```typescript
interface ChatState {
  messages: ChatMessage[];
  isLoading: boolean;
  currentRole: 'clinician' | 'admin' | 'practice_owner';
  context: {
    selectedPatient?: string;
    selectedSession?: string;
  };
  agentStatus: 'ready' | 'error' | 'loading';
}

interface ChatMessage {
  id: string;
  type: 'user' | 'agent';
  content: string;
  timestamp: Date;
  actions?: AgentAction[];
  metadata?: any;
}

interface AgentAction {
  type: string;
  label: string;
  data: any;
}
```

### **API Integration**
```typescript
class CongruenceAgentAPI {
  private baseUrl = 'http://127.0.0.1:8001';
  
  async getStatus(): Promise<AgentStatus> {
    const response = await fetch(`${this.baseUrl}/agent/status`);
    return response.json();
  }
  
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/agent/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    return response.json();
  }
}
```

### **Message Handling**
```typescript
const handleSendMessage = async (message: string) => {
  // Add user message to chat
  addMessage({
    type: 'user',
    content: message,
    timestamp: new Date()
  });
  
  // Show loading state
  setIsLoading(true);
  
  try {
    // Send to agent
    const response = await agentAPI.sendMessage({
      message,
      user_id: currentUser.id,
      role: currentRole,
      context: chatContext
    });
    
    // Add agent response
    addMessage({
      type: 'agent', 
      content: response.response,
      actions: response.actions,
      timestamp: new Date()
    });
    
    // Update context
    updateContext(response.context);
    
  } catch (error) {
    // Handle error
    addMessage({
      type: 'agent',
      content: 'Sorry, I encountered an error. Please try again.',
      timestamp: new Date()
    });
  } finally {
    setIsLoading(false);
  }
};
```

### **Action Button Handling**
```typescript
const handleActionClick = (action: AgentAction) => {
  switch (action.type) {
    case 'generate_note':
      // Either send follow-up message or open modal
      handleSendMessage('Please generate the clinical note now');
      break;
      
    case 'select_patient':
      // Open patient selector modal
      openPatientSelector();
      break;
      
    case 'view_analytics':
      // Navigate to analytics page or send message
      handleSendMessage('Show me the practice analytics dashboard');
      break;
      
    default:
      // Generic handler - send action as message
      handleSendMessage(`Execute ${action.label}`);
  }
};
```

---

## 🎨 Styling Guidelines

### **Color Scheme**
- **Primary**: Clinical blue (#2563eb)
- **Secondary**: Soft gray (#64748b) 
- **Success**: Medical green (#059669)
- **Warning**: Attention orange (#d97706)
- **Error**: Alert red (#dc2626)
- **Background**: Clean white (#ffffff)
- **Surface**: Light gray (#f8fafc)

### **Typography**
- **Headers**: Inter/Roboto, 600 weight
- **Body**: Inter/Roboto, 400 weight  
- **Code/Data**: Monaco/Consolas, 400 weight

### **Message Styling**
```css
.user-message {
  background: #2563eb;
  color: white;
  margin-left: auto;
  max-width: 70%;
  border-radius: 18px 18px 4px 18px;
}

.agent-message {
  background: #f8fafc;
  color: #1e293b;
  margin-right: auto;
  max-width: 80%;
  border-radius: 18px 18px 18px 4px;
  border: 1px solid #e2e8f0;
}

.action-btn {
  background: #2563eb;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 8px;
  margin: 4px;
  cursor: pointer;
  font-size: 14px;
}

.action-btn:hover {
  background: #1d4ed8;
}
```

---

## 🧪 Testing Scenarios

### **Basic Chat Flow**
1. User selects "Clinician" role
2. User types: "Generate SOAP note for Sarah"
3. Agent responds with tool acknowledgment
4. User clicks "Generate Clinical Note" action
5. Agent provides guidance or next steps

### **Role Switching**
1. User switches from "Clinician" to "Practice Owner"
2. Starter prompts update appropriately
3. User asks for analytics
4. Agent provides analytics-related response

### **Error Handling**
1. Network error during message send
2. Agent service unavailable
3. Invalid role or malformed request

### **Context Management**
1. User selects a patient
2. Context panel updates
3. Subsequent messages include patient context
4. Agent responses reference selected patient

---

## 🚀 Implementation Priority

### **Phase 1: Core Chat (Week 1)**
- [ ] Basic chat interface with message bubbles
- [ ] API integration for sending/receiving messages
- [ ] Role selector and basic role handling
- [ ] Loading states and error handling

### **Phase 2: Actions & Context (Week 2)** 
- [ ] Action button rendering and handling
- [ ] Context panel with patient/session info
- [ ] Starter prompts per role
- [ ] Message persistence/history

### **Phase 3: Polish & Mobile (Week 3)**
- [ ] Responsive design for mobile/tablet
- [ ] Advanced styling and animations
- [ ] Keyboard shortcuts and accessibility
- [ ] Performance optimization

---

## 📋 Acceptance Criteria

### **Must Have**
- ✅ Chat interface with user/agent messages
- ✅ Role-based starter prompts and permissions
- ✅ Action buttons that trigger follow-up interactions
- ✅ Real-time communication with backend API
- ✅ Context panel showing current state
- ✅ Mobile-responsive design
- ✅ Error handling for network/API issues

### **Should Have**
- ✅ Message timestamps and status indicators
- ✅ Typing indicators and loading states
- ✅ Keyboard shortcuts (Enter to send, etc.)
- ✅ Message history persistence
- ✅ Copy/share message functionality

### **Nice to Have**
- ✅ Message search functionality
- ✅ Export chat history
- ✅ Voice input/output
- ✅ Dark mode support
- ✅ Customizable themes per role

---

## 🔗 Backend API Reference

The backend is **live and working** at `http://127.0.0.1:8001`

**Test it now:**
```bash
# Check agent status
curl http://127.0.0.1:8001/agent/status

# Send a chat message
curl -X POST http://127.0.0.1:8001/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, I need help with clinical documentation",
    "user_id": "test_user", 
    "role": "clinician",
    "context": {}
  }'
```

**Expected Response:**
```json
{
  "response": "I can help you with clinical documentation...",
  "actions": [
    {"type": "generate_note", "label": "Generate Clinical Note", "data": {}}
  ],
  "tools_used": [],
  "context": {},
  "metadata": {"model_used": "gpt-4o-mini"}
}
```

---

## 🎯 Success Metrics

- **Usability**: Users can complete common tasks (generate notes, check analytics) in < 3 clicks
- **Performance**: Messages send/receive in < 2 seconds
- **Accessibility**: WCAG 2.1 AA compliance
- **Mobile**: Fully functional on mobile devices
- **Error Recovery**: Graceful handling of network/API errors

---

**The backend is ready and waiting for your frontend! Start building and test against the live API at `http://127.0.0.1:8001` 🚀**