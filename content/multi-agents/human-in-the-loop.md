# Human-in-the-Loop Patterns

Learn how to integrate human oversight and control into multi-agent systems for safety and quality.

## Why Human-in-the-Loop?

While autonomous agents are powerful, human oversight is crucial for safety, quality, and accountability.

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Human-in-the-Loop Overview                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Fully Autonomous:              Human-in-the-Loop:                     │
│                                                                          │
│   ┌─────────┐                    ┌─────────┐                            │
│   │  Agent  │──────────────►     │  Agent  │                            │
│   └─────────┘    No checks       └────┬────┘                            │
│                                       │                                  │
│                                  ┌────▼────┐                            │
│                                  │ Approval │ ◄── Human checkpoint      │
│                                  │  Point   │                            │
│                                  └────┬────┘                            │
│                                       │                                  │
│   Risks:                              ▼                                  │
│   - Errors compound             Controlled                              │
│   - No accountability           execution                               │
│   - Safety concerns                                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Approval Patterns

### Pattern 1: Gate Approval

Agents pause at key gates for human approval.

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from datetime import datetime
import asyncio

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"

@dataclass
class ApprovalRequest:
    """Request for human approval."""
    id: str
    agent_id: str
    action: str
    details: Dict[str, Any]
    risk_level: str  # low, medium, high, critical
    created_at: datetime = field(default_factory=datetime.now)
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewer: Optional[str] = None
    feedback: Optional[str] = None
    resolved_at: Optional[datetime] = None


class ApprovalGate:
    """Gate requiring human approval to proceed."""
    
    def __init__(self, name: str, risk_threshold: str = "medium"):
        self.name = name
        self.risk_threshold = risk_threshold
        self.pending_requests: Dict[str, ApprovalRequest] = {}
        self.approval_callback: Optional[Callable] = None
    
    def set_approval_callback(self, callback: Callable):
        """Set callback for when approval is needed."""
        self.approval_callback = callback
    
    async def request_approval(
        self,
        agent_id: str,
        action: str,
        details: Dict[str, Any],
        risk_level: str = "medium"
    ) -> ApprovalRequest:
        """Request approval for an action."""
        
        request = ApprovalRequest(
            id=f"approval_{datetime.now().strftime('%Y%m%d%H%M%S')}_{agent_id}",
            agent_id=agent_id,
            action=action,
            details=details,
            risk_level=risk_level
        )
        
        self.pending_requests[request.id] = request
        
        # Notify callback
        if self.approval_callback:
            await self.approval_callback(request)
        
        return request
    
    async def wait_for_approval(
        self,
        request_id: str,
        timeout: float = 300.0
    ) -> ApprovalRequest:
        """Wait for approval decision."""
        
        start = datetime.now()
        
        while (datetime.now() - start).total_seconds() < timeout:
            request = self.pending_requests.get(request_id)
            
            if request and request.status != ApprovalStatus.PENDING:
                return request
            
            await asyncio.sleep(1.0)
        
        # Timeout - auto-reject for safety
        request = self.pending_requests.get(request_id)
        if request:
            request.status = ApprovalStatus.REJECTED
            request.feedback = "Approval timeout - auto-rejected for safety"
        
        return request
    
    def approve(self, request_id: str, reviewer: str, feedback: str = ""):
        """Approve a request."""
        
        if request_id in self.pending_requests:
            request = self.pending_requests[request_id]
            request.status = ApprovalStatus.APPROVED
            request.reviewer = reviewer
            request.feedback = feedback
            request.resolved_at = datetime.now()
    
    def reject(self, request_id: str, reviewer: str, feedback: str):
        """Reject a request."""
        
        if request_id in self.pending_requests:
            request = self.pending_requests[request_id]
            request.status = ApprovalStatus.REJECTED
            request.reviewer = reviewer
            request.feedback = feedback
            request.resolved_at = datetime.now()
    
    def request_revision(self, request_id: str, reviewer: str, feedback: str):
        """Request revision before approval."""
        
        if request_id in self.pending_requests:
            request = self.pending_requests[request_id]
            request.status = ApprovalStatus.NEEDS_REVISION
            request.reviewer = reviewer
            request.feedback = feedback


class ApprovalGatedAgent:
    """Agent that requires approval for certain actions."""
    
    def __init__(self, agent_id: str, llm_client: Any, approval_gate: ApprovalGate):
        self.id = agent_id
        self.llm = llm_client
        self.gate = approval_gate
        self.risk_assessment_enabled = True
    
    async def execute_with_approval(
        self,
        action: str,
        details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an action with approval if needed."""
        
        # Assess risk
        risk = await self._assess_risk(action, details)
        
        # Low risk - auto-approve
        if risk == "low":
            return await self._execute(action, details)
        
        # Higher risk - request approval
        request = await self.gate.request_approval(
            self.id, action, details, risk
        )
        
        print(f"[{self.id}] Waiting for approval: {request.id}")
        
        # Wait for decision
        result = await self.gate.wait_for_approval(request.id)
        
        if result.status == ApprovalStatus.APPROVED:
            return await self._execute(action, details)
        elif result.status == ApprovalStatus.NEEDS_REVISION:
            # Revise and retry
            revised = await self._revise(action, details, result.feedback)
            return await self.execute_with_approval(action, revised)
        else:
            return {
                "status": "rejected",
                "reason": result.feedback
            }
    
    async def _assess_risk(self, action: str, details: Dict) -> str:
        """Assess risk level of an action."""
        
        if not self.risk_assessment_enabled:
            return "medium"
        
        prompt = f"""Assess the risk level of this action:

Action: {action}
Details: {details}

Consider:
- Could this cause harm?
- Is this reversible?
- What's the potential impact?

Return only: low, medium, high, or critical"""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        risk = response.choices[0].message.content.strip().lower()
        
        if risk in ["low", "medium", "high", "critical"]:
            return risk
        return "medium"
    
    async def _execute(self, action: str, details: Dict) -> Dict[str, Any]:
        """Execute the action."""
        # Implementation depends on action type
        return {"status": "completed", "action": action}
    
    async def _revise(self, action: str, details: Dict, feedback: str) -> Dict:
        """Revise based on feedback."""
        
        prompt = f"""Revise this action based on feedback:

Action: {action}
Original Details: {details}
Feedback: {feedback}

Provide revised details that address the feedback."""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {**details, "revision": response.choices[0].message.content}
```

### Pattern 2: Escalation

Agents escalate to humans when uncertain.

```python
class EscalationManager:
    """Manage escalations to human operators."""
    
    def __init__(self):
        self.escalations: List[Dict[str, Any]] = []
        self.handlers: Dict[str, Callable] = {}
    
    def register_handler(self, escalation_type: str, handler: Callable):
        """Register a handler for an escalation type."""
        self.handlers[escalation_type] = handler
    
    async def escalate(
        self,
        agent_id: str,
        reason: str,
        context: Dict[str, Any],
        escalation_type: str = "general"
    ) -> Dict[str, Any]:
        """Escalate an issue to a human."""
        
        escalation = {
            "id": f"esc_{len(self.escalations)}",
            "agent_id": agent_id,
            "reason": reason,
            "context": context,
            "type": escalation_type,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        self.escalations.append(escalation)
        
        # Call registered handler
        if escalation_type in self.handlers:
            response = await self.handlers[escalation_type](escalation)
            escalation["response"] = response
            escalation["status"] = "handled"
            return response
        
        # Default: wait for manual handling
        return {"status": "escalated", "escalation_id": escalation["id"]}


class EscalatingAgent:
    """Agent that escalates when uncertain."""
    
    def __init__(
        self,
        agent_id: str,
        llm_client: Any,
        escalation_manager: EscalationManager,
        confidence_threshold: float = 0.7
    ):
        self.id = agent_id
        self.llm = llm_client
        self.escalation = escalation_manager
        self.confidence_threshold = confidence_threshold
    
    async def execute_task(self, task: str) -> Dict[str, Any]:
        """Execute a task, escalating if uncertain."""
        
        # Get initial response and confidence
        response, confidence = await self._try_task(task)
        
        if confidence >= self.confidence_threshold:
            return {
                "status": "completed",
                "result": response,
                "confidence": confidence
            }
        
        # Escalate
        escalation_result = await self.escalation.escalate(
            self.id,
            f"Low confidence ({confidence:.0%}) on task",
            {
                "task": task,
                "attempted_response": response,
                "confidence": confidence
            },
            escalation_type="low_confidence"
        )
        
        # If human provided answer, use it
        if "answer" in escalation_result:
            return {
                "status": "completed_with_help",
                "result": escalation_result["answer"],
                "escalation": escalation_result
            }
        
        # Return uncertain response
        return {
            "status": "uncertain",
            "result": response,
            "confidence": confidence,
            "escalation": escalation_result
        }
    
    async def _try_task(self, task: str) -> tuple[str, float]:
        """Try a task and assess confidence."""
        
        prompt = f"""Complete this task and rate your confidence:

Task: {task}

Provide:
1. Your response to the task
2. Your confidence level (0-100%)

Format:
RESPONSE: [your response]
CONFIDENCE: [number]%"""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.choices[0].message.content
        
        # Parse response and confidence
        try:
            response_part = text.split("RESPONSE:")[1].split("CONFIDENCE:")[0].strip()
            confidence_part = text.split("CONFIDENCE:")[1].strip()
            confidence = float(confidence_part.replace("%", "")) / 100
        except:
            response_part = text
            confidence = 0.5
        
        return response_part, confidence
```

### Pattern 3: Review Queue

All outputs go through a review queue.

```python
from queue import Queue
from threading import Lock

@dataclass
class ReviewItem:
    """Item awaiting review."""
    id: str
    agent_id: str
    output_type: str
    content: Any
    context: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    reviewed: bool = False
    reviewer: Optional[str] = None
    approved: Optional[bool] = None
    feedback: Optional[str] = None


class ReviewQueue:
    """Queue for human review of agent outputs."""
    
    def __init__(self):
        self.queue: List[ReviewItem] = []
        self.reviewed: List[ReviewItem] = []
        self._lock = Lock()
    
    def add(
        self,
        agent_id: str,
        output_type: str,
        content: Any,
        context: Dict[str, Any] = None
    ) -> str:
        """Add an item for review."""
        
        item = ReviewItem(
            id=f"review_{len(self.queue)}_{agent_id}",
            agent_id=agent_id,
            output_type=output_type,
            content=content,
            context=context or {}
        )
        
        with self._lock:
            self.queue.append(item)
        
        return item.id
    
    def get_pending(self, limit: int = 10) -> List[ReviewItem]:
        """Get pending review items."""
        
        with self._lock:
            pending = [item for item in self.queue if not item.reviewed]
            return pending[:limit]
    
    def review(
        self,
        item_id: str,
        reviewer: str,
        approved: bool,
        feedback: str = ""
    ):
        """Review an item."""
        
        with self._lock:
            for item in self.queue:
                if item.id == item_id:
                    item.reviewed = True
                    item.reviewer = reviewer
                    item.approved = approved
                    item.feedback = feedback
                    self.reviewed.append(item)
                    break
    
    def get_review_result(self, item_id: str) -> Optional[ReviewItem]:
        """Get the result of a review."""
        
        for item in self.reviewed:
            if item.id == item_id:
                return item
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get review statistics."""
        
        total_reviewed = len(self.reviewed)
        approved = sum(1 for item in self.reviewed if item.approved)
        
        return {
            "pending": len([item for item in self.queue if not item.reviewed]),
            "reviewed": total_reviewed,
            "approved": approved,
            "rejected": total_reviewed - approved,
            "approval_rate": approved / total_reviewed if total_reviewed > 0 else 0
        }


class ReviewedAgent:
    """Agent whose outputs go through review."""
    
    def __init__(self, agent_id: str, llm_client: Any, review_queue: ReviewQueue):
        self.id = agent_id
        self.llm = llm_client
        self.review_queue = review_queue
    
    async def generate_for_review(
        self,
        task: str,
        output_type: str = "general"
    ) -> str:
        """Generate output and submit for review."""
        
        # Generate
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": task}]
        )
        
        content = response.choices[0].message.content
        
        # Submit for review
        review_id = self.review_queue.add(
            self.id,
            output_type,
            content,
            {"task": task}
        )
        
        return review_id
    
    async def generate_and_wait(
        self,
        task: str,
        output_type: str = "general",
        timeout: float = 300.0
    ) -> Dict[str, Any]:
        """Generate, submit for review, and wait for result."""
        
        review_id = await self.generate_for_review(task, output_type)
        
        # Wait for review
        start = datetime.now()
        
        while (datetime.now() - start).total_seconds() < timeout:
            result = self.review_queue.get_review_result(review_id)
            
            if result:
                return {
                    "status": "reviewed",
                    "approved": result.approved,
                    "content": result.content,
                    "feedback": result.feedback
                }
            
            await asyncio.sleep(1.0)
        
        return {"status": "timeout", "review_id": review_id}
```

## Interactive Collaboration

```python
class InteractiveAgent:
    """Agent that collaborates interactively with humans."""
    
    def __init__(self, agent_id: str, llm_client: Any):
        self.id = agent_id
        self.llm = llm_client
        self.conversation_history: List[Dict[str, str]] = []
    
    async def propose_plan(self, goal: str) -> Dict[str, Any]:
        """Propose a plan for human feedback."""
        
        prompt = f"""Create a plan for this goal:

Goal: {goal}

Provide:
1. Step-by-step plan
2. Key decisions that need human input
3. Potential risks
4. Questions for the human

Format the plan clearly."""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        plan = response.choices[0].message.content
        
        self.conversation_history.append({
            "role": "assistant",
            "content": f"Proposed plan:\n{plan}"
        })
        
        return {
            "plan": plan,
            "awaiting_feedback": True
        }
    
    async def incorporate_feedback(self, feedback: str) -> Dict[str, Any]:
        """Incorporate human feedback into the plan."""
        
        self.conversation_history.append({
            "role": "user",
            "content": f"Feedback: {feedback}"
        })
        
        prompt = f"""The human provided this feedback on your plan:

{feedback}

Update your plan to address this feedback. Explain what you changed."""
        
        messages = self.conversation_history + [{"role": "user", "content": prompt}]
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        updated = response.choices[0].message.content
        
        self.conversation_history.append({
            "role": "assistant",
            "content": updated
        })
        
        return {
            "updated_plan": updated,
            "awaiting_feedback": True
        }
    
    async def execute_step(self, step: str, human_input: str = None) -> Dict[str, Any]:
        """Execute a step, optionally with human input."""
        
        prompt = f"""Execute this step:

Step: {step}

{"Human input: " + human_input if human_input else ""}

Provide the result of executing this step."""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.choices[0].message.content
        
        return {
            "step": step,
            "result": result,
            "human_input_used": human_input is not None
        }
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│             Human-in-the-Loop - Summary                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Patterns:                                                               │
│    • Gate Approval - Pause at checkpoints for approval                 │
│    • Escalation - Agents escalate when uncertain                       │
│    • Review Queue - All outputs reviewed before use                    │
│    • Interactive - Ongoing human-agent collaboration                   │
│                                                                          │
│  When to Use:                                                            │
│    • High-stakes decisions                                             │
│    • Safety-critical applications                                      │
│    • Quality-sensitive outputs                                         │
│    • Regulatory requirements                                           │
│                                                                          │
│  Benefits:                                                               │
│    • Maintains human control                                           │
│    • Catches errors before damage                                      │
│    • Builds trust in AI systems                                        │
│    • Provides accountability                                           │
│                                                                          │
│  Best Practices:                                                         │
│    • Auto-approve low-risk actions                                     │
│    • Clear escalation criteria                                         │
│    • Don't bottleneck on humans                                        │
│    • Learn from human feedback                                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Research Team Lab](/learn/multi-agents/advanced-multi-agent/research-team) →
