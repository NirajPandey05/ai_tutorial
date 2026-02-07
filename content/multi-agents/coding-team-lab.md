# Lab: Build a Coding Team

In this lab, you'll build a multi-agent coding team where agents collaborate to design, implement, review, and test code.

## Learning Objectives

By the end of this lab, you will:
- Create specialized coding agents (architect, developer, reviewer, tester)
- Implement a code review workflow
- Handle iterative improvement based on feedback
- Build an end-to-end code generation pipeline

## Prerequisites

- Python 3.9+
- OpenAI API key
- Understanding of multi-agent patterns from previous lessons

## Part 1: Project Setup

```
coding_team/
├── agents/
│   ├── __init__.py
│   ├── architect.py
│   ├── developer.py
│   ├── reviewer.py
│   └── tester.py
├── core/
│   ├── __init__.py
│   ├── code_state.py
│   └── team.py
├── output/           # Generated code goes here
├── main.py
└── requirements.txt
```

**requirements.txt:**
```
openai>=1.0.0
python-dotenv
```

## Part 2: Code State Management

**core/code_state.py**:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class CodeStatus(Enum):
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"
    TESTING = "testing"
    COMPLETE = "complete"
    NEEDS_REVISION = "needs_revision"

@dataclass
class CodeFile:
    """Represents a code file."""
    name: str
    content: str
    language: str = "python"
    version: int = 1
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update(self, new_content: str, reason: str):
        """Update the file content."""
        self.history.append({
            "version": self.version,
            "content": self.content,
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        })
        self.content = new_content
        self.version += 1

@dataclass
class CodeProject:
    """Represents a code project."""
    name: str
    description: str
    requirements: List[str]
    design: Optional[str] = None
    files: Dict[str, CodeFile] = field(default_factory=dict)
    status: CodeStatus = CodeStatus.DESIGN
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_file(self, name: str, content: str, language: str = "python"):
        """Add a file to the project."""
        self.files[name] = CodeFile(name=name, content=content, language=language)
    
    def add_review(self, reviewer: str, approved: bool, feedback: str, file_feedback: Dict[str, str] = None):
        """Add a review."""
        self.reviews.append({
            "reviewer": reviewer,
            "approved": approved,
            "feedback": feedback,
            "file_feedback": file_feedback or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def add_test_result(self, tester: str, passed: bool, results: Dict[str, Any]):
        """Add test results."""
        self.test_results.append({
            "tester": tester,
            "passed": passed,
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
```

## Part 3: Coding Agents

**agents/architect.py**:

```python
from typing import Dict, Any, List
from openai import AsyncOpenAI

class ArchitectAgent:
    """Designs software architecture and structure."""
    
    def __init__(self, llm_client: AsyncOpenAI, model: str = "gpt-4"):
        self.llm = llm_client
        self.model = model
        self.name = "Software Architect"
    
    async def create_design(
        self,
        project_name: str,
        description: str,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Create a software design."""
        
        requirements_text = "\n".join(f"- {r}" for r in requirements)
        
        prompt = f"""Design a software solution for:

Project: {project_name}
Description: {description}

Requirements:
{requirements_text}

Provide a complete design including:

1. **Architecture Overview**: High-level structure and patterns
2. **File Structure**: List of files to create with their purposes
3. **Class/Function Design**: Key classes and functions for each file
4. **Data Flow**: How data moves through the system
5. **Dependencies**: External libraries needed

Format the file structure as:
FILES:
- filename.py: purpose
- filename2.py: purpose

Be specific and practical. Design for clean, maintainable code."""
        
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert software architect."},
                {"role": "user", "content": prompt}
            ]
        )
        
        design = response.choices[0].message.content
        
        # Extract files from design
        files = self._extract_files(design)
        
        return {
            "design": design,
            "files": files,
            "architect": self.name
        }
    
    def _extract_files(self, design: str) -> List[Dict[str, str]]:
        """Extract file list from design."""
        
        files = []
        
        if "FILES:" in design:
            files_section = design.split("FILES:")[1].split("\n\n")[0]
            
            for line in files_section.strip().split("\n"):
                if line.strip().startswith("-"):
                    parts = line.strip("- ").split(":")
                    if len(parts) >= 2:
                        files.append({
                            "name": parts[0].strip(),
                            "purpose": parts[1].strip()
                        })
        
        # Default files if none extracted
        if not files:
            files = [
                {"name": "main.py", "purpose": "Main entry point"},
                {"name": "core.py", "purpose": "Core functionality"}
            ]
        
        return files
    
    async def refine_design(
        self,
        original_design: str,
        feedback: str
    ) -> Dict[str, Any]:
        """Refine design based on feedback."""
        
        prompt = f"""Refine this software design based on feedback:

Original Design:
{original_design}

Feedback:
{feedback}

Provide the updated design addressing all feedback points."""
        
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "design": response.choices[0].message.content,
            "files": self._extract_files(response.choices[0].message.content),
            "architect": self.name
        }
```

**agents/developer.py**:

```python
from typing import Dict, Any, List
from openai import AsyncOpenAI

class DeveloperAgent:
    """Implements code based on designs."""
    
    def __init__(self, llm_client: AsyncOpenAI, model: str = "gpt-4"):
        self.llm = llm_client
        self.model = model
        self.name = "Senior Developer"
    
    async def implement_file(
        self,
        file_name: str,
        file_purpose: str,
        design: str,
        existing_files: Dict[str, str] = None
    ) -> str:
        """Implement a single file."""
        
        existing_context = ""
        if existing_files:
            existing_context = "Existing files in project:\n"
            for name, content in existing_files.items():
                # Show structure, not full content
                existing_context += f"\n--- {name} ---\n"
                existing_context += content[:500] + "\n...\n"
        
        prompt = f"""Implement this file based on the design:

File: {file_name}
Purpose: {file_purpose}

Design Context:
{design[:2000]}

{existing_context}

Requirements:
1. Write complete, working code
2. Include docstrings and comments
3. Follow Python best practices
4. Handle errors appropriately
5. Make it compatible with other project files

Provide ONLY the code, no explanations. Start with imports."""
        
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert Python developer. Write clean, production-ready code."},
                {"role": "user", "content": prompt}
            ]
        )
        
        code = response.choices[0].message.content
        
        # Clean up code block markers if present
        if code.startswith("```"):
            code = code.split("\n", 1)[1]
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]
        
        return code.strip()
    
    async def implement_project(
        self,
        design: str,
        files: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """Implement all files in a project."""
        
        implemented = {}
        
        for file_info in files:
            code = await self.implement_file(
                file_name=file_info["name"],
                file_purpose=file_info["purpose"],
                design=design,
                existing_files=implemented
            )
            implemented[file_info["name"]] = code
        
        return implemented
    
    async def revise_file(
        self,
        file_name: str,
        current_code: str,
        feedback: str,
        design: str
    ) -> str:
        """Revise a file based on feedback."""
        
        prompt = f"""Revise this code based on feedback:

File: {file_name}

Current Code:
```python
{current_code}
```

Feedback to Address:
{feedback}

Design Reference:
{design[:1000]}

Provide the complete revised code. Address all feedback points."""
        
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert Python developer."},
                {"role": "user", "content": prompt}
            ]
        )
        
        code = response.choices[0].message.content
        
        if code.startswith("```"):
            code = code.split("\n", 1)[1]
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]
        
        return code.strip()
```

**agents/reviewer.py**:

```python
from typing import Dict, Any, List
from openai import AsyncOpenAI
import json

class ReviewerAgent:
    """Reviews code for quality and issues."""
    
    def __init__(self, llm_client: AsyncOpenAI, model: str = "gpt-4"):
        self.llm = llm_client
        self.model = model
        self.name = "Code Reviewer"
    
    async def review_file(self, file_name: str, code: str) -> Dict[str, Any]:
        """Review a single file."""
        
        prompt = f"""Review this Python code:

File: {file_name}
```python
{code}
```

Evaluate:
1. Code Quality (1-10)
2. Readability (1-10)
3. Error Handling (1-10)
4. Best Practices (1-10)

List specific issues found and suggestions.

Return as JSON:
{{
    "scores": {{
        "quality": 8,
        "readability": 7,
        "error_handling": 6,
        "best_practices": 8
    }},
    "issues": ["issue 1", "issue 2"],
    "suggestions": ["suggestion 1", "suggestion 2"],
    "approved": true/false
}}"""
        
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert code reviewer."},
                {"role": "user", "content": prompt}
            ]
        )
        
        try:
            content = response.choices[0].message.content
            # Extract JSON
            start = content.find("{")
            end = content.rfind("}") + 1
            return json.loads(content[start:end])
        except:
            return {
                "scores": {"quality": 5, "readability": 5, "error_handling": 5, "best_practices": 5},
                "issues": ["Could not parse review"],
                "suggestions": [],
                "approved": False
            }
    
    async def review_project(
        self,
        files: Dict[str, str],
        design: str
    ) -> Dict[str, Any]:
        """Review entire project."""
        
        # Review each file
        file_reviews = {}
        for name, code in files.items():
            file_reviews[name] = await self.review_file(name, code)
        
        # Overall review
        files_text = "\n\n".join(
            f"--- {name} ---\n{code[:500]}..."
            for name, code in files.items()
        )
        
        prompt = f"""Review this complete project:

Design:
{design[:1000]}

Files:
{files_text}

Individual file reviews:
{json.dumps(file_reviews, indent=2)}

Provide overall project review:
1. Does the implementation match the design?
2. Are all components working together?
3. What's missing or incomplete?
4. Overall approval recommendation?

Return as JSON:
{{
    "design_match": true/false,
    "integration_score": 8,
    "missing_items": ["item 1"],
    "overall_feedback": "summary",
    "approved": true/false
}}"""
        
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            content = response.choices[0].message.content
            start = content.find("{")
            end = content.rfind("}") + 1
            overall = json.loads(content[start:end])
        except:
            overall = {
                "design_match": False,
                "integration_score": 5,
                "missing_items": [],
                "overall_feedback": content,
                "approved": False
            }
        
        return {
            "file_reviews": file_reviews,
            "overall": overall,
            "reviewer": self.name
        }
```

**agents/tester.py**:

```python
from typing import Dict, Any, List
from openai import AsyncOpenAI
import json

class TesterAgent:
    """Creates and validates tests."""
    
    def __init__(self, llm_client: AsyncOpenAI, model: str = "gpt-4"):
        self.llm = llm_client
        self.model = model
        self.name = "QA Engineer"
    
    async def generate_tests(
        self,
        file_name: str,
        code: str
    ) -> str:
        """Generate tests for a file."""
        
        prompt = f"""Generate comprehensive tests for this code:

File: {file_name}
```python
{code}
```

Create tests using pytest that:
1. Test all public functions/methods
2. Include edge cases
3. Test error handling
4. Have clear test names

Provide complete test code."""
        
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert QA engineer."},
                {"role": "user", "content": prompt}
            ]
        )
        
        code = response.choices[0].message.content
        
        if code.startswith("```"):
            code = code.split("\n", 1)[1]
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]
        
        return code.strip()
    
    async def validate_code(
        self,
        files: Dict[str, str]
    ) -> Dict[str, Any]:
        """Validate code logic without running."""
        
        files_text = "\n\n".join(
            f"--- {name} ---\n```python\n{code}\n```"
            for name, code in files.items()
        )
        
        prompt = f"""Analyze this code and predict test outcomes:

{files_text}

For each file, identify:
1. Functions that would pass tests
2. Functions that might fail
3. Potential runtime errors
4. Logic issues

Return as JSON:
{{
    "predictions": {{
        "filename.py": {{
            "likely_pass": ["func1", "func2"],
            "likely_fail": ["func3"],
            "potential_errors": ["description"],
            "logic_issues": ["issue"]
        }}
    }},
    "overall_confidence": 0.8
}}"""
        
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            content = response.choices[0].message.content
            start = content.find("{")
            end = content.rfind("}") + 1
            return json.loads(content[start:end])
        except:
            return {
                "predictions": {},
                "overall_confidence": 0.5
            }
```

## Part 4: Team Coordinator

**core/team.py**:

```python
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

from core.code_state import CodeProject, CodeStatus
from agents.architect import ArchitectAgent
from agents.developer import DeveloperAgent
from agents.reviewer import ReviewerAgent
from agents.tester import TesterAgent

class CodingTeam:
    """Coordinates the coding team."""
    
    def __init__(self, llm_client, model: str = "gpt-4", output_dir: str = "./output"):
        self.architect = ArchitectAgent(llm_client, model)
        self.developer = DeveloperAgent(llm_client, model)
        self.reviewer = ReviewerAgent(llm_client, model)
        self.tester = TesterAgent(llm_client, model)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    async def build_project(
        self,
        name: str,
        description: str,
        requirements: list,
        max_iterations: int = 3
    ) -> CodeProject:
        """Build a complete project."""
        
        project = CodeProject(
            name=name,
            description=description,
            requirements=requirements
        )
        
        print(f"🏗️ Building: {name}")
        print("=" * 50)
        
        # Phase 1: Design
        print("\n📐 Phase 1: Architecture Design")
        design_result = await self.architect.create_design(
            name, description, requirements
        )
        project.design = design_result["design"]
        project.status = CodeStatus.IMPLEMENTATION
        print(f"   Designed {len(design_result['files'])} files")
        
        # Phase 2: Implementation
        print("\n💻 Phase 2: Implementation")
        files = await self.developer.implement_project(
            design_result["design"],
            design_result["files"]
        )
        
        for file_name, code in files.items():
            project.add_file(file_name, code)
        
        print(f"   Implemented {len(files)} files")
        project.status = CodeStatus.REVIEW
        
        # Phase 3: Review and Iterate
        for iteration in range(max_iterations):
            print(f"\n🔍 Phase 3: Review (iteration {iteration + 1})")
            
            review = await self.reviewer.review_project(
                {name: f.content for name, f in project.files.items()},
                project.design
            )
            
            project.add_review(
                self.reviewer.name,
                review["overall"]["approved"],
                review["overall"]["overall_feedback"],
                {name: str(r) for name, r in review["file_reviews"].items()}
            )
            
            if review["overall"]["approved"]:
                print("   ✅ Review passed!")
                break
            
            print("   ⚠️ Changes requested, revising...")
            project.status = CodeStatus.NEEDS_REVISION
            
            # Revise files based on feedback
            for file_name, file_review in review["file_reviews"].items():
                if not file_review.get("approved", True):
                    feedback = "\n".join(file_review.get("issues", []))
                    feedback += "\n" + "\n".join(file_review.get("suggestions", []))
                    
                    revised = await self.developer.revise_file(
                        file_name,
                        project.files[file_name].content,
                        feedback,
                        project.design
                    )
                    
                    project.files[file_name].update(revised, f"Revision {iteration + 1}")
            
            project.status = CodeStatus.REVIEW
        
        # Phase 4: Testing
        print("\n🧪 Phase 4: Testing")
        validation = await self.tester.validate_code(
            {name: f.content for name, f in project.files.items()}
        )
        
        project.add_test_result(
            self.tester.name,
            validation.get("overall_confidence", 0) > 0.7,
            validation
        )
        
        # Generate test files
        for file_name in project.files:
            if not file_name.startswith("test_"):
                tests = await self.tester.generate_tests(
                    file_name,
                    project.files[file_name].content
                )
                project.add_file(f"test_{file_name}", tests)
        
        print(f"   Generated {sum(1 for n in project.files if n.startswith('test_'))} test files")
        
        project.status = CodeStatus.COMPLETE
        
        # Save to disk
        await self._save_project(project)
        
        return project
    
    async def _save_project(self, project: CodeProject):
        """Save project files to disk."""
        
        project_dir = self.output_dir / project.name.replace(" ", "_").lower()
        project_dir.mkdir(exist_ok=True)
        
        for file_name, file_obj in project.files.items():
            file_path = project_dir / file_name
            file_path.write_text(file_obj.content)
        
        # Save design
        design_path = project_dir / "DESIGN.md"
        design_path.write_text(f"# {project.name}\n\n{project.design}")
        
        print(f"\n📁 Saved to: {project_dir}")
```

## Part 5: Main Application

**main.py**:

```python
import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

from core.team import CodingTeam

load_dotenv()

async def main():
    """Run the coding team."""
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    team = CodingTeam(client, output_dir="./output")
    
    # Define project
    project_name = "Task Manager CLI"
    description = """
    A command-line task management application that allows users to:
    - Add, list, complete, and delete tasks
    - Set task priorities
    - Filter tasks by status or priority
    - Persist tasks to a JSON file
    """
    
    requirements = [
        "Use argparse for CLI interface",
        "Store tasks in JSON file",
        "Support task priorities (low, medium, high)",
        "Support task statuses (pending, complete)",
        "Provide helpful error messages",
        "Include a main entry point"
    ]
    
    # Build project
    project = await team.build_project(
        name=project_name,
        description=description,
        requirements=requirements,
        max_iterations=2
    )
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 PROJECT SUMMARY")
    print("=" * 50)
    print(f"Name: {project.name}")
    print(f"Status: {project.status.value}")
    print(f"Files: {len(project.files)}")
    print(f"Reviews: {len(project.reviews)}")
    
    print("\n📁 Generated Files:")
    for name in project.files:
        print(f"   - {name}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Part 6: Run the Lab

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY=your-key-here

# Run
python main.py
```

## Expected Output

```
🏗️ Building: Task Manager CLI
==================================================

📐 Phase 1: Architecture Design
   Designed 3 files

💻 Phase 2: Implementation
   Implemented 3 files

🔍 Phase 3: Review (iteration 1)
   ⚠️ Changes requested, revising...

🔍 Phase 3: Review (iteration 2)
   ✅ Review passed!

🧪 Phase 4: Testing
   Generated 3 test files

📁 Saved to: ./output/task_manager_cli

==================================================
📊 PROJECT SUMMARY
==================================================
Name: Task Manager CLI
Status: complete
Files: 6
Reviews: 2

📁 Generated Files:
   - main.py
   - task_manager.py
   - storage.py
   - test_main.py
   - test_task_manager.py
   - test_storage.py
```

## Challenges

1. **Add Documentation Agent**: Create an agent that generates README and docstrings.

2. **Implement Real Testing**: Actually run the generated tests and feed results back.

3. **Add Debugging Agent**: Create an agent that can fix failing tests.

4. **Multi-Language Support**: Extend to support JavaScript or other languages.

## Summary

You've built a complete coding team with:
- ✅ Architect agent for design
- ✅ Developer agent for implementation
- ✅ Reviewer agent for code review
- ✅ Tester agent for test generation
- ✅ Iterative improvement workflow
- ✅ File persistence

This pattern can be extended for larger projects, different languages, and more complex workflows.

---

🎉 **Congratulations!** You've completed the Multi-Agent Workflows module!
