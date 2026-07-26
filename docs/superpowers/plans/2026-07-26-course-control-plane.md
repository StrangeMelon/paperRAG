# Course Control Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create durable Chinese course documents that let a learner resume the Paper RAG rebuild without relying on conversational memory.

**Architecture:** The workspace separates course metadata from the eventual runtime package. `COURSE_PLAN.md` defines scope and phases, `SOURCE_MANIFEST.md` defines the source-to-target index, and `LEARNING_STATE.md` is the single mutable resume point.

**Tech Stack:** Markdown, Git, SHA-256, Python/Pytest project conventions.

---

### Task 1: Establish the course roadmap

**Files:**
- Create: `COURSE_PLAN.md`

- [ ] **Step 1: Record scope and explicit exclusions**

Write the exact source baseline, runtime-equivalence target, included Paper RAG/DeerFlow integration scope, and frontend/upstream exclusions.

- [ ] **Step 2: Record ordered phase boundaries**

List P0 through P12 in dependency order so every phase reaches a runnable, testable state before the next begins.

### Task 2: Establish resume and source controls

**Files:**
- Create: `LEARNING_STATE.md`
- Create: `SOURCE_MANIFEST.md`

- [ ] **Step 1: Create the mutable resume point**

Record the current phase, prior completion, next file, verified constraints, validation evidence, and required end-of-lesson fields.

- [ ] **Step 2: Create the static source index**

Map every Paper RAG package, project script/config group, test group, and project-specific DeerFlow integration to one course phase.

### Task 3: Verify the course control plane

**Files:**
- Verify: `COURSE_PLAN.md`
- Verify: `LEARNING_STATE.md`
- Verify: `SOURCE_MANIFEST.md`

- [ ] **Step 1: Confirm all files exist and are readable**

Run: `find . -maxdepth 3 -type f | sort`

Expected: the three course-control documents and this plan are present.

- [ ] **Step 2: Confirm the declared next project file is unique**

Run: `rg -n "下一个项目文件" LEARNING_STATE.md`

Expected: exactly one entry naming `pyproject.toml`.
