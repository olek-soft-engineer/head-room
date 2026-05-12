# README Redesign — Design Spec

**Date:** 2026-05-11  
**Status:** Approved  
**Reference:** [lean-ctx README](https://github.com/yvgude/lean-ctx) — structural and visual inspiration

---

## Goal

Redesign `README.md` to match the crispness and scannability of lean-ctx's README while
foregrounding what makes Headroom distinct: library + proxy + MCP (not CLI-only), reversible
compression (CCR), and cross-agent memory.

---

## Structure (ordered)

### 1. ASCII block logo

Six-row block-letter HEADROOM (same double-width box-drawing style as lean-ctx):

```
  ██╗  ██╗███████╗ █████╗ ██████╗ ██████╗  ██████╗  ██████╗ ███╗   ███╗
  ██║  ██║██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██╔═══██╗████╗ ████║
  ███████║█████╗  ███████║██║  ██║██████╔╝██║   ██║██║   ██║██╔████╔██║
  ██╔══██║██╔══╝  ██╔══██║██║  ██║██╔══██╗██║   ██║██║   ██║██║╚██╔╝██║
  ██║  ██║███████╗██║  ██║██████╔╝██║  ██║╚██████╔╝╚██████╔╝██║ ╚═╝ ██║
  ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝     ╚═╝
                  The context compression layer for AI agents
```

Tagline on line 7, centered, plain text (no `<h3>`).

### 2. Power-stats line (centered `<p>`)

```
60–95% fewer tokens · library · proxy · MCP · 6 algorithms · local-first · reversible
```

### 3. Badge row (centered `<p>`)

Keep current badges in current order:
CI · codecov · PyPI · npm · Kompress-base model · Tokens saved · License · Docs

### 4. Nav links (centered `<p>`)

```
Docs · Install · Proof · Agents · Discord
```

Anchor-linked to section IDs in the document:
- **Docs** → `#documentation`
- **Install** → `#get-started-60-seconds`
- **Proof** → `#proof`
- **Agents** → `#agent-compatibility-matrix`
- **Discord** → external Discord URL

---

### 5. Horizontal rule + blockquote elevator pitch

```
> Headroom compresses everything your AI agent reads — tool outputs, logs, RAG chunks,
> files, and conversation history — before it reaches the LLM. Same answers, fraction of the tokens.
```

Immediately followed by the existing `HeadroomDemo-Fast.gif`, centered using raw HTML:

```html
<p align="center">
  <img src="HeadroomDemo-Fast.gif" alt="Headroom in action" width="820">
  <br/><sub>Live: 10,144 → 1,260 tokens — same FATAL found.</sub>
</p>
```

---

### 6. What it does

Seven bullets, one clause each. No prose paragraphs.

- **Library** — `compress(messages)` in Python or TypeScript, inline in any app
- **Proxy** — `headroom proxy --port 8787`, zero code changes, any language
- **Agent wrap** — `headroom wrap claude|codex|cursor|aider|copilot` in one command
- **MCP server** — `headroom_compress`, `headroom_retrieve`, `headroom_stats` for any MCP client
- **Cross-agent memory** — shared store across Claude, Codex, Gemini, auto-dedup
- **`headroom learn`** — mines failed sessions, writes corrections to `CLAUDE.md` / `AGENTS.md`
- **Reversible (CCR)** — originals never deleted; LLM retrieves on demand

---

### 7. How it works (30 seconds)

Keep the existing ASCII pipeline diagram verbatim. Replace all prose beneath it with exactly
four tight bullets (no more than one clause each), mirroring lean-ctx's pattern:

- **ContentRouter** — detects content type, selects the right compressor
- **SmartCrusher / CodeCompressor / Kompress-base** — compress JSON, AST, or prose
- **CacheAligner** — stabilizes prefixes so provider KV caches actually hit
- **CCR** — stores originals locally; LLM calls `headroom_retrieve` if it needs them

Drop the "Canonical pipeline lifecycle" paragraphs and "Provider slices" paragraphs from main
body — they move to `<details>` (see Section 12).

---

### 8. Get started (60 seconds)

Three numbered steps, bash only — no prose in between:

```bash
# 1 — Install
pip install "headroom-ai[all]"          # Python
npm install headroom-ai                 # Node / TypeScript

# 2 — Pick your mode
headroom wrap claude                    # wrap a coding agent
headroom proxy --port 8787              # drop-in proxy, zero code changes
# or: from headroom import compress      # inline library

# 3 — See the savings
headroom stats
```

One-liner extras note below the block:
`Granular extras: [proxy], [mcp], [ml], [agno], [langchain], [evals]. Requires Python 3.10+`

---

### 9. Proof

Keep both existing tables verbatim (workloads savings + accuracy benchmarks).  
Keep `headroom-savings.png` with the leaderboard link.  
Keep the reproduce snippet: `python -m headroom.evals suite --tier 1`

---

### 10. Agent compatibility matrix

Replace the current verbose "Notes" column with ≤5-word entries. Add a `●` column for wrap support.

| Agent | `headroom wrap` | Notes |
|---|:---:|---|
| Claude Code | ● | `--memory` · `--code-graph` |
| Codex | ● | shares memory with Claude |
| Cursor | ● | prints config — paste once |
| Aider | ● | starts proxy + launches |
| Copilot CLI | ● | starts proxy + launches |
| OpenClaw | ● | installs as ContextEngine plugin |

Footer note: *Any OpenAI-compatible client works via `headroom proxy`.*

---

### 11. When to use · When to skip  *(new section)*

**Great fit if you…**
- run AI coding agents daily and want savings without changing your code
- work across multiple agents and want shared memory
- need reversible compression — originals always retrievable via CCR

**Skip it if you…**
- only use a single provider's native compaction and don't need cross-agent memory
- work in a sandboxed environment where local processes can't run

---

### 12. Three `<details>` blocks

Each collapsed by default. Exact `<summary>` labels specified below.

**a. `<summary><b>Integrations — drop Headroom into any stack</b></summary>`**  
Copy the current `<details>` block with heading "Drop Headroom into any stack" (lines ~196–213 of current README) verbatim — the integrations table is already inside a `<details>` today.

**b. `<summary><b>What's inside</b></summary>`**  
Copy the current `<details>` block with heading "What's inside" (lines ~215–228 of current README) verbatim — SmartCrusher, CodeCompressor, Kompress-base, image compression, CacheAligner, IntelligentContext, CCR, cross-agent memory, SharedContext, `headroom learn`.

**c. `<summary><b>Pipeline internals</b></summary>`**  
Move the "Canonical pipeline lifecycle" section (lines ~107–135 of current README, beginning "Headroom now exposes one stable request lifecycle…") and the "Provider slices" section (lines ~137–148, beginning "Provider and tool-specific behavior…") here. Content unchanged — just relocated out of the main body.

---

### 13. Install

Keep current `pip` / `npm` / `docker` block verbatim.  
Keep extras and Python version note.  
**Note:** Both §8 (quick-start 60-second block) and §13 (full install matrix with docker + extras) are intentional and survive. §8 is the happy path; §13 is the complete reference.

---

### 14. headroom learn

Keep `headroom_learn.gif` in a slim subsection **immediately before Section 15 (Docs)**,
under a `## headroom learn` heading. Centered using raw HTML:

```html
<p align="center">
  <img src="headroom_learn.gif" alt="headroom learn in action" width="720">
</p>
```

One-sentence caption below: *"headroom learn — mines failed sessions, writes corrections to CLAUDE.md / AGENTS.md"*

---

### 15. Documentation

Keep current two-column doc links table verbatim.

---

### 16. Compared to

Keep current comparison table verbatim.  
Keep attribution paragraph for RTK and lean-ctx.

---

### 17. Contributing · Community · License

Contributing bash block unchanged.  
Community bullet list (leaderboard, Discord, Kompress-base) unchanged.  
License line unchanged.

---

## What changes vs. today

| Area | Before | After |
|---|---|---|
| Logo | Plain `# Headroom` | ASCII block logo (6 rows) |
| Tagline | 2-sentence paragraph | 1-line centered, below logo |
| Power stats | None | Centered line below tagline |
| Nav links | None | `Docs · Install · Proof · Agents · Discord` |
| Section headings | Flat prose | Time-boxed where helpful |
| Agent notes | 15–20 word sentences | ≤5 words |
| Pipeline/provider | In main body (~80 lines) | Collapsed `<details>` |
| "When to skip" | Missing | Added |
| GIF placement | Inline, no caption | Centered with token-count caption |

## What does NOT change

- All benchmark numbers and tables
- All code snippets
- All existing GIFs (HeadroomDemo-Fast.gif, headroom_learn.gif, headroom-savings.png)
- Attribution paragraph for RTK and lean-ctx
- Docs links
- Contributing / Community / License

---

## Non-goals

- No new GIFs or screenshots
- No 3-column demo table (no 3 separate GIFs available)
- No rewrite of benchmark methodology
- No changes to docs site
