# CLAUDE.md — Figma Designer Agent

This file is the source of truth for how this project is built. Read it fully before
writing or changing code. It exists so any agent (or human) can pick up the work without
re-deriving the architecture.

**Two models, on purpose.** The **generator** (`MODEL_*`, currently `gpt-oss:20b-cloud` via
Ollama Cloud — free, reliable native tool calls, ~5s per call) writes Plugin API scripts and
runs ~50 times per design. The **vision critic** (`CRITIC_*`, optional) looks at a screenshot
of each finished section and reports what is visibly broken; it runs a handful of times, so a
paid hosted vision model costs very little here even though the generator is free. Neither
model is referenced outside `agent/llm.py` — both are `.env` lines. See sections 5 and 8b.

---

## 1. What this project is

A local agent that turns a **plain-language instruction** into a **complete Figma design**.
You type "a mobile sign-in screen with email + password and a Google button," and the
system builds it on a real Figma canvas — tokens, components, and layout, not a flat pile
of rectangles.

It also **edits** a design that already exists: connect to a file, say "make every primary
button use the accent colour", and it changes those nodes in place. Same harness, separate
pipeline — see section 20 for why they are not the same job.

And you do not have to type any of it. **Attach a screenshot** and it rebuilds what the
screenshot shows; attach a spec document and it builds to that; **dictate** the instruction
instead of typing it. All three become the same thing — words the pipeline already knows
how to build from — see section 21.

The agent does **not** know Figma from memory. It becomes competent through the harness:
docs are retrieved into context on demand (section 9), work happens in small atomic steps,
and every step is checked against ground truth — the file's metadata and its real geometry
(section 8) — then corrected. Treat the harness, not the model, as where quality comes from.

**Corollary learned the hard way:** if a piece of work is mechanical and always the same
shape, the harness does it *itself* rather than asking the model. See section 6a — this is
the single biggest source of reliability in the project.

### Screens are FRAMES, side by side on ONE page

Figma's own model, which this project got wrong for a long time: a **page** is
a workspace and a **frame** is a screen. Several screens are sibling frames
laid out side by side on one page.

```
PAGE: Website Design          <- one workspace, created by nobody
├── FRAME: Login              <- a screen
├── FRAME: Sign Up            <- a screen, beside it
└── FRAME: Dashboard          <- a screen, beside that
      ├── FRAME: Sidebar      <- a SECTION inside the dashboard
      └── FRAME: Content
```

The agent used to build **everything into one frame**, so "a login and a
dashboard" produced a sign-in form stacked on top of a dashboard in a single
tall column, and a re-run stamped fresh work over the old. Now:

- `planner.plan_screens` asks one small question — *which screens does this
  need?* — before anything is drawn. Sections (hero, nav, footer) are filtered
  out deterministically, because promoting one to a top-level frame scatters a
  screen's parts across the canvas.
- The harness creates **one frame per screen**, positions computed in Python so
  they can never overlap each other or existing work (section 6a).
- Each screen is **planned separately** and each step carries its `screen_index`,
  so a dashboard step cannot append into the login frame.
- A re-run matches a screen to an existing frame **by name**, so it extends
  `Login` instead of overwriting whichever frame happened to be largest.
- Most requests are one screen, and that path is unchanged.

Never create a Figma page per screen, and never `figma.createPage()`.

### The single most important design fact

You **cannot** write to the Figma canvas from Python. Figma's document is only mutable from
inside the Plugin API sandbox — JavaScript running inside the Figma desktop app. The REST
API is read-only for canvas structure. So the architecture is necessarily split:

- The **agent** (this Python code) is the brain. It plans and generates Plugin API JS.
- A **Figma plugin** is the hands. It receives JS over a WebSocket, runs it inside Figma,
  and sends back the result plus a screenshot.

---

## 2. Golden rules (do not violate these)

1. **The model is one swappable endpoint.** All model access goes through `agent/llm.py`.
   A `.env` change (or one extra client class), never a change scattered through the
   codebase. See section 5.
2. **Simplicity over cleverness.** Readable by someone new. Small functions, explicit
   control flow, type hints, standard library where possible. No agent frameworks — the
   loop is hand-rolled on purpose.
3. **Small atomic Figma steps.** One logical operation per script. Figma scripts are
   atomic: a failed script changes nothing, so retry is always safe.
4. **Validate every step — structurally and visually.** Read back metadata, and run the
   visual gate on every step that puts something on the canvas (section 8).
5. **Tokens -> components -> composition.** Never hardcode colors/spacing into final nodes.
   The harness creates the tokens itself (section 6a) and hands the model their names.
6. **Every Figma script returns the node IDs it touched.** State lives in Python, not in
   the model's memory.
7. **If it is mechanical, the harness writes it — not the model.** Root frame, design
   tokens, layout auditing and placeholder recovery are all Python-authored scripts. Each
   one was added *after* watching the model fail the same thing repeatedly (section 6a).
8. **Fix the harness from real traces, not from guesses.** Every gotcha and error hint in
   this repo came from an actual failing run. When something breaks, read the trace and
   close that specific hole.

---

## 3. Architecture

```
  Instruction  (CLI: main.py   |   Web dashboard: webapp.py)
      |
      v
+-----------------------------+        +--------------+        +-------------------+
|  AGENT  (Python, this repo) |        |   BRIDGE     |        |  FIGMA DESKTOP    |
|                             |        |  WebSocket   |        |                   |
|  llm.py     -> the model    |  JS -> |  localhost   |  JS -> |  plugin ui.html   |
|  loop.py    -> the loop     | -----> |              | -----> |  -> code.js       |
|  planner.py -> decomposition|        |  request/    |        |  -> Plugin API    |
|  scaffold.py-> harness JS   |        |  response    |        |  -> Canvas        |
|  critic.py  -> visual gate  | <----- |  matching    | <----- |                   |
|  tools/     -> exec + read  |  IDs,  |              |  IDs,  |  (returns result  |
|  knowledge/ -> docs         |  meta, |              |  meta, |   + screenshot)   |
|  state.py   -> run state    |  PNG   |              |  PNG   |                   |
+-----------------------------+        +--------------+        +-------------------+
      ^
      |  web/ -> dashboard: file gallery, settings, setup guide (webapp.py only)
```

Each part has one responsibility:

| Component | Responsibility |
|---|---|
| **Model client** (`agent/llm.py`) | The only place the model is configured. One method: `complete(messages, tools)`. The swap point. Also normalizes tool calls that small models emit as plain text. |
| **Agent loop** (`agent/loop.py`) | Drives the run: inspect -> scaffold -> plan -> per-step (generate -> execute -> structural gate -> visual gate -> retry). Hand-rolled. |
| **Planner** (`agent/planner.py`) | Expands the instruction into a design brief, then into an ordered plan. |
| **Scaffold** (`agent/scaffold.py`) | Python-authored Figma scripts the model never writes: tokens, text styles, placeholder sections (section 6a). |
| **Critic** (`agent/critic.py`) | The visual gate: deterministic geometry analysis, plus optional screenshot critique (section 8). |
| **Tools** (`tools/`) | The functions the model may call: `execute_figma_js`, `get_metadata`, `get_screenshot`. Keep this set small. `query_docs` is deliberately **not** offered — the gotchas are inlined in the system prompt instead (section 9). |
| **Knowledge** (`knowledge/`) | Plugin API gotchas + typings; retrieves the relevant slice per step (section 9). |
| **Bridge** (`bridge/`) | WebSocket server + message protocol. Sends JS to the plugin, matches responses by id, tracks which file is connected. |
| **State** (`agent/state.py`) | Holds the plan, created node IDs, tokens, per-step results. Feeds the model concise summaries, never full history. |
| **Reference** (`agent/reference.py`) | Attachments -> text: a screenshot read by a vision model, a spec read as-is. One conversion at the front, so nothing downstream needs to know about images (section 21). |
| **Editor** (`agent/editor.py`) | The mirror of the renderer, for changing nodes that already exist: a declarative edit list compiled into correct Plugin API calls (section 20). |
| **Inventory** (`agent/inventory.py`) | The canvas as an addressable index, so an edit can name a real node instead of guessing a selector (section 20). |
| **Edit loop** (`agent/edit_loop.py`) | The edit pipeline: read the canvas -> adopt its styles -> plan changes -> apply and verify one at a time. |
| **Web dashboard** (`web/`) | Optional browser UI: file gallery, credentials, setup guide, live run log. No agent logic. |

The **Figma plugin** (`figma_plugin/`) is the only non-Python part. Keep it thin: execute
received JS, return the result + a screenshot.

---

## 4. Directory structure

```
figma-agent/
|-- CLAUDE.md                # this file
|-- README.md                # human quickstart
|-- .env.example             # config template (copy to .env — or use the dashboard)
|-- requirements.txt
|-- config.py                # loads .env into a typed Settings object
|-- main.py                  # CLI:      python main.py "your instruction"
|-- webapp.py                # dashboard: python webapp.py  -> localhost:8787
|
|-- agent/
|   |-- llm.py               # ModelClient — THE model swap point (section 5)
|   |-- loop.py              # the orchestration loop, CREATE mode (section 6)
|   |-- edit_loop.py         # the orchestration loop, EDIT mode (section 20)
|   |-- editor.py            # declarative edits -> Figma JS (section 20)
|   |-- inventory.py         # the existing canvas, as addressable ids (section 20)
|   |-- reference.py         # attachments -> reference text (section 21)
|   |-- planner.py           # instruction -> design brief -> ordered plan
|   |-- scaffold.py          # Python-authored Figma scripts (section 6a)
|   |-- critic.py            # the visual gate + design-system checks (section 8)
|   |-- requirements.py      # did the design contain what was ASKED for? (section 8d)
|   |-- metrics.py           # what a run cost: calls, round trips, retries (section 19)
|   |-- prompts.py           # system prompt + templates + few-shot exemplars
|   |-- state.py             # RunState dataclass
|
|-- tools/
|   |-- registry.py          # tool JSON schemas + dispatch
|   |-- bridge_io.py         # the one timed, measured path to the plugin (section 19)
|   |-- figma_exec.py        # execute_figma_js
|   |-- figma_read.py        # get_metadata, get_screenshot
|   |-- docs.py              # query_docs (calls knowledge/)
|
|-- bridge/
|   |-- server.py            # asyncio websocket server, id-based req/resp matching
|   |-- protocol.py          # Request/Response dataclasses (the wire contract)
|
|-- knowledge/
|   |-- gotchas.md           # the Plugin API traps (section 11)
|   |-- api_types.d.ts       # Figma Plugin API typings, for retrieval grounding
|   |-- index.py             # chunk + retrieve (section 9)
|
|-- web/                     # the dashboard (webapp.py only; no agent logic)
|   |-- app.py               # stdlib HTTP server + JSON API + run orchestration
|   |-- registry.py          # local history of Figma files seen, with thumbnails
|   |-- settings_store.py    # UI-entered credentials, layered over .env
|   |-- static/index.html    # single-page UI: gallery, settings, setup guide, themes
|
|-- figma_plugin/            # TypeScript/JS — the only non-Python part
|   |-- manifest.json
|   |-- code.ts / code.js    # runs in Figma: eval JS, call Figma API, screenshot
|   |-- ui.html              # holds the WebSocket; relays to code.js via postMessage
|
|-- bench/                   # the design-quality benchmark (section 16a)
|   |-- spec.py              # Task + Criterion dataclasses; tasks are DATA
|   |-- tasks/*.json         # frozen instructions + acceptance criteria
|   |-- capture.py           # one-round-trip read of the finished design
|   |-- score.py             # the deterministic scorer
|   |-- run.py               # CLI: run tasks, save results, re-score offline
|   |-- results/             # git-ignored; one file per run, never overwritten
|
|-- tests/                   # no Figma, no network, no model — all fakes
    |-- test_bridge.py       # protocol round-trips + handshake + disconnects
    |-- test_edit_loop.py    # edit mode end to end (section 20)
    |-- test_editor.py       # the edit compiler, weighted to what it REFUSES
    |-- test_inventory.py    # the canvas index, listing and id resolution
    |-- test_reference.py    # attachments: decoding, refusing, describing (section 21)
    |-- test_loop.py         # loop logic with a fake ModelClient + fake bridge
    |-- test_critic.py      # the visual gate's geometry, contrast + design checks
    |-- test_requirements.py# requirement coverage, and its false-positive rules
    |-- test_metrics.py     # the run recorder: cost, latency, failure reasons
    |-- test_scaffold.py    # palette parsing + generated JS actually compiles
    |-- test_llm.py          # tool-call recovery for small models
    |-- test_settings.py     # settings precedence, masking, dashboard API
    |-- test_registry.py     # file-gallery history
    |-- test_docs.py         # retrieval never answers with silence
    |-- fixtures/
```

Rule: **one concern per file.** If a file starts doing two jobs, split it.

---

## 5. The model and the swap point

The one hard requirement is **tool calling**. Everything else is a trade-off. All model
access goes through **one file**, `agent/llm.py`, which speaks the OpenAI-compatible chat
API — so switching provider is a `.env` change, not a code change.

### The three options, with real measured numbers

| Option | Setup | Speed | Notes |
|---|---|---|---|
| **Ollama Cloud** (current) | `ollama signin`, `ollama pull gpt-oss:20b-cloud` | ~5s/call | Free "Low Usage" tier, native tool calls, **no vision**. Runs on Ollama's infra via the local endpoint. |
| **Hosted API** (OpenRouter, Anthropic, …) | API key | seconds | Best quality; costs money. Watch the account balance vs. the model's default `max_tokens` — a 402 here looks like a config bug but isn't. |
| **Fully local** (`qwen2.5-coder:7b`) | `ollama pull` | **minutes**/call | Free and offline, but without a GPU a full run takes a very long time. Also emits tool calls as plain text — see below. |

```
# Current .env
MODEL_BASE_URL=http://localhost:11434/v1
MODEL_API_KEY=ollama
MODEL_NAME=gpt-oss:20b-cloud
```

Not every `:cloud` model is free — `glm-5.2:cloud` returns 403 "requires a subscription".
**Verify with a real request** (the dashboard's "Test connection" button does exactly this)
rather than assuming from the name.

### Interface

```python
# agent/llm.py — the ONLY place a model provider is defined.
class ModelClient:
    """One method the whole system depends on. Swap the body, keep the shape."""
    def complete(self, messages: list[dict], tools: list[dict]):
        """Return the assistant message (text and/or tool calls). If the model is
        multimodal it can also SEE images passed in messages — section 8."""
        ...
```

### Tool-call recovery (why `llm.py` is bigger than one method)

Small models frequently emit a *correct* tool call as plain JSON text in `content` instead
of populating the API's `tool_calls` field — `qwen2.5-coder:7b` does this on every call, and
`tool_choice: "required"` does not fix it. `agent/llm.py` normalizes three real shapes seen
in live runs back into proper tool calls:

- a bare `{"name": ..., "arguments": {...}}`
- an invented `{"calls": [ ... ]}` wrapper when batching
- several ```` ```json ```` blocks buried in prose

The loop never knows the difference. Keep this in `llm.py` — it is provider quirk handling,
which is exactly what the swap point is for.

### Cost discipline

Debug the **harness** (does the loop close, do tools fire, do the gates block) against the
fakes in `tests/` — 367 tests run with no network, no Figma and no model. Spend real model
calls on genuine design runs.

---

## 6. The agent loop (the workflow)

```python
# agent/loop.py — this mirrors the real code
def run(instruction, bridge, llm, max_retries, max_steps) -> RunResult:
    state = RunState(instruction)

    inspect_file(state, bridge)            # 1. READ-ONLY: existing nodes + geometry,
                                           #    and discover REAL Inter style strings.
                                           #    Never assume canvas state.

    state.enhanced_brief = planner.enhance_instruction(instruction, llm)   # 2. design brief
    create_root_frame(state, bridge)       # 3. HARNESS-AUTHORED (section 6a)
    bootstrap_tokens(state, bridge)        # 4. HARNESS-AUTHORED (section 6a)

    state.plan = planner.make_plan(state.enhanced_brief, state, llm)[:max_steps]

    for step in state.plan:                # 5. one step at a time
        run_step(...)                      #    retries + both gates, below

    final_validation(state, bridge)        # 6. screenshot + layout review + binding audit
    return state.result()


def run_step(step, state, bridge, llm, max_retries, index, total):
    docs = query_docs(step, sources=STEP_DOC_SOURCES)   # TYPINGS only (s.9)
    landed_ids, prior_defects, prior_error = [], [], ""
    seen_calls = set()                    # spans every attempt, not just one

    for attempt in range(max_retries):
        # prior_node_ids turns this into a REPAIR, not a rebuild
        outcome = converse_step(step, docs, state, bridge, llm, label, index,
                                prior_node_ids=landed_ids,
                                prior_defects=prior_defects,
                                prior_error=prior_error,
                                seen_calls=seen_calls)
        _remember(landed_ids, outcome.created_node_ids)      # whatever hit the canvas

        if outcome.ok:
            # Judged on THIS step's nodes only -- section 8.
            defects = visual_gate(step, state, bridge, llm, label,
                                  outcome.created_node_ids)
            if not defects:
                state.record_step_result(step, outcome)      # passed both gates
                state.record_section(outcome.section_name)   # later steps must know
                return
            state.add_node_ids(outcome.created_node_ids)
            prior_defects, prior_error = defects, ""         # next attempt REPAIRS
            continue

        # A script threw. Scripts are atomic, but earlier scripts in the same
        # attempt did land -- so repair from those rather than rebuilding.
        state.add_node_ids(outcome.created_node_ids)
        docs = augment_with_error(docs, outcome.summary)     # exact error + hint
        prior_defects, prior_error = [], outcome.summary
    state.mark_failed(step)
    fallback_for_step(step, state, bridge)                   # placeholder (section 6a)
```

**The single most important property of this loop:** a retry never re-issues the
original step description as its headline instruction. Doing so produced a second
copy of the section every time — the model was being told to build the thing it had
just built. A retry that has anything on the canvas is framed as *"these nodes exist,
fix them in place, do not append"*.

Non-negotiable behaviors: inspect before creating; small scripts that `return` their node
IDs; on error read the message and fix (never blind-retry); feed the model concise state, not
the full transcript.

### Guardrails inside the step loop

All of these exist because a live run burned a whole step budget without them:

| Guardrail | Why |
|---|---|
| `MAX_TOOL_TURNS_PER_STEP = 8` | A confused model can otherwise loop forever. |
| Repeated-call guard | Identical calls are refused — one step ran the same `createPaintStyle` script 8×; another created 7 duplicate footer frames. **The guard is owned by `run_step`, so it spans every attempt**; when it lived in `converse_step` it reset with each retry's fresh conversation, which is how those 7 footers happened. |
| Repair-mode retries | A retry that has nodes on the canvas is told to modify them, never to rebuild. See the note under the loop above. |
| `MAX_RENDERS_PER_STEP = 3` | A second `render_ui` in one step REPLACES the first rather than being refused — a model that says "the previous render was missing the card background" is self-correcting, and refusing it meant the fix it had already written could never run. Bounded, because an endless rebuild is the other failure mode. |
| `MAX_REFUSALS_PER_STEP = 3` | A refusal only teaches the model something if it then does something else. One live step was refused the same `get_metadata` five times and re-issued it every turn until the budget ran out. |
| Ran a script, never said "done" | Counts as a **completed** step, not a failure. Failing it would delete a finished section over a missing sign-off. |
| ~~`MAX_DOC_QUERIES_PER_STEP`~~ | Gone. It existed to stop steps burning their turns on `query_docs`; removing the tool removed the problem (section 9). |
| "You replied with text" | If a step ends with no script ever executed, it is a FAILURE, not a success. |
| `ERROR_HINTS` | Recurring errors map to their exact fix, fed back on retry (section 7). |

---

## 6a. What the harness does itself (do not delegate these)

Everything here was moved out of the model's hands **after watching it fail repeatedly**.
These are Python-authored scripts in `agent/scaffold.py` and `agent/loop.py`. They are
deterministic, unit-tested, and compiled as JavaScript in CI.

| Harness does it | Why the model couldn't |
|---|---|
| **Screen frames** (`create_screens`) | The model tried `layoutSizingHorizontal = 'FILL'` on a child of the PAGE — never legal — and failed 3× in a row. Every "append into root frame" step then had no parent, so 7 of 10 steps failed. It also has no way to place several screens without overlapping them; positions are arithmetic, so Python owns them. |
| **Reuse of an existing screen** | Re-running must *continue* a design, not stamp a second copy on top. Screens are matched to existing frames **by name**; a single-screen run additionally falls back to the old shape heuristic (an auto-layout frame, preferring one we made, then matching width). Anything new is placed clear of existing content. |
| **Design tokens** (`bootstrap_tokens`) | Token steps failed more than any other: `createVariableSet`, `figma.createStyle`, collection-id-vs-object, invented mode ids. The palette is spelled out in the request, so we parse the hex codes and write the API calls ourselves. **The USER'S instruction is read first, the brief only fills the gaps** — a run that read only the brief turned a nine-colour instruction into three tokens, because the brief had scattered the same colours into table cells (`1 px solid #E5E7EB`) that no `name: #hex` pattern can see. |
| **Text styles** | Same story, plus font-style guessing. A fixed Inter type scale is created instead. |
| **Layout auditing** (`agent/critic.py`) | Overlap and overflow are geometry, not judgement — section 7 says do arithmetic in Python. |
| **Placeholder recovery** | When a section step exhausts its retries, a labelled `TODO — <section>` frame keeps the page's structure and makes the gap visible, instead of leaving a hole. Tokens/components get no placeholder: a fake component is worse than none. |
| **Surviving a failed variable** (`build_token_script`) | A run created ONE of six colour tokens, so nothing had a readable text or background colour to bind to and the result was 1.0:1 invisible copy. The variable and the paint style now fail independently: a style that is not variable-backed is worth far more than no style at all. |
| **Colour roles + contrast** (`describe_palette`, `readable_pairings`) | Token *names* come from the brief, so they can be `deep-navy`, `cta`, `soft-cream` — nothing tells the model which is a background and which is a foreground. It bound text to whatever sounded right and produced invisible copy that passed every gate, because geometry cannot see contrast. Roles are now derived from WCAG luminance and the legal fg/bg pairs are *measured* and handed over as fact. |
| **The Plugin API itself** (`agent/renderer.py`) | The decisive one. A real 29-step run called `execute_figma_js` for EVERY step and lost most of them to `FILL can only be set on children of auto-layout frames`, `HUG can only be set on…`, `Reparenting would create a component inside a component`, `counterAxisAlignItems … received 'END'` and `findAll callback crashed`. None of those is a design mistake; all are mistakes about an API that never changes. A section step is now given `render_ui` and NOT `execute_figma_js`, so they are unreachable. Telling the model to prefer the renderer did not work — removing the alternative did. |
| **Refusing component steps** (`planner._drop_component_steps`) | The same run planned five "Create a Button component" steps across five screens. Each either failed outright or left a loose component on the canvas that no section ever used. Components add nothing to a static mockup: consistency comes from the styles the harness already made and the renderer already applies. |
| **Palette roles** (`assign_roles`, `complete_palette`) | Pure luminance ordering scrambled a nine-colour brief: `#E5E7EB` was ranked a *surface*, `#111827` became *text-muted*, and the near-black page fill was labelled *text*. A token whose NAME declares its job (`Border: #E5E7EB`) now wins over the arithmetic, two backgrounds are disambiguated by which one the text is readable on, and any role the brief left unfilled is MIXED from the two colours we do have rather than aliased onto the page fill. |
| **Removing work that failed** (`discard_nodes`) | Every build is additive, so the attempt that ENDS a step left its output on the canvas with nothing to remove it. A run shipped a 1440x900 white void, a `TODO` band marking the gap that void already filled, and four stacked copies of one sign-in form. |
| **Refusing to split a screen sideways** (`_collapse_side_by_side`) | A screen frame is a VERTICAL auto-layout, so "add the left panel" then "add the right panel" appends two full-width bands and the form lands UNDER the artwork. Both bands are individually well-formed, so no gate can see it. The planning prompt says this; the prompt was ignored. |
| **Field labels** (`_without_duplicate_labels`) | `input` renders its own label, so a spec that also writes the label out produced "Email" stacked on "Email". The vision critic caught it correctly — but judgement cannot fail a step, so a whole repair budget went on it and nothing was fixed. The renderer owns field layout, so the renderer resolves the collision. |
| **Tracking what's been built** (`record_section`) | `existing_sections` was filled in once, only when reusing a root frame — so on a fresh run every step was told the page was empty and duly rebuilt what was already there. Each completed section (and each TODO placeholder) is now recorded as it lands. |

Consequence for the planner: it is told the root frame and all styles **already exist** and
must not plan steps for them. Plans went from 30+ trivial steps to 6–12 meaningful ones.

---

## 7. Making the loop effective (performance)

These are the levers that move UI quality and keep the run affordable. Apply all of them.

- **Keep each step small.** One logical operation = a small target the model rarely misses,
  and an atomic unit that costs nothing to retry.
- **Gate before advancing.** A step is not "done" until it passes the structural check and
  (if visual) the visual gate. Never let a broken step compound into the next. ✅ implemented
- **Feed errors back verbatim, with a targeted fix.** `augment_with_error` includes the exact
  error text, plus a specific correction from `ERROR_HINTS` for errors the model has proven it
  cannot self-correct from (FILL ordering, font style strings, `lineHeight` shape, optional
  chaining, `createComponentSet`). ✅ implemented
- **Use few-shot exemplars.** Three known-good scripts are in every step prompt. ✅ implemented
- **Do arithmetic in Python, not the model.** Positioning math, palette conversion, root
  geometry, and all overlap/overflow analysis are computed in code; the model makes
  *decisions*, not calculations. ✅ implemented
- **Batch reads.** The visual gate reads the whole subtree's geometry in ONE round trip
  rather than per-node. ✅ implemented
- **Concise state feedback.** Pass node IDs and short summaries; never replay the transcript.
  Context is the scarcest resource even on a frontier model. ✅ implemented
- **Escalate by decomposing, not repeating.** After N identical failures, re-plan into smaller
  steps rather than retrying. Partially done: identical repeated calls are refused and a
  failed section falls back to a placeholder, but there is no automatic re-planning yet.
- **Screenshot at checkpoints, not every micro-step.** The gate runs only on steps that put
  something on the canvas — never for token or component-definition steps. ✅ implemented
- **Cache the stable prefix.** On a paid hosted model, the system prompt + gotchas + exemplars
  are a fixed prefix worth caching. ✗ not implemented (no benefit on the current free tier).

---

## 8. The visual gate (`agent/critic.py`)

This is what makes designs actually *look* right. It has **four parts that catch different
bugs**, and only the last one needs a model at all.

The dividing line throughout is **fact vs judgement**. A 0x0 node renders nothing and a
1.2:1 contrast ratio is invisible — those are arithmetic, and they gate. "This gap is 17px
instead of 16px" is a polish note — it is reported and never fails a step. Getting this
backwards is how a critic starts replacing working sections with `TODO` placeholders.

### 8a. Deterministic geometry analysis (always on, free)

`find_layout_defects(tree)` reads the real node tree — position, size, visibility, text and
font size, `critic.MAX_TREE_DEPTH` (8) levels deep — and computes what is visibly wrong. No model, no tokens, and
it cannot hallucinate a defect:

| Defect | What it catches |
|---|---|
| `collapsed` | A node with no area. The classic Figma trap: a TEXT node collapses to ~0px and silently vanishes from the render. |
| `collapsed-text` | Text under 8px wide — needs `textAutoResize` and an explicit width. |
| `clipped-text` | Node height smaller than its own font size, so glyphs are cut off. |
| `overflow` | A child extending outside its parent's bounds. |
| `duplicate-section` | A section that rebuilds what a sibling already has. A dashboard run built its sidebar twice — once alone, then again inside the shell — and both copies were individually well-formed, so only their identical text gives them away. |
| `overlap` | Two siblings overlapping by more than 2px — **only** checked when the parent is not auto-layout, since auto-layout cannot produce overlap (avoids false positives). |
| `empty-frame` | A frame over 40×40 with no children **and nothing painted in it** — or painted the same colour as what is behind it. The fill test matters: the renderer's own `box` (chart areas, image placeholders, the glowing shapes in a hero) is a deliberately childless FILLED block, so without it the harness failed its own output. Frames under 40×40 are icon placeholders and ignored. |
| `invisible` / `empty-text` | Hidden nodes and text with no characters. |
| `contrast` | Text under ~3:1 against its resolved background — **invisible copy**. See 8b. |

The 2px tolerances exist so rounding never produces noise. This is section 7's "do
arithmetic in Python" applied to layout: overlap is geometry, not judgement.

### 8b. Contrast (always on, free) — the defect geometry is blind to

Section 6a explains that colour *roles* are derived from WCAG luminance so the builder is
handed the legal foreground/background pairs as fact. Nothing checked the result. A model
that ignored the list produced text that was the right size, in the right place, overlapping
nothing — and completely unreadable. Every gate passed it.

The layout read now returns each node's resolved fill and whether it is token-backed, so the
same arithmetic runs over what actually landed:

- The background is the nearest **filled ancestor**, not the direct parent — a transparent
  wrapper between the text and the fill must not hide the defect.
- Large text (>=24px, or >=18.66px bold) is held to WCAG's lower 3:1 bar, not 4.5:1.
- **Only the unreadable band blocks.** Below AA but above ~3:1 is legible-but-not-accessible:
  reported as advisory. Failing a section over 4.4:1 — and eventually demoting it to a
  placeholder — would be a worse design outcome than shipping it.
- Silent when either colour is unresolvable. Text over an image or a gradient has no single
  background colour, and inventing one produces confident nonsense.

### 8c. Design-system adherence (always on, free, never blocking)

The rules CLAUDE.md states as prose, measured against the finished tree. All advisory, all
reported in `RunResult.design_notes`:

| Check | What it catches |
|---|---|
| `off-scale-spacing` | An auto-layout gap or padding off the 8px scale — what makes a page feel arrhythmic even when nothing overlaps. |
| `off-ramp-type` | A `fontSize` outside the ramp `bootstrap_tokens` created: a text style was set aside for an ad-hoc size. |
| `untokenised-fill` | Golden rule 5. A hardcoded colour matching no token, so changing the palette will not change it. Runs *after* `audit_variable_bindings` has rebound everything close to a token, so what survives is genuinely off-palette. Nodes under 24x24 are exempt — a one-off icon colour is normal. |

The scale and the ramp are defined once, in `critic.py`, and imported by `bench/score.py`
so the benchmark cannot drift from the gate that runs during a build.

### 8d. Requirement coverage (`agent/requirements.py`)

Every check above measures *how* something was built. None asked whether the sign-in screen
has a password field. Two rules keep this honest:

1. **Requirements come from the USER'S instruction, never the brief.** The brief is written
   by the model, so triggering on it would let the agent set its own homework and mark it
   complete.
2. **Only what was literally named is asserted.** Nothing is assumed about what a "landing
   page" ought to contain, so a reported miss is always something asked for and absent.

Quoted copy counts too: if the instruction says `'Welcome back'`, that text must be on the
canvas. Like the benchmark's `requirements` dimension this is a **proxy** — it checks that a
node named or reading "password" exists, which is evidence, not proof.

`success` now fails a run that satisfied **none** of a clearly-specified instruction (three
or more requirements, zero met). One missing item is a flaw, not a failure — only
zero-of-many is the wrong design. This closes the gap where a run could match nothing the
user asked for and still report a green tick.

### 8e. Screenshot critique (a SEPARATE vision model)

**The critic is its own model** (`CRITIC_*` in `.env`, built by
`llm.build_critic_client`). The generator needs reliable tool calling and runs ~50 times per
design; the critic needs eyes and runs a handful of times. One model rarely does both well,
and because the critic runs so rarely, a paid hosted vision model costs very little even when
the generator is free. Leave `CRITIC_MODEL_NAME` blank and screenshot critique is skipped
entirely — no images are ever sent, so a text-only endpoint never eats a 400 per step.

Three rules make this safe to switch on. Without them a vision model makes output **worse**:

1. **Structured, severity-tagged defects.** The critic returns JSON
   (`{severity, element, problem}`), and only `blocking` can fail a step. `minor` items are
   recorded in `RunResult.warnings` and never gate. The old parser treated every non-`CLEAN`
   line as blocking — and a vision model always has something to say about a work in
   progress, so it would have failed nearly every step. Anything unparseable, or with an
   unrecognised severity, is treated as minor: **ambiguity must never block.**
2. **Scoped to the section.** The screenshot is of the node the step built, not the whole
   page, matching the geometry gate's scoping.
3. **A visual complaint must never produce a placeholder.** On the final attempt, if only the
   vision critic is unhappy (no geometry defects), the section is **kept** and the notes go to
   `warnings`. Geometry defects are facts — a 0×0 node renders nothing. "It could look better"
   is judgement, and trading a real section for an empty `TODO` frame over judgement is a
   regression, not a gate.

### 8f. How the critique is assembled

When the model can see images, `get_screenshot` is called and the model is sent **both**
signals: metadata is structural ground truth (sizes, counts, hierarchy), the screenshot is
visual ground truth (does it read as a clean UI). They catch different bugs — a field can be
the right size in metadata and still visually overlap its label.

```python
{"role": "user", "content": [
    {"type": "text", "text": "Critique this screen. List concrete visual defects, or reply CLEAN."},
    {"type": "text", "text": f"metadata: {metadata_json}"},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
]}
```

If a configured critic turns out to reject images, that is recorded once and never retried
for the rest of the run.

### Choosing a critic model — probe, never trust the name

`python check_critic.py` sends a solid-colour PNG and checks the model names the **right**
colour (the prompt never says which). That separates three failures the SDK reports almost
identically: an endpoint that rejects images, a model name that is dead, and — the dangerous
one — an endpoint that **accepts the image and answers from the text alone**, where critique
still comes back and every defect in it is invented.

This is not theoretical. Probing this project's own endpoint found `qwen3-vl:235b` and
`gemma3:27b` both **retired** by the provider, and `kimi-k3:cloud` returning 403. Two of
those were recommended in an earlier draft of this file purely from the model's name.

Measured on Ollama Cloud (`gemma4:cloud` is what `.env` uses):

| Model | Vision | Contract | False positives | Speed |
|---|---|---|---|---|
| `gemma4:cloud` | ✅ | valid JSON | **none** — stayed CLEAN on a good section | ~3s |
| `gemma4:31b-cloud` | ✅ | valid JSON | none | ~3s |
| `minimax-m3:cloud` | ✅ | valid JSON | **blocked a well-formed section** | ~14s |

The false-positive column is the one that matters. A critic that fails good sections is worse
than no critic: it burns retries and, before the "never placeholder on a visual complaint"
rule (above), it would have replaced real sections with `TODO` frames. Test any candidate
against a *good* section, not just a broken one.

### 8g. The run FIXES what it finds (`repair_remaining_defects`)

Everything above only ever *reported* at the end: a finished run listed "2
layout issues" and "6 design-system notes" in the dashboard and had done
nothing about any of them, and a step that exhausted its retries left a `TODO`
placeholder that stayed one. All of it is repairable — the section exists, we
know exactly what is wrong with it, and `render_ui` replaces sections — so the
final validation now has another go before reporting.

- Defects are attributed to the **top-level section** that contains them,
  because that is the unit `render_ui` can rebuild.
- A `TODO` placeholder is itself a target. It has no defects of its own (it is
  a tidy little frame), so nothing else would ever come back to it.
- Bounded on both axes (`FINAL_REPAIR_PASSES`, `MAX_FINAL_REPAIRS`) and it
  stops early when a pass changes nothing — an unbounded polish loop is how a
  five-minute build becomes a twenty-minute one.

Two related fixes, both cases of the harness marking its own homework:

- **The placeholder failed the design checks.** It used 14px text (off the
  ramp) and two hardcoded greys, so six of a run's "design system notes" were
  about the harness's own scaffolding. It now uses the token styles and a ramp
  size, and `TODO` frames are exempt from the design review — a gap marker is
  not a design decision.
- **An input with no placeholder rendered an empty text node**, so its box came
  back as "408x56 frame is empty" — a defect the harness caused and reported.
  The renderer now falls back to a hint derived from the label, and refuses an
  empty `text` node outright.

### How it gates

A step that succeeds structurally but fails the visual gate is **not done**. The defect list
becomes the retry's *headline instruction* — "these nodes exist, fix exactly these problems,
do not append" — not a footnote appended to the docs blob. At the end, `final_layout_review`
reports remaining defects in `RunResult.layout_defects` and in the dashboard.

**The gate is scoped to the step's own nodes** (`find_layout_defects(tree, scope_ids=...)`).
Reading the whole root frame meant a defect left by step 2 failed step 7, which then spent
its entire retry budget on a problem it did not cause and could not see — and, before repair
mode, built a duplicate section on each of those attempts. Whole-page analysis now happens
exactly once, in `final_layout_review`. An id that isn't in the tree yields **no** defects;
it must never fall back to judging the whole page.

**Cost control.** The gate only runs on steps that put something on the canvas — never for
token or component-definition steps (section 8's "checkpoints, not every micro-step").

---

## 9. RAG in detail (grounding the model in the real API)

Purpose: replace baked-in Figma knowledge. The model should reason over the *actual* current
Plugin API surface, not its training memory.

**What to embed.**
- `api_types.d.ts` — the Figma Plugin API typings (npm: `@figma/plugin-typings`). Chunk it by
  interface/method: each chunk is one type or method signature plus its doc comment.
- `gotchas.md` — the traps in section 11. Chunk by rule (one trap per chunk).

**Current implementation (`knowledge/index.py`): BM25, no vector store.** `gotchas.md` is
split at each `##` heading, `api_types.d.ts` at each interface/type, and the step description
is scored against those chunks. Zero dependencies. `retrieve(query) -> str` is the whole
interface and every scorer behind it is a swappable backend (`set_backend`), so the promise
that embeddings are a one-file change is now structural rather than aspirational.

The scorer used to count how many query words a chunk contained, which rewards LONG chunks —
and the longest chunk in the corpus is the typings preamble. Measured, on the queries a real
plan produces:

| Query (a plain-English plan step) | Old top hit | Now |
|---|---|---|
| "set the line height on a text node" | the file **header** | `TextNode` |
| "bind a paint to a colour variable" | `RGBA` | `VariablesAPI` |
| "make a child fill its auto layout parent" | — | `AutoLayoutMixin` |

Two changes did it, both dependency-free:

- **BM25** weights rare terms above common ones (IDF) and divides out chunk length. Long
  low-signal chunks can no longer win on volume alone.
- **Identifier-aware tokenization.** `setBoundVariableForPaint` is indexed both whole and as
  set/bound/variable/for/paint, so plain English can reach a compound API name. The planner
  is *told* to keep steps under 20 words and strip out API detail, so this gap was structural.

**Caching**, three layers, because the same text was otherwise re-derived constantly:
the parsed corpus, the term statistics, and the answers (`retrieve` is memoized — a retrying
step asks an identical question up to `max_retries` times). Chunk keywords were a `@property`
that re-ran the tokenizer over the whole corpus on *every scored chunk of every call*.
Measured: **0.31ms -> 0.031ms** per uncached call, and ~0 on a repeat.

**Embeddings are built but off.** `EmbeddingBackend` works and falls back to BM25 with a
logged warning if `sentence-transformers` is absent — it pulls in PyTorch, a heavier
dependency than this entire project, and at ~650 lines of typings BM25 measurably wins. Turn
it on when the corpus grows past a few thousand lines or gains prose that shares no
vocabulary with the queries.

**The gotchas are not retrieved — they are carried.** The whole corpus is ~4k tokens, which
is small enough to simply live in the system prompt (`prompts.system_prompt()`). As a *tool*
it cost two or three round trips per step and needed its own budget guardrail to stop steps
searching until they ran out of turns — and since the entire conversation is resent on every
turn anyway, a "saved" lookup saved nothing. Per-step `retrieve()` is now restricted to
`api_types.d.ts` (`STEP_DOC_SOURCES`), so the same context is never spent twice.

`query_docs` is no longer in `TOOL_SCHEMAS`, but `dispatch` still answers it: a small model
that calls it from memory should get documentation back, not an "unknown tool" error.

**Retrieval must never answer with silence.** An empty result caused the worst failure mode
observed: the model rephrased the same search up to 6 times in a row and burned the step's
entire budget. `query_docs` returns an explicit *"no match — stop searching and attempt a
script; the error will tell you more"*.

**Always in the step prompt, regardless of retrieval:**
- the **original instruction and the design brief**. The planner is told to keep each step
  under 20 words and to strip out colours, fonts and pixel values — so without this the
  builder, which makes every visual decision, was the only stage that never saw the design.
  It is the single largest quality lever in the harness.
- the **plan outline** with the current step marked `>>> THIS STEP`, so a section is built to
  sit between its actual neighbours
- the palette as `name · hex · role`, plus the **measured** WCAG-AA pairings
- the root frame id and the sections already inside it
- the token and text-style names the harness created (section 6a)
- the **real** Inter style strings, read at runtime with `listAvailableFontsAsync()` — a run
  died on `"Inter SemiBold"` because the actual style is `"Semi Bold"` with a space
- three **few-shot exemplars**: section (create → resize → append → *then* sizing), text
  (font-load first, real width), and clone-edit-reassign for fills

**Not yet built:** provider-side prompt caching of the stable prefix. The system prompt is
already a byte-identical ~5k-token prefix per process, so an endpoint that caches prefixes
automatically already benefits; explicit `cache_control` breakpoints need a paid endpoint to
verify against and are not worth writing blind. Token counts are now recorded (section 19),
so the benefit is measurable the moment someone switches.

---

## 10. The bridge (Python <-> Figma contract)

WebSocket server in Python. The Figma plugin connects as a client. Messages are JSON, matched
by `id`.

```python
# bridge/protocol.py — the wire contract. Keep it tiny and stable.
from dataclasses import dataclass
from typing import Any, Literal

@dataclass
class Request:
    id: str
    type: Literal["exec", "screenshot", "metadata", "ping", "hello"]
    code: str | None = None      # for "exec"
    node_id: str | None = None   # for "screenshot"/"metadata"

@dataclass
class Response:
    id: str
    ok: bool
    result: Any = None
    image_base64: str | None = None   # for screenshots -> feeds section 8
    error: str | None = None
```

The plugin has **two parts** (a Figma constraint, not a choice):
- `ui.html` — has network access, holds the WebSocket, relays to the worker via `postMessage`.
  Its bridge URL is editable and saved in `figma.clientStorage`.
- `code.js` — has the Plugin API, `eval`s received JS, captures the return value, takes a
  screenshot, posts the result back to `ui.html`.

Bridge rules: edits work only in Figma's **design editor**, not Dev Mode (read-only) — fail
clearly if the plugin reports Dev Mode. Every request gets exactly one response, matched by
`id`, with a graceful timeout. The bridge is dumb: it moves messages and matches ids, no agent
logic.

### Three things learned from live failures

1. **The `hello` handshake.** On connect, the bridge asks which file the plugin is in and
   gets back `{fileKey, fileName}`. This is what lets the dashboard show a gallery and target
   a specific file. It requires `"enablePrivatePluginApi": true` in the manifest — otherwise
   `figma.fileKey` is always `undefined`. The handshake talks to the socket directly rather
   than through the pending-futures path, because the normal reader has not started yet at
   that point and routing through it deadlocks.
2. **Keepalive pings are disabled** (`ping_interval=None`). Figma throttles a plugin's UI
   iframe when its window is not focused, so it misses the 20s ping deadline; the server then
   killed a healthy connection ("no close frame received or sent"), `ui.html` reconnected 2s
   later, and the cycle repeated forever. This link is loopback-only and every request already
   has its own timeout.
3. **Disconnects are normal, not errors.** `ConnectionClosed` is caught and logged as one
   line; letting it propagate made the websockets library dump a stack trace on every routine
   plugin close.

Use `localhost`, not `127.0.0.1` — Figma's manifest validator rejects raw IP literals in
`networkAccess.allowedDomains` ("must be a valid URL").

---

## 11. Figma Plugin API gotchas (the model must follow these)

Full version lives in `knowledge/gotchas.md` and is always in context. Essentials:

- **Colors are 0-1, not 0-255.** `{r:1,g:0,b:0}` is red. Paint `color` takes `{r,g,b}`;
  opacity is a separate paint field. (Variable *values* use `{r,g,b,a}` — the one exception.)
- **Fills/strokes are read-only arrays.** Clone, modify, reassign — never mutate in place.
- **Load fonts before touching text.** `await figma.loadFontAsync({family, style})` first, and
  **verify the exact style string** with `listAvailableFontsAsync()` — Inter is `"Semi Bold"`
  *with a space*, not `"SemiBold"`. Guessing throws.
- **`resize()` resets sizing modes to FIXED.** Call `resize()` *before* setting HUG/FILL.
- **TEXT nodes ignore FILL by default** and collapse to ~0px wide. For wrapping text: set
  `textAutoResize='HEIGHT'` and an explicit width, then verify `node.width > 0`.
- **HUG/FILL need an auto-layout parent.** Append the child first, *then* set
  `layoutSizingHorizontal/Vertical`. `HUG` is only valid on the auto-layout frame or a TEXT child.
- **`setBoundVariableForPaint` returns a NEW paint** — capture and reassign it.
- **Every script returns its node IDs:** `return { createdNodeIds: [...] }`.
- **Scripts are atomic.** A failed script makes zero changes. Read the error, fix, retry.
- **Never use `figma.notify()`** — it throws. Use `return`.
- **Position top-level nodes away from (0,0).**
- **Switch pages with `await figma.setCurrentPageAsync(page)`** — the sync setter throws.
- **The plugin preloads Inter only.** Load any other font family before use.

Added after real failures (all in `gotchas.md` with worked examples):

- **`create*` is SYNC, `get*Async`/`set*Async` is ASYNC.** There is no
  `createPaintStyleAsync`, `createTextStyleAsync`, `createVariableAsync` or
  `createVariableCollectionAsync` — adding "Async" to a creator throws "not a function".
- **These variable APIs do not exist:** `figma.createVariableSet`, `figma.variableSets`,
  `figma.variables.getVariableByName`, `figma.getLocalVariableByName`, `figma.createVariable`.
  The real set is `createVariableCollection`, `createVariable(name, collectionOBJECT, type)`,
  `getLocalVariablesAsync`, `getLocalVariableCollectionsAsync`, `setBoundVariableForPaint`.
  There is no lookup-by-name — list and filter.
- **There is no "spacing style".** Spacing is a `FLOAT` variable. `figma.createStyle()` does
  not exist.
- **There is no `{type: 'VARIABLE'}` paint.** Start from a SOLID paint, bind it, reassign.
- **Exact enum values:** `textCase` is `'UPPER'` (not `'UPPERCASE'`); `layoutSizing*` is
  `'FIXED'|'HUG'|'FILL'` (not `'AUTO'`); `counterAxisAlignContent` is `'AUTO'|'SPACE_BETWEEN'`;
  `primaryAxisAlignItems` uses `'MIN'`/`'MAX'`, not `'LEFT'`/`'RIGHT'`.
- **`lineHeight`/`letterSpacing` are objects**, not numbers: `{unit:'PIXELS', value:56}`.
- **No optional chaining (`?.`) or nullish coalescing (`??`)** — the sandbox rejects them
  with "unexpected token in expression: '?'".
- **`figma.createComponentSet()` does not exist** — use `figma.combineAsVariants([a,b], parent)`,
  or skip variants entirely for a static mockup.
- **Always `await figma.getNodeByIdAsync(id)`** — the sync getter throws under
  `documentAccess: "dynamic-page"`. Check every call in the script, not just the first.
- **Style ids (`S:...`) are not node ids.** Do not put them in `createdNodeIds`.
- **`vectorPaths` needs `data` and `windingRule`** (not `d`), and only `M L C Q Z` commands.
  For simple icons, prefer an ELLIPSE or rounded RECTANGLE.

Canonical script shape:

```javascript
// load fonts -> create/mutate -> RETURN ids.  Small and atomic.
await figma.loadFontAsync({ family: "Inter", style: "Semi Bold" }); // exact style string
const frame = figma.createFrame();
frame.resize(320, 52);                 // resize BEFORE sizing modes
figma.currentPage.appendChild(frame);
frame.x = 200; frame.y = 200;          // keep off (0,0)
return { createdNodeIds: [frame.id] }; // ALWAYS return ids
```

---

## 12. Coding conventions (keep it easy to understand)

- Type hints on every function; a one-line docstring saying *why*.
- Functions stay small (~40 lines). One job each.
- Explicit over implicit. No metaprogramming, no magic, no deep inheritance.
- Errors are visible: catch, log a clear message, recover or fail loudly. Never swallow.
- Few, boring dependencies. Currently exactly four: `openai`, `websockets`, `python-dotenv`,
  `pytest`. The dashboard uses stdlib `http.server` — **no web framework**. Justify anything
  new; `sentence-transformers` + a vector store only if retrieval genuinely needs it.
- Config comes from `config.py` (a typed `Settings`), not scattered `os.getenv`.
- Naming says intent: `execute_figma_js`, not `run`.
- No secrets in code — only `.env` (git-ignored). `.env.example` shows the shape.

---

## 13. Anti-patterns (do NOT do these)

- Do not add an agent framework. The loop is hand-rolled for clarity and control.
- Do not touch a model provider outside `agent/llm.py`.
- Do not put agent logic in the bridge or the plugin. They only move and execute.
- Do not generate large multi-operation Figma scripts. One logical step per call.
- Do not skip the inspect-first, the structural gate, or the visual gate to "save time."
- Do not hardcode colors/spacing into final nodes. Bind to variables.
- Do not feed the whole transcript back to the model. Summarize; pass node IDs.
- Do not guess font style strings. Discover them at runtime.
- Do not screenshot every micro-step. Checkpoints only (section 8).
- Do not ask the model to do mechanical work the harness can do deterministically (6a).
- Do not hardcode a position for anything created on a re-run — check what exists first, or
  you stamp a second copy on top of the user's work.

---

## 14. Configuration & setup

```bash
# .env — model settings can also be entered in the dashboard instead
MODEL_BASE_URL=http://localhost:11434/v1
MODEL_API_KEY=ollama
MODEL_NAME=gpt-oss:20b-cloud     # must support TOOL CALLING

# Bridge. Use "localhost", not "127.0.0.1" — Figma's manifest validator
# rejects raw IP literals in networkAccess.allowedDomains.
BRIDGE_HOST=localhost
BRIDGE_PORT=9223

# Loop limits (guardrails against runaway runs / spend)
MAX_RETRIES=3
MAX_STEPS=40
```

Three places must agree on host/port: `.env`, `figma_plugin/manifest.json`'s `networkAccess`,
and the plugin's Bridge URL field (editable in its UI, saved in `clientStorage`).

**Credentials without `.env`.** `webapp.py` boots with no model configured and collects it in
the dashboard's Settings panel: base URL, key, model name, a "Test connection" button that
makes one real call, and a label on each field showing whether it came from `.env` or the UI.
Values are saved to `web/runtime_settings.json` (git-ignored) and **take precedence over
`.env`**; the API key is never sent back to the browser (masked). `main.py` still requires
`.env` and fails loudly — it has nowhere to ask.

**Run preferences are declared once**, in `settings_store.PREF_SPECS`, with their type,
bounds and where the default comes from. The dashboard renders its inputs from that
declaration (`/api/settings` -> `prefs_schema`) rather than hardcoding min/max in HTML, so
the limit the UI shows is the limit the store enforces is the limit the run uses. Numeric
defaults come from `.env` via `Settings`, which closes a real divergence: `MAX_RETRIES=5`
used to apply to the CLI and be silently ignored by the dashboard, which ran 3 and displayed
3 from the same file.

Coercion is shared between reads and writes, so a value can only ever round-trip to itself.
Anything rejected comes back to the UI with its reason (`pref_errors`) instead of the input
silently reverting, and the settings file is written via a temp file and renamed, so an
interrupted save cannot leave JSON that reads back as "no settings at all" and loses the
user's API key.

Run:

```bash
pip install -r requirements.txt
cd figma_plugin && npm install && npm run build && cd ..

# Figma Desktop: Plugins -> Development -> Import plugin from manifest ->
#   figma_plugin/manifest.json, then run it inside a DESIGN file (not Dev Mode).

python webapp.py       # dashboard at localhost:8787 — gallery, settings, setup guide
# or
python main.py "a mobile sign-in screen with email, password, and a Google button"
```

**Python does not hot-reload.** After changing any agent code, restart `webapp.py` — several
debugging sessions were lost to a stale process still running the old harness.

### The dashboard (`webapp.py`)

Optional browser UI; the CLI still works unchanged. It adds:

- a **file gallery** built automatically from files opened with the plugin running (name +
  real screenshot + last-seen, stored in git-ignored `web/known_files.json`). No Figma
  account or REST token — it only ever shows files it has actually seen via the handshake.
- **choosing which file to build in.** Picking one that isn't currently open puts the run in
  "waiting for file" and starts it the moment that file connects.
- **removing a file from the gallery** — and, as a separate opt-in, emptying its canvas.
  There is deliberately no "delete the Figma file": a plugin runs INSIDE a file, the Plugin
  API has no `deleteFile` and `figma.fileKey` is read-only (verified against the typings),
  and Figma's REST API has no delete-file endpoint. So the dialog says exactly that, and
  offers the two things that are real. Clearing the canvas is guarded by the same identity
  check as the thumbnail capture — the script always runs in whichever file the plugin is
  in now, so without re-checking, deleting one card could wipe a different design.
- live status pills for plugin/model, the run log, and the final screenshot.
- **dark/light themes** (follows the OS, remembers your choice).
- a **Connect Figma** guide with the exact manifest path to import, step 4 turning green when
  the plugin connects.

---

## 15. Build phases — current status

| Phase | What it is | Status |
|---|---|---|
| **0** — bridge only | Hardcoded script in, screenshot back | ✅ done |
| **1** — model + one tool | `execute_figma_js`, loop closes | ✅ done |
| **2** — observe + recover | metadata, screenshots, retry-on-error, structural gate | ✅ done |
| **3** — visual gate + retrieval | Geometry gate ✅; screenshot critique built but **inactive** (text-only model); keyword retrieval ✅, embeddings ✗ | ◑ partial |
| **4** — planning | brief → plan → tokens/components/composition | ✅ done |
| **5** — discipline + polish | harness-authored tokens ✅, final review ✅, dashboard ✅; prompt caching ✗ | ◑ partial |

Do not start a phase until the previous one runs end to end.

**Known gaps, in rough priority order:**
1. **No baseline recorded yet.** The benchmark exists (section 16a) but has never been run
   against live Figma, so every improvement in this file is still argued from code reading
   rather than measured. Capture a baseline before changing anything else.
2. ~~**No requirement-coverage check.**~~ **Done** — `agent/requirements.py` (section 8d).
   Requirements are derived from the user's own instruction, checked against the finished
   tree, and reported as `requirements_met` / `requirements_missing`. `success` now fails a
   run that satisfied none of a clearly-specified instruction.
3. **The planner emits `list[PlanStep]`** — each step now carries the `screen_index` it
   belongs to, so screen assignment is data rather than a guess. Section detection is still
   keyword-matched English (`_is_section_step`), and per-section acceptance criteria are
   still not emitted; that half of the gap is open.
4. ~~**No deterministic design checks** beyond geometry~~ **Done** — sections 8b and 8c.
   Contrast is now enforced against what landed (unreadable blocks, below-AA is advisory),
   and spacing-scale, type-ramp and token-backing are measured and reported.
5. **No vision critic is configured yet.** The architecture is in place and safe to enable
   (section 8b), but `CRITIC_MODEL_NAME` is blank, so screenshot critique never runs. This is
   now the biggest single quality lever available — it is the only check that can see
   contrast, balance and whether the screen reads as the product that was asked for.
6. Design quality on a 20B model. Real, but it was masked by the plumbing gaps above; retest
   it once the benchmark exists.
7. Embeddings over `api_types.d.ts` are **built but off by default** (section 9): BM25 wins
   at this corpus size and needs no dependency. No provider-side prompt caching yet — the
   prefix is stable and token counts are now recorded, so the benefit is measurable the
   moment someone points this at a paid endpoint.
8. **`bench/` still records no `visual` dimension**, and the benchmark has still never been
   run against live Figma. Gap #1 is unchanged and remains the most important one: every
   improvement above is argued from unit tests and measured micro-benchmarks, not from a
   design-quality baseline.

---

## 16. Testing

**367 tests, no network, no Figma, no model.** `pytest` runs the whole suite in ~7 seconds.

| File | Covers |
|---|---|
| `test_bridge.py` | Protocol round-trips over a real loopback socket, the `hello` handshake, timeouts, abrupt disconnects |
| `test_loop.py` | The loop with `FakeModelClient` (scripted tool calls) + `FakeBridge` (canned results): retries, both gates, root-frame reuse, repeat guard, doc-query budget, placeholder fallback |
| `test_critic.py` | The visual gate: geometry, contrast (including what must NOT be flagged), design-system adherence — and that a clean tree reports **nothing** |
| `test_requirements.py` | Requirement coverage, weighted towards its false-positive rules: a wrongly-reported miss is worse than a missed check |
| `test_metrics.py` | The run recorder: latency distributions, per-thread isolation, and that measuring never breaks what it measures |
| `test_scaffold.py` | Palette parsing, and that generated JS **actually compiles** via `new AsyncFunction(...)`, exactly as the plugin evals it |
| `test_llm.py` | Tool-call recovery for models that emit them as text |
| `test_settings.py` | Settings precedence (UI over `.env`), key masking, dashboard API |
| `test_prompts.py` | That the step prompt actually carries the brief, the plan outline and the repair framing — the information-plumbing regressions |
| `test_registry.py`, `test_docs.py` | File-gallery history; retrieval never answering with silence; the gotchas being carried rather than searched for |
| `test_bench.py` | The scorer (section 16a): that it catches duplicates, placeholders, off-scale spacing and unbound fills — and that an unmeasured dimension is excluded rather than scored zero |

Two rules that keep this suite honest:

- **`FakeBridge` auto-serves harness-authored scripts** (inspect, root frame, tokens, fonts,
  layout read, audit) so each test only scripts the model-driven calls it cares about. Adding
  a new harness script means teaching `_harness_response` about it, not editing every test.
- **Generated JavaScript is compiled in CI.** A syntax slip in `scaffold.py` or `critic.py`
  fails here rather than halfway through a live run.
- **Every stateful path a test touches must be redirected to `tmp_path`.** `DashboardServer`
  falls back to the real `web/history.json` when no `History` is passed, so each test that
  drove a run wrote a fake "a dashboard / Untitled / 5ms" entry into the user's own History
  tab — 50 of them, which buried every real run. Settings and the file registry were already
  redirected; the log was the one that was not. When adding a fake, check what it writes.

Also verify plugin changes: `cd figma_plugin && npm run build && npm run lint` — the ESLint
plugin catches disallowed Figma API usage. Keep Figma-dependent checks as a manual smoke test.

---

## 16a. The design benchmark (`bench/`)

`pytest` proves the harness *works*. It says nothing about whether the designs are any
**good** — which is the only thing that actually matters. The benchmark closes that gap.

```bash
python -m bench.run --list                 # the task set
python -m bench.run login --repeat 3       # build it for real, score it, save it
python -m bench.run --all --repeat 3       # the full sweep
python -m bench.run --rescore bench/results/<file>.json   # no Figma needed
```

**Six frozen tasks**, chosen to stress different shapes: `login` and `signup` (form-heavy,
catches silently dropped requirements), `dashboard` (data-dense, repeated elements),
`product` (non-text nodes), `settings` (two-column — the layout a vertical-only plan
misses), `landing` (longest, tests whether quality survives many steps).

**The instruction strings are FROZEN.** Rewording one invalidates every result recorded
before it, which defeats the point of having a benchmark. Add a new task instead.

### Scoring

| Dimension | Weight | Computed from |
|---|---|---|
| `requirements` | 30% | Share of the task's acceptance criteria found in the node tree |
| `figma_correctness` | 15% | Failed steps, placeholder fallbacks, **duplicate sections** |
| `layout` | 15% | Geometry defects (`critic`) + share of spacing values on the scale |
| `design_system` | 15% | Share of fills that are token-backed rather than ad hoc |
| `typography` | 10% | Type-ramp adherence + share of text that is actually readable |
| `visual` | 15% | A vision judge — **absent until one is configured** |

Two rules that keep the number honest:

1. **An unmeasured dimension is excluded, never scored zero.** The total is renormalised
   over what was actually measured (`measured_weight`, currently 0.85). Scoring `visual` as
   zero would make merely switching on a judge look like a 15-point improvement.
2. **85% is deterministic** — computed from the node tree, no model involved. It cannot
   drift and cannot flatter a run. Always report the deterministic sub-scores separately
   from the judge, so a change that only moved the judge is visible as exactly that.

`requirements` is a **proxy**: it checks that text matching `/password/i` exists somewhere,
which is evidence a password field was built, not proof it was built well. Read it that way.

### Comparing two things

- **Change one variable at a time.** Never model *and* architecture in the same comparison.
- **Run `--repeat 3` minimum.** Single-run variance on a small model is wide enough to
  swallow most real improvements; one run of A beating one run of B tells you nothing.
- **Compare paired per task**, not on the aggregate — a change can help forms and hurt
  dashboards, and the mean hides it.
- Every run saves its full capture, so a scorer change can be re-applied to every result
  already recorded (`--rescore`) without rebuilding anything in Figma.

---

## 17. Quick reference

| I want to... | Look in |
|---|---|
| Change the model / provider | `.env` or the dashboard's Settings (only `agent/llm.py` defines access) |
| Turn on / change the vision critic | `CRITIC_*` in `.env` (section 8b) |
| Set the model that reads attachments | Settings → Vision model, or `VISION_*` in `.env` (section 21) |
| Handle a new provider quirk | `agent/llm.py` (section 5) |
| Change how a run is orchestrated | `agent/loop.py` (create) / `agent/edit_loop.py` (edit) |
| Add an edit operation | `agent/editor.py` — `OPS` + the method, then the schema in `tools/registry.py` (section 20) |
| Change how an edit finds its target | `agent/inventory.py` — `find` / `resolve` (section 20) |
| Change what a screenshot is read for | `agent/prompts.py` — `IMAGE_REFERENCE_PROMPT` (section 21) |
| Accept another attachment type | `agent/reference.py` — `IMAGE_TYPES` / `TEXT_TYPES` |
| Change how a colour gets its ROLE | `agent/scaffold.py` — `assign_roles` / `_ROLE_WORDS` (section 6a) |
| Add a `kind` the spec can use | `agent/renderer.py` — `node()` dispatch + `ALIASES` |
| Change what the harness builds itself | `agent/scaffold.py` (section 6a) |
| Tune the visual gate / add a defect check | `agent/critic.py` (section 8) |
| Add a design-system rule (spacing, type, tokens) | `agent/critic.py` — `_check_design` (section 8c) |
| Change what counts as a satisfied requirement | `agent/requirements.py` — `_ELEMENTS` (section 8d) |
| See what a run cost / add a metric | `agent/metrics.py` (section 19) |
| Switch retrieval to embeddings | `knowledge/index.py` — `set_backend("embeddings")` (section 9) |
| Add or change a dashboard preference | `web/settings_store.py` — `PREF_SPECS` (section 14) |
| Change what every step prompt says | `agent/prompts.py` (system prompt, exemplars, notes) |
| Change retrieval / what docs are injected | `knowledge/index.py` (section 9) |
| Add/adjust a tool the model can call | `tools/registry.py` + the tool file |
| Change the Python<->Figma message shape | `bridge/protocol.py` |
| Fix a Figma API mistake pattern | `knowledge/gotchas.md` (section 11) |
| Map a recurring error to its fix | `ERROR_HINTS` in `agent/loop.py` |
| Change what a "design" decomposes into | `agent/planner.py` (plan) / `enhance_instruction` (brief) |
| Know whether a change actually helped | `python -m bench.run --all --repeat 3` (section 16a) |
| Add a benchmark task or criterion | `bench/tasks/*.json` — data, no code change |
| Change how design quality is scored | `bench/score.py`, then `--rescore` past results |
| Change the dashboard UI | `web/static/index.html` (single file) |
| Change the dashboard API or run wiring | `web/app.py` |
| Change the plugin's behaviour or UI | `figma_plugin/code.ts` + `ui.html` (rebuild with `npm run build`) |

---

## 18. Debugging a bad run

1. **Restart `webapp.py` first.** Python does not hot-reload; a stale process has wasted more
   time here than any real bug.
2. **Read the trace, not the screenshot.** The log names the failing step, the exact Plugin
   API error, and which gate rejected it.
3. **Ask where the failure belongs.** A mechanical, always-identical failure belongs in
   `scaffold.py` (section 6a) or `ERROR_HINTS` — not in another prompt tweak. Three rounds of
   prompt edits failed to stop `FILL can only be set on children of auto-layout frames`;
   moving the root frame into the harness fixed it permanently.
4. **Add a gotcha with a worked example**, then confirm retrieval actually surfaces it for the
   query that failed (`knowledge.index.retrieve("...")`).
5. **Verify every API name against the real typings** before writing guidance:
   `grep <name> figma_plugin/node_modules/@figma/plugin-typings/plugin-api.d.ts`. Several
   confident-looking APIs in earlier drafts simply did not exist.
6. **Write a test from the real trace.** Every guardrail in section 6 exists because a live
   run produced it; the tests encode those exact traces so they cannot regress.

---

## 18a. Stopping a run

A run is minutes of model calls and Figma round trips, and until now the only
way out was killing the process. `POST /api/run/stop` sets an event the loop
checks at every safe point: between steps, between attempts, and between
tool-calling turns.

Stopping is **cooperative**, and the wording matters because the promise has to
be one the code can keep. A model call or a Figma round trip already in flight
cannot be interrupted, so what stopping guarantees is that no NEW work starts —
in practice the run ends within one model call. The UI says "Stopping after the
current step" rather than implying it halts instantly.

Three properties make it safe rather than merely possible:

- **The work is kept.** `Cancelled` unwinds the stack, and the nodes the
  interrupted step had already created are recorded on the way out. A
  half-finished design is still the user's design, and reporting it as having
  built nothing while they can see the nodes is worse than useless.
- **Only the cheap finishing work runs.** The screens are still photographed
  and the layout still reviewed (both bridge-only), but the end-of-run repair
  pass is skipped outright — spending model calls on a run somebody just asked
  to stop is precisely what they asked not to happen.
- **`stopped` is its own outcome**, not a failure. It is reported separately
  from `success`, shown as its own badge, and recorded in history as `stopped`,
  because "you stopped it" and "it broke" are different things to be told.

---

## 18a2. Fixing what the review finds (`repair_remaining_defects`)

Every gate in this project ran *while* the design was being built, and judged
only the section the current step had just made. So a defect could survive
simply because the step that caused it had already passed — and the run ended
by **reporting** it. A real run finished with "2 layout issues", "6
design-system notes" and two `TODO` placeholders listed in the dashboard, and
nothing done about any of them.

All of that is repairable. The section exists, the defect names the node, and
`render_ui` replaces a section rather than editing it. So after the last step:

1. Re-read every screen and attribute each remaining defect to the **top-level
   section** that contains it — that is the unit `render_ui` can rebuild.
2. Rebuild those sections, with the specific problems as the instruction.
3. A `TODO` placeholder counts as a target: it is a section that was never
   built, and it has no defects of its own, so nothing else would revisit it.
4. Re-check, and stop early if a pass changed nothing.

Bounded on three axes, because a polish loop is how a five-minute build becomes
a twenty-minute one: `FINAL_REPAIR_PASSES` (2), `MAX_FINAL_REPAIRS` (8), and a
no-improvement early exit. It is a user preference (`final_repair`), on by
default, because it costs model calls.

---

## 18b. One screen at a time (the dashboard's pager)

A five-frame design photographed as one page renders at a size where nothing on
any screen is legible — the picture you most want after a run was the least
useful thing it produced. `capture_screens` renders each frame separately and
the result view pages through them with ‹ › and dots.

The images are **not** in the status payload. `/api/status` is polled every
1.5s, so five full-page PNGs in it is megabytes a minute for pictures that never
change; the payload carries `screen_names` (615 bytes for five screens) and each
image is fetched once from `/api/screens?i=N`.

---

## 18c. When the plugin goes away

Figma being closed, the file being switched, or the socket timing out is the
common way a run ends — and it used to surface as `Run crashed`, with every node
thrown away. A real run built two complete screens, lost the bridge during its
final repair, and reported nothing.

The nodes are on the user's canvas whether or not the run finished, so `_run`
catches the transport failure and returns what was built. Three rules keep the
report honest:

- **Nothing in the wind-up touches the bridge.** Whatever killed the run has
  usually taken the bridge with it, and a tidy-up that throws again lands back
  where it started.
- **`ended_early` is its own outcome**, like `stopped`. "Figma disconnected" and
  "the design is wrong" are different things to be told, and the dashboard says
  so ("Figma disconnected — this is what was built").
- **It is never a success.** The work is real, but the run never reached its own
  final validation, so nothing about it has been checked.

---

## 19. Observability (`agent/metrics.py`)

Every claim in this file used to be argued from reading the code. That is honest while there
are no numbers, but it means nobody can tell whether a change helped — and the two most
expensive things in a run were the two nobody counted.

`RunMetrics` records, per run:

| Measured | Why it is the number you want |
|---|---|
| Model calls: count, mean, **p50/p95**, errors | ~50 calls per design. A mean hides the one 40s call that made a run feel broken; p95 is what you tune against. |
| `model_transient_retries` | Endpoint blips absorbed by `llm.py`. A run that "felt slow" is often this, not the model. |
| Prompt / completion tokens | Best-effort (not every endpoint reports `usage`). Makes prompt-caching benefit measurable. |
| Figma round trips, **split by request type** | A slow `screenshot` and a slow `exec` have completely different causes. |
| `plugin_wait_seconds` | Time spent waiting for the user to open Figma. Not a failure, but it dominates wall clock and was otherwise blamed on the agent. |
| Steps planned / completed / failed, and `retry_rate` | Attempts per step. `1.0` means nothing needed a second try; it is the single best predictor of a slow, expensive run. |
| `failure_reasons`, bucketed | `no-script-run` (a prompt problem) and `script-error` (belongs in `ERROR_HINTS`) read identically in the log and call for different fixes. |
| `gate_failures` by gate | How often geometry vs vision is the thing rejecting work. |

Three design choices, each load-bearing:

1. **Instrumented at the seams, not the call sites.** Model timing lives in `llm.py` (golden
   rule 1 guarantees every caller goes through it) and Figma timing in `tools/bridge_io.py`.
   The loop, the planner and the critic are measured without any of them knowing.
2. **Thread-local, not a parameter.** The alternative is threading a metrics object through
   `run -> run_step -> converse_step -> dispatch -> the tool functions`. The dashboard
   already runs each run on its own thread, which is exactly the right granularity.
3. **`current()` never returns None.** Outside a run it returns a throwaway, so instrumented
   code is a plain `metrics.current().observe(...)` with no guard — a guard that would
   eventually be forgotten somewhere.

The dashboard passes its own `RunMetrics` into `loop.run` and polls it live, which is how the
UI shows "Step 4 of 9 · attempt 2" instead of asking the user to read a scrolling log. Every
run also logs one summary line, and `RunResult.metrics` is saved with benchmark results.

**A metric must never break the thing it measures.** The status endpoint reads the recorder
from the HTTP thread while the run thread writes it; a torn read returns `None` rather than
500ing, because a skipped progress frame is a far better outcome than a dead dashboard.

---

## 20. Edit mode — changing a design that already exists

Everything above builds a design from nothing. Edit mode is the other half:
connect to a file that already has work in it and say *"make every primary
button use the accent colour"* or *"add a Forgot password link under the
password field"*.

```bash
python main.py --edit "make every primary button use the accent colour"
# or the dashboard's Create / Edit switch above the instruction box
```

### It is a separate pipeline on purpose

Create and edit are the same shape — read, plan, one step at a time, gate — and
merging them would make both worse:

| | Create | Edit |
|---|---|---|
| Ownership | Owns its frame and every id in it | Owns nothing; every target is the user's work |
| A mistake | Adds something ugly beside the good work | **Damages** something that was already right |
| The gate asks | "Is this section well formed?" | "Is this node still well formed, and did the edits actually apply?" |
| Screens | Plans them | Must never create one |

What they share is *imported*, not copied: the model plumbing from `loop`, the
palette roles from `scaffold`/`renderer`, the geometry and contrast checks from
`critic`, and — for `insert` and `replace` — the renderer itself.

### The mode is declared, never inferred

Guessing from the wording means an edit request misread as a build stamps a
second screen beside the design it was supposed to change, and undoing that is
manual work in someone else's file. The dashboard has an explicit switch; the
CLI has `--edit`.

An **empty file is refused** rather than quietly falling back to create mode,
without spending a single model call.

### Targeting: the model names ids, the harness verifies them

The one thing that decides whether an edit lands is whether it points at the
right node, and a model has no way to refer to a node it cannot see. So
`agent/inventory.py` reads the canvas into an addressable index and the prompt
carries it as a listing, **id first on every line** — the id is the only part
that must be copied exactly, and burying it after the name made it likelier to
be paraphrased.

```
PAGE 'Nexora'
  1:2  FRAME  "Login"  1440x900  #FFFFFF
    1:3  FRAME  "Auth Card"  440x520  #FFFFFF
      1:4  TEXT  "Heading"  text='Welcome back'  32px  300x40
      1:9  FRAME  "Button / Log in"  372x48  #6C5CE7
```

- **The user's Figma selection is the answer to "which one?"** The inventory
  read returns `figma.currentPage.selection`; when it is not empty the listing
  is scoped to it (plus its descendants — selecting a card means its contents)
  and the prompt says so.
- **`resolve` checks every id before compiling.** An id the model invented never
  reaches Figma. Figma's own error for a missing node names neither the bad id
  nor the real ones, so a model that gets it simply invents another; the message
  the harness returns names the mistake and points back at the listing.
- **`find` is arithmetic, so Python does it.** `{"name": "Button", "screen":
  "Login"}` is matched, ranked and expanded here. A selector the model cannot
  express is a selector it cannot get wrong — and a selector matching nothing is
  an *error*, never an empty edit, because applying zero changes and reporting
  success is the most confusing possible outcome.
- The reader is `critic.NODE_READER_JS`, the same one the visual gate uses. A
  second reader would drift, and "what the agent sees" would stop matching
  "what is checked".

### `edit_ui` — the only build tool an edit step gets

`render_ui` is **not** offered: building a fresh section is how "make the button
purple" turns into a second copy of the whole screen. Structural changes go
through `insert` and `replace`, which are anchored to a node that already
exists and reuse the renderer to build their subtree.

```json
[{"op": "set_fill", "target": "1:9",  "color": "accent"},
 {"op": "set_text", "target": "1:10", "value": "Sign in"},
 {"op": "insert",   "parent": "1:3", "index": 3, "spec": {"kind": "text", "value": "..."}}]
```

Ops: `set_fill`, `set_text`, `set_text_style`, `set_size`, `set_spacing`,
`set_radius`, `set_visible`, `set_name`, `reorder`, `delete`, `insert`,
`replace`. Deliberately small — a vocabulary the model cannot overreach is worth
more than one that covers every Figma property.

`agent/editor.py` owns the API, exactly as the renderer does for building:

- **`set_text` loads the font the node already uses.** Guessing a style string
  killed a live run ("Inter SemiBold" — the real style has a space). Editing
  never has to guess: the node knows. A `figma.mixed` font is handled rather
  than crashed on.
- **`set_size` on an auto-layout child sets a fixed sizing mode first**, or the
  layout silently undoes the resize and the edit reports success while nothing
  visibly changed.
- **Colours are roles or real tokens; a hex is refused** (golden rule 5).
- **Spacing stays on the 8px scale and text styles stay on the ramp**, by being
  names rather than numbers.

### Three safety properties

1. **Nothing is destructive by default.** `delete` is the only op that removes
   anything, nothing else implies it, and it **refuses to delete a top-level
   screen frame** — far too much damage for one mis-parsed word. Delete the
   section inside it instead.
2. **The batch is not atomic, and says so.** Atomicity is right for building one
   section and wrong for a batch of independent edits: one stale target would
   discard nine good changes. Each edit is wrapped, and the script returns
   `appliedEdits` and `failedEdits` separately — so a retry is told *exactly*
   which edit failed and is told not to re-apply the ones that worked. A step
   that only partly lands is reported as partly applied, because pretending
   nothing happened sends the user looking for changes that are really there.
3. **The gate judges only what the edit touched.** The file is the user's own
   work and may well have had problems before the agent arrived. Failing an edit
   for one of those would have the agent undoing a change it was asked to make
   in order to fix something nobody mentioned.

### It adopts the file's styles rather than imposing its own

`adopt_existing_styles` reads the file's real paint and text styles and derives
the palette roles from those. Creating a fresh set would mean "make the button
purple" introduces a *second*, slightly different purple that no other node
references — precisely the untokenised drift section 8c exists to catch. A file
with no colour styles is a warning, not a silent hardcoded fill.

The canvas is **re-read between steps**: an edit changes what the ids mean — a
`replace` makes the old id dead — so a stale listing would have the next step
targeting a node that is gone.

### What edit mode deliberately does not do

- No vision critic and no end-of-run repair pass. Both judge whether a design is
  *good*; an edit run is not being asked for an opinion about the user's design,
  only to make the change. Unrequested changes are damage.
- No screens, no tokens, no scaffolding. It never adds to a file what the file
  did not ask for.

---

## 21. Saying what you want: attachments and dictation

A design request is not always something you can type. Sometimes you have a
screenshot of the thing you want, or a spec document someone sent you, or your
hands are busy. All three now reach the same place: the text the pipeline
already knows how to build from.

```bash
python main.py --attach reference.png "rebuild this for Nexora AI"
python main.py --attach brand.md --attach hero.png "a landing page to this spec"
```

In the dashboard: a paperclip and a microphone next to the Run button. You can
also **drag a file onto the composer or paste a screenshot straight in**, which
is how a screenshot usually arrives.

### One conversion, at the front, into words

The loop, the planner, the scaffold, the renderer and the critic already work
from a written instruction, and none of them needed to learn about images:

```
screenshot.png ──► [vision model] ──┐
spec.md        ──► [read as text] ──┴──► RunState.references ──► brief, screens, PALETTE
```

The vision prompt (`IMAGE_REFERENCE_PROMPT`) is shaped to feed the stages that
come next, not to be nice prose. It asks for the colours as `Name: #RRGGBB`,
using the exact role names `Background/Surface/Border/Text/Text muted/Accent` —
**because that is the shape `scaffold.extract_palette` reads and
`assign_roles` recognises** (section 6a). Get that wrong and a screenshot's
colours never become tokens; the design would be laid out like the reference
and coloured like nothing at all. It also asks for every visible string quoted
exactly, because real copy is the difference between a rebuild and a wireframe.

### `references` is not the instruction, and that distinction is load-bearing

`RunState.instruction` stays the user's own words. `RunState.references` holds
what came out of the attachments, and `design_source()` is the two together.

| Reads | What it uses | Why |
|---|---|---|
| The brief, the screen plan, the palette | `design_source()` | An attached screenshot describes the design at least as well as the sentence does — and its colours are *facts about a real image*, the best palette source available. |
| `agent/requirements.py` | `instruction` **only** | Section 8d. The vision output is the model's own words; letting it become the instruction would let the agent write its own homework and then mark it complete, quietly removing the one check it does not grade. |

### Nothing is ever silently ignored

The worst possible outcome here is not an error — it is a run that accepted
your screenshot, ignored it, and built something generic while you watched. So:

- **Type and size are checked before the run starts**, in `reference.from_payload`,
  and an unusable attachment is a 400 the user can read rather than a run that
  spends model calls and *then* reports it could not open the file. Limits:
  6 files, 6MB each, 16MB total, text trimmed at 6k characters (the whole
  conversation is resent every turn, so a long spec would crowd out the palette).
- **An image with no vision model configured is refused**, naming the control
  that fixes it. `Settings -> Vision model` really exists — `EDITABLE` in
  `web/settings_store.py` includes `vision_model_name`, and
  `Settings.vision_settings()` falls through `VISION_* -> CRITIC_* -> the main
  model`, so naming a multimodal model is usually the only thing to do.
- **A PDF is refused with the thing to do instead** ("export the page as PNG"),
  rather than being parsed badly. Parsing PDFs needs a dependency this project
  does not want, and a half-read spec is worse than a refused one.
- **A vision model that errors on one image warns and continues.** One bad
  attachment must not take down a run that has five good ones.

### Dictation runs in the browser

The microphone uses the Web Speech API — no audio upload, no transcription
endpoint, no key, no dependency, and nothing added to the stdlib-only server.
It appends to whatever is already typed rather than replacing it, because
dictation is normally used to finish a sentence. Two things worth knowing:

- It stops the moment a run starts. A microphone that stays live because you
  looked away is not something to leave to chance.
- Chrome, Edge and Safari implement it; Firefox does not. An unsupported
  browser is told so, rather than given a button that does nothing. Note that
  the browser's speech service is the vendor's, not local.

Attachments work in **edit mode** too — "make it look like this" is a real
request — and the reference text reaches both the edit planner and every edit
step.
