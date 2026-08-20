# Figma Designer Agent

Turns a plain-language instruction into a real Figma design. See
[CLAUDE.md](CLAUDE.md) for the full architecture and the reasoning behind it
— this file is just the quickstart.

## How it works, in one sentence

A Python agent plans and generates small Figma Plugin API scripts; a Figma
plugin (running inside Figma Desktop) executes them over a local WebSocket
bridge and reports back node ids, metadata, and screenshots.

## Setup

**1. Python side**

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt
```

Then either copy `.env.example` to `.env` and fill it in, **or** skip that
entirely and run `python webapp.py` — the dashboard boots without credentials
and its **Settings** panel will collect them (with a "Test connection" button
so you find out immediately if the endpoint or key is wrong). Values entered
there are saved to `web/runtime_settings.json` (git-ignored) and take
precedence over `.env`; the panel labels each field with where it came from.
The CLI (`main.py`) still requires `.env`, since it has nowhere to ask.

Edit `.env` with a model endpoint. **It must support tool/function calling** —
that's the one hard requirement, verify it before committing to a provider.
The default host/port in `.env.example` (`localhost:9223`) must match
`figma_plugin/manifest.json`'s `networkAccess` and `figma_plugin/ui.html`'s
`BRIDGE_URL` if you ever change it. Use `localhost`, not `127.0.0.1` — Figma's
manifest validator rejects raw IP literals in `allowedDomains` ("must be a
valid URL").

Three viable options, in order of what we'd actually recommend:

1. **Ollama Cloud** (recommended) — `ollama signin`, then `ollama pull
   gpt-oss:20b-cloud`, point `.env` at `http://localhost:11434/v1` /
   `gpt-oss:20b-cloud`. Free "Low Usage" tier, confirmed tool-calling, ~5-10s
   per call — runs on Ollama's infra, not your machine. Not every `:cloud`
   model is free, though — `glm-5.2:cloud` returned a 403 "requires a
   subscription" when we tried it, so verify with a real request rather than
   assuming from the name.
2. **A hosted API** (OpenRouter, etc.) — fastest/highest quality, but real
   money; watch your account's balance vs. the model's default `max_tokens`.
3. **Fully local** (`qwen2.5-coder:7b` via Ollama, no `:cloud`) — free and
   fully offline, but without a real GPU expect **minutes per call**, not
   seconds — a full run can take a long time. `agent/llm.py` has a fallback
   for local models that emit tool calls as plain-text JSON instead of using
   the API's real `tool_calls` field, which some small local models do.

**2. Figma plugin side**

```bash
cd figma_plugin
npm install
npm run build          # compiles code.ts -> code.js
```

In Figma Desktop: **Plugins → Development → Import plugin from manifest…**,
select `figma_plugin/manifest.json`. Open any design file (not Dev Mode —
Dev Mode is read-only) and run the plugin from that same menu. Leave its
window open; it stays connected for the whole run.

**Check your model can actually drive a run** (optional, ~5 seconds):

```bash
python check_model.py            # the model in .env
python check_model.py --list     # what your endpoint offers
```

Tool calling is the one hard requirement. A model without it produces a run
where every step says "replied with text instead of calling the tool" — which
reads like a bug in the agent and isn't one. Free tiers are **per-model quota
buckets**, so when one rate-limits mid-project, probe another name and switch
`MODEL_NAME`; `.env.example` lists the ones verified on Ollama Cloud, with the
retired and subscription-only ones marked so you don't retry them.

**3. Run it**

CLI:

```bash
python main.py "a mobile sign-in screen with email, password, and a Google button"
```

The CLI starts the bridge and waits (up to 2 minutes) for the plugin to
connect, then runs the agent loop: inspect the canvas → plan → build one
step at a time, retrying on error → final screenshot.

**To change a design that already exists**, use `--edit`:

```bash
python main.py --edit "make every primary button use the accent colour"
python main.py --edit "add a 'Forgot password?' link under the password field"
```

**To build from a screenshot or a spec**, use `--attach` (repeatable):

```bash
python main.py --attach reference.png "rebuild this for Nexora AI"
python main.py --attach brand.md --attach hero.png "a landing page to this spec"
```

An attachment is used twice over: read as words (so the brief, the screens and
the palette come from it), and **placed on the canvas as a real image** — the
logo you attached is the logo, not a grey placeholder. Figma accepts PNG, JPEG
and GIF for that; a WEBP is still read as a reference, and the run says so
rather than quietly dropping it. See CLAUDE.md section 22.

An image needs a multimodal model — name one under **Settings → Vision model**,
or set `VISION_MODEL_NAME` in `.env` (a configured `CRITIC_MODEL_NAME` already
counts). Without one, the run says so rather than quietly ignoring the file and
building something generic. Text files (`.md`, `.json`, `.css`, `.svg`, …) are
read directly and need no vision model. PDFs are refused — export the page as
a PNG.

Edit mode reads the file, works out which nodes the request is about, and
changes those in place — it never builds a new screen. If you **select
something in Figma first**, that is what gets changed; select nothing and the
agent finds the nodes itself. It binds to the file's own colour and text
styles rather than introducing new ones, and the only thing that ever removes
anything is an explicit request to delete. See CLAUDE.md section 20.

Or the dashboard:

```bash
python webapp.py
```

Opens a local web UI at `http://127.0.0.1:8787` (dark/light theme, remembered
per browser) where you type an instruction and pick which Figma file to run it
against, instead of always targeting whatever's currently open. A **Create /
Edit** switch above the instruction box chooses whether the run builds a new
design or changes the one that's already there — declared rather than guessed
from the wording, because an edit request misread as a build stamps a second
screen beside the design it was meant to change.

Next to Run there's a **paperclip** and a **microphone**. Attach a screenshot
(or drag one onto the box, or just paste it) and the agent rebuilds what it
shows; attach a spec and it builds to it. The microphone dictates the
instruction using your browser's own speech recognition — nothing is uploaded
and nothing extra is installed, though it needs Chrome, Edge or Safari. It also has a
built-in **Connect Figma** guide with the exact manifest path to import, and
live plugin/model status in the header. The gallery is built automatically: every file you
open with the plugin running gets a real screenshot saved to a local registry
(`web/known_files.json`, never committed) — there's no Figma account
integration, it only ever shows files it's actually seen. Picking a file
that isn't currently open just waits (with a message in the log panel) until
you open that file in Figma Desktop and start the plugin, then starts the run
automatically.

## It builds what the brief actually says

Two things a brief states that used to be quietly dropped:

- **The frame size.** "Frame size for both: 1440 x 900px" is honoured to the
  end of the run. It used to be applied when the frame was created and then
  overruled by the end-of-run fit pass, which rounded every desktop screen to a
  multiple of 1024 — so a 1440x900 design came back 1440x2048 with a screenful
  of blank canvas under it.
- **The typeface.** A brief that names a display font ("Playfair Display") gets
  it on the headings, with the UI font kept for labels, inputs and buttons.
  Font names are checked against the fonts Figma really has before anything
  uses one; a family this file does not have falls back to Inter and says so.

## The design is clickable

The finished file is wired as a **prototype**, not just drawn: the sign-in
button opens the dashboard, "Back" goes back, a page taller than the screen
scrolls, and every screen has a way to be reached when you press play in Figma.
Most of it is decided by matching labels to screen names in Python; the model
is asked once, and only about the screens nothing obviously opens. Turn it off
with the **Clickable prototype** preference. See CLAUDE.md section 23.

In edit mode, "make this button open the settings screen" is a normal request.

## Project layout

See CLAUDE.md section 4 for the full map. Short version:

| Directory | What's in it |
|---|---|
| `agent/` | The loop, planner, prompts, model client, run state |
| `tools/` | The functions the model can call (`execute_figma_js`, `get_metadata`, `get_screenshot`, `query_docs`) |
| `bridge/` | The WebSocket server + wire protocol (Python ↔ plugin), including the file-identity handshake |
| `knowledge/` | Plugin API gotchas + typings, retrieved into context per step |
| `figma_plugin/` | The plugin itself — the only non-Python part |
| `web/` | The dashboard: local HTTP API (`app.py`), file-history registry (`registry.py`), the single-file UI (`static/index.html`) |
| `tests/` | Fake-model / fake-bridge tests, no Figma or network needed |

## Testing

```bash
pytest
```

`tests/test_bridge.py` exercises the wire protocol over a real loopback
socket (no Figma). `tests/test_loop.py` drives `agent/loop.py` with a
scripted fake model and fake bridge — proves retry and termination logic
without spending API credits or needing Figma open. `tests/test_registry.py`
covers the file-history store.

## Current status

Bridge (with a file-identity handshake), plugin exec/metadata/screenshot,
the model-driven tool-calling loop, bounded per-step retries, a token-
binding audit, a naive keyword-based `knowledge/index.py` retriever, a
model-generated plan, and the web dashboard are all wired end-to-end (Phases
0–4 from CLAUDE.md section 12, plus the dashboard). Not yet built: enforcing
the tokens-before-components-before-composition discipline in code
(currently just prompted), and a real embeddings-backed retriever if the
keyword one proves too coarse.
#   F i g m a - D e s i g n - a g e n t 
 
 