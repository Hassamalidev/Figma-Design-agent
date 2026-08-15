# Figma plugin -- the "hands"

This is the non-Python half of the agent (see the root [CLAUDE.md](../CLAUDE.md)
section 1). It has no logic of its own: `ui.html` holds a WebSocket connection
to the Python bridge, and `code.js` executes whatever script it's handed
inside the Plugin API sandbox and reports back the result.

## Develop

```bash
npm install
npm run build     # compiles code.ts -> code.js (npm run watch to keep rebuilding)
npm run lint
```

`code.js` is generated -- always edit `code.ts` and rebuild, never `code.js`
directly.

## Install into Figma Desktop

Plugins → Development → Import plugin from manifest… → select
`figma_plugin/manifest.json`.

## Run

Open a file in the **design editor** (not Dev Mode -- Dev Mode is read-only),
then Plugins → Development → Figma Designer Agent. Leave its window open; it
stays connected to the Python bridge (`ws://localhost:9223` by default) for
the whole run, unlike a typical run-once plugin.

If you change `BRIDGE_HOST`/`BRIDGE_PORT` in the root `.env`, update the
`BRIDGE_URL` constant in `ui.html` and the `networkAccess.allowedDomains` /
`devAllowedDomains` entries in `manifest.json` to match -- all three must
agree.
