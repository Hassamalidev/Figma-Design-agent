# Figma Plugin API gotchas

These are the traps that cause most generated-script failures. Each section is
a standalone chunk `knowledge/index.py` can retrieve by keyword match against
the current step, so keep headings and keywords literal and specific.

## A PAGE is a workspace, a FRAME is a screen

This is Figma's structure, and getting it wrong produces designs that overlap.

- A **page** (`figma.currentPage`) is a workspace for a project. You do not
  create one per screen, and you should not create pages at all.
- A **frame** is a screen. Several screens live SIDE BY SIDE as sibling frames
  on the same page.
- A **section inside a screen** (nav bar, hero, sidebar, footer) is a child
  frame of that screen's frame -- not a top-level frame of its own.

```
PAGE: Website Design          <- one workspace
├── FRAME: Login              <- a screen
├── FRAME: Sign Up            <- a screen, beside it
└── FRAME: Dashboard          <- a screen, beside that
      ├── FRAME: Sidebar      <- a SECTION inside the dashboard
      └── FRAME: Content      <- another section
```

The harness has already created one frame per screen and hands you the id of
the one you are building. So:

- **Never create a top-level frame.** Anything you create is appended into the
  frame id you were given, or into a node you created inside it.
- **Never touch another screen's frame.** They are siblings on the same page;
  appending to the wrong one puts a sign-in form inside a dashboard.
- **Never call `figma.createPage()`** or switch pages. Everything belongs on
  the current page.

A loose frame that is never appended to a parent lands at (0, 0) and sits on
top of whatever is already there -- which is how a run ends up with two
designs overlapping.

## Colors are 0-1, not 0-255

`{r: 1, g: 0, b: 0}` is red. `SolidPaint.color` takes only `{r, g, b}` --
opacity is a separate property on the paint object (`paint.opacity`), not
part of the color. The one exception: variable *values* for a COLOR variable
use `{r, g, b, a}` (alpha included).

```javascript
const paint = { type: 'SOLID', color: { r: 0.94, g: 0.98, b: 1 }, opacity: 1 };
```

## Fills and strokes are read-only arrays

`node.fills` and `node.strokes` return frozen arrays. You cannot mutate an
element in place. Clone, modify the clone, then reassign the whole array:

```javascript
const fills = clone(node.fills);
fills[0] = { ...fills[0], color: { r: 1, g: 1, b: 1 } };
node.fills = fills;
```

## Load fonts before touching text; verify the exact style string

`await figma.loadFontAsync({ family, style })` must complete before you set
`characters` or any font property, or it throws. Never guess the style
string -- discover it with `figma.listAvailableFontsAsync()` first. Common
trap: Inter's semibold style is `"Semi Bold"` **with a space**, not
`"SemiBold"` or `"Semibold"`.

```javascript
await figma.loadFontAsync({ family: 'Inter', style: 'Semi Bold' });
const text = figma.createText();
text.fontName = { family: 'Inter', style: 'Semi Bold' };
text.characters = 'Sign in';
```

## resize() resets sizing modes to FIXED

Call `node.resize(w, h)` **before** setting `layoutSizingHorizontal` /
`layoutSizingVertical` to `HUG` or `FILL`. Calling `resize()` after silently
snaps sizing back to `FIXED`, which looks like the sizing call was ignored.

## TEXT nodes ignore FILL by default and can collapse to ~0px wide

A fresh TEXT node hugs on both axes. If you want it to wrap inside a fixed
width, you must explicitly set `textAutoResize = 'HEIGHT'` and give it a
real width; otherwise it can collapse to an unreadable sliver. After
creating a text node, verify `node.width > 0` before moving on.

```javascript
text.textAutoResize = 'HEIGHT';
text.resize(280, text.height);
```

## HUG and FILL need an auto-layout parent (the #1 repeated error)

"FILL can only be set on children of auto-layout frames" and "HUG can only be
set on auto-layout frames or text children" both mean the same thing: you set
sizing too early, or the parent has no `layoutMode`.

Two conditions must BOTH hold before you touch `layoutSizingHorizontal` /
`layoutSizingVertical`:
1. the node is already `appendChild`-ed to its parent, AND
2. that parent has `layoutMode = 'VERTICAL'` or `'HORIZONTAL'` (not `'NONE'`).

Always in this order -- create, size, **parent**, then sizing:

```javascript
const parent = await figma.getNodeByIdAsync(parentId);
// The parent MUST be auto-layout for FILL/HUG to be legal on its children.
if (parent.layoutMode === 'NONE') parent.layoutMode = 'VERTICAL';

const section = figma.createFrame();
section.layoutMode = 'VERTICAL';        // so HUG is legal on itself
section.resize(1440, 200);              // resize BEFORE sizing modes
parent.appendChild(section);            // parent FIRST
section.layoutSizingHorizontal = 'FILL';   // now legal
section.layoutSizingVertical = 'HUG';
return { createdNodeIds: [section.id] };
```

If you don't control the parent, skip FILL/HUG entirely and just `resize()` to
a fixed size -- a fixed-size frame in the right place beats a failed script.

## setBoundVariableForPaint returns a NEW paint object

`figma.variables.setBoundVariableForPaint(paint, field, variable)` does not
mutate `paint` -- it returns a new paint you must put back into the node's
fills/strokes array yourself:

```javascript
let paint = fills[0];
paint = figma.variables.setBoundVariableForPaint(paint, 'color', colorVar);
node.fills = [paint];
```

## Every script returns the node ids it touched

State lives in Python (`agent/state.py`), not in the model's memory. Always
end a script with `return { createdNodeIds: [...] }` (or an empty array for
read-only/inspection scripts). The bridge and the agent loop depend on this
shape to track what exists.

## Scripts are atomic -- read the error, fix, retry

A script that throws makes zero changes to the document (Figma discards a
plugin's tree edits if a synchronous run throws before completion in most
cases, and in all cases the agent loop should treat a thrown error as "no
changes happened"). Never blind-retry the same code. The thrown message
almost always names the exact problem (wrong node type, unloaded font,
read-only property) -- fix that specific thing and resubmit.

## Never call figma.notify()

`figma.notify()` throws in this headless/automated context. Communicate
results only via the script's `return` value.

## Position top-level nodes away from (0,0)

New nodes default to (0,0) and will stack on top of any existing content
there. Give top-level frames an explicit `x`/`y` away from the origin.

## Switch pages with the async setter

`await figma.setCurrentPageAsync(page)` -- the synchronous `figma.currentPage
= page` setter throws under `documentAccess: "dynamic-page"` (the mode this
plugin uses). Always await it before touching `figma.currentPage`.

## Only Inter is preloaded

The plugin preloads Inter at startup. Any other font family must be loaded
with `figma.loadFontAsync` before first use, in the same script that uses it.

## Dev Mode is read-only

Canvas edits only work when the plugin runs inside Figma's design editor. If
the plugin reports it is running in Dev Mode, no `exec` script can mutate
the document -- surface that to the user instead of retrying.

## The script is a BARE async function BODY -- never wrap it in a function

Your `code` is inserted directly as the body of an async function. Write
statements at the top level and `return` at the end. Do **not** declare a
function around it and do not call it yourself.

```javascript
// WRONG -- "function name expected" / "not a function"
async function () { ... }
async function createHeader() { ... }
createHeader();

// RIGHT -- just the statements
const frame = figma.createFrame();
return { createdNodeIds: [frame.id] };
```

You may still `await` freely at the top level, since the body is already
inside an async function.

## CREATORS are sync, GETTERS are async -- never add "Async" to a creator

This trips people up right after they learn the getters are async. There is
no `Async` variant of any create call:

```javascript
// WRONG -- all "not a function"
await figma.createPaintStyleAsync();   await figma.createTextStyleAsync();
await figma.createVariableAsync();     await figma.createVariableCollectionAsync();

// RIGHT -- creators are synchronous, no await
const paintStyle = figma.createPaintStyle();
const textStyle  = figma.createTextStyle();
const frame      = figma.createFrame();
const collection = figma.variables.createVariableCollection('Tokens');

// RIGHT -- only LOOKUPS are async
await figma.getLocalPaintStylesAsync();
await figma.getNodeByIdAsync(id);
await node.setFillStyleIdAsync(style.id);
```

Rule of thumb: `create*` = sync, `get*Async` / `set*Async` = await.

## Listing styles: use the async getters (getLocalPaintStylesAsync)

`figma.getPaintStyles()` does not exist -- calling it throws "not a
function". Under `documentAccess: "dynamic-page"` you must use the async
local-style getters:

```javascript
const styles = await figma.getLocalPaintStylesAsync();
const primary = styles.find(s => s.name === 'Primary');

// Also async, same pattern:
// await figma.getLocalTextStylesAsync()
// await figma.getLocalEffectStylesAsync()
```

## Applying a style: setFillStyleIdAsync, NOT a "STYLE" paint

There is no `{ type: 'STYLE', styleId }` paint. That shape fails paint
validation. To link a node to a paint style, set the style id instead:

```javascript
// WRONG
node.fills = [{ type: 'STYLE', styleId: style.id }];

// RIGHT
await node.setFillStyleIdAsync(style.id);
// (text color works the same way -- setFillStyleIdAsync on the TEXT node)
```

## Effects need blendMode and visible

A DROP_SHADOW effect is rejected unless every required field is present.
`blendMode` and `visible` are the ones most often forgotten:

```javascript
node.effects = [{
  type: 'DROP_SHADOW',
  color: { r: 0, g: 0, b: 0, a: 0.1 },   // RGBA here
  offset: { x: 0, y: 2 },
  radius: 4,
  spread: 0,
  visible: true,        // required
  blendMode: 'NORMAL',  // required
}];
```

## Searching across pages requires loadAllPagesAsync() first

Under `documentAccess: "dynamic-page"`, `figma.root.findOne(...)` /
`findAll(...)` throw ("Cannot call with documentAccess: dynamic-page without
calling figma.loadAllPagesAsync() first"). Either await that first, or --
better -- avoid a document-wide search entirely by tracking the ids you
created.

```javascript
await figma.loadAllPagesAsync();
const page = figma.root.children.find(p => p.name === 'Design');
await figma.setCurrentPageAsync(page);   // async setter, always
```

Prefer `figma.getNodeByIdAsync(id)` with an id you already returned from an
earlier script over searching the whole document.

## Instances come from the component, not from `figma`

```javascript
// WRONG -- figma.createInstance is not a function
const instance = figma.createInstance(component);

// RIGHT
const instance = component.createInstance();
```

## There is no "spacing style" -- use a FLOAT variable

Figma has paint styles, text styles, effect styles and grid styles. There is
no spacing/size style, and `figma.createStyle()` does not exist. Spacing
tokens are **variables** of type `'FLOAT'`:

```javascript
const collection = figma.variables.createVariableCollection('Spacing');
const modeId = collection.modes[0].modeId;          // the default mode
const spacing = figma.variables.createVariable('spacing/md', collection, 'FLOAT');
spacing.setValueForMode(modeId, 24);
return { createdNodeIds: [] };
```

## Variables: pass the COLLECTION OBJECT, and use the real type strings

```javascript
// WRONG
figma.variables.createVariable('x', collection.id, 'FLOAT');  // "Cannot call
//   createVariable with a collection id ... pass the collection node instead"
figma.createVariable(...)              // not a function -- it's figma.variables.*
figma.VariableType.SPACING             // undefined -- no such enum

// RIGHT -- collection OBJECT, type as a plain string
const v = figma.variables.createVariable('x', collection, 'FLOAT');
```

Valid resolved types are exactly `'COLOR'`, `'FLOAT'`, `'STRING'`, `'BOOLEAN'`.
Always take the mode from `collection.modes[0].modeId` -- do not invent a
"default" mode id.

Reuse an existing collection instead of making a new one each time:

```javascript
const existing = await figma.variables.getLocalVariableCollectionsAsync();
const collection = existing.find(c => c.name === 'Spacing')
  || figma.variables.createVariableCollection('Spacing');
```

## Variable APIs that DO NOT EXIST (commonly hallucinated)

None of these are real -- they all throw "not a function" or "cannot read
property of undefined":

```javascript
figma.createVariableSet(...)        figma.variableSets
figma.variables.createVariableSet(...)
figma.variables.getVariableByName(...)
figma.getLocalVariableByName(...)   figma.createVariable(...)
variable.setValue('DEFAULT', ...)   variable.resolvedType = 'COLOR'  // read-only
```

The complete, real set is: `createVariableCollection(name)`,
`createVariable(name, collectionObject, type)`,
`getLocalVariablesAsync()`, `getLocalVariableCollectionsAsync()`,
`getVariableByIdAsync(id)`, and `setBoundVariableForPaint(...)`.

There is no lookup-by-name helper -- list and filter:

```javascript
const all = await figma.variables.getLocalVariablesAsync();
const accent = all.find(v => v.name === 'color/accent');
```

Full working example -- create a collection once, then variables in it:

```javascript
const collections = await figma.variables.getLocalVariableCollectionsAsync();
const collection = collections.find(c => c.name === 'Tokens')
  || figma.variables.createVariableCollection('Tokens');
const modeId = collection.modes[0].modeId;

const accent = figma.variables.createVariable('color/accent', collection, 'COLOR');
accent.setValueForMode(modeId, { r: 0, g: 0.4, b: 0.8, a: 1 });   // RGBA, 0-1

const spaceMd = figma.variables.createVariable('space/md', collection, 'FLOAT');
spaceMd.setValueForMode(modeId, 24);                              // a plain number

return { createdNodeIds: [] };   // variables are not nodes
```

## Binding a variable to a fill: there is no "VARIABLE" paint type

```javascript
// WRONG -- fails paint validation
node.fills = [{ type: 'VARIABLE', variableId: v.id }];

// RIGHT -- start from a SOLID paint, then bind and REASSIGN
let paint = { type: 'SOLID', color: { r: 0, g: 0, b: 0 } };
paint = figma.variables.setBoundVariableForPaint(paint, 'color', accent);
node.fills = [paint];
```

## Style ids are NOT node ids

`createPaintStyle()` / `createTextStyle()` return styles whose ids look like
`S:1a2b3c...`. They are not nodes: `figma.getNodeByIdAsync('S:...')` returns
null. Do not put style ids in `createdNodeIds` and do not try to read them
back with metadata -- return `createdNodeIds: []` from a script that only
creates styles.

## Exact enum values (guessing these fails validation)

- `textCase`: `'ORIGINAL' | 'UPPER' | 'LOWER' | 'TITLE' | 'SMALL_CAPS' |
  'SMALL_CAPS_FORCED'` -- **not** `'UPPERCASE'`.
- `layoutSizingHorizontal` / `layoutSizingVertical`: `'FIXED' | 'HUG' |
  'FILL'` -- **not** `'AUTO'`.
- `counterAxisAlignContent`: `'AUTO' | 'SPACE_BETWEEN'` only.
- `primaryAxisAlignItems`: `'MIN' | 'CENTER' | 'MAX' | 'SPACE_BETWEEN'` --
  **not** `'LEFT'`/`'RIGHT'`/`'TOP'`. Left/top is `'MIN'`, right/bottom is `'MAX'`.
- `counterAxisAlignItems`: `'MIN' | 'CENTER' | 'MAX' | 'BASELINE'`.
- The auto-layout alignment properties are `primaryAxisAlignItems` /
  `counterAxisAlignItems` -- there is no `primaryAxisAlign` / `counterAxisAlign`.

## Always use the async node getter

`figma.getNodeById(id)` throws under `documentAccess: "dynamic-page"`. Use
`await figma.getNodeByIdAsync(id)` -- and check every call in the script, not
just the first one.

## Vector paths need `data` and `windingRule`

```javascript
vector.vectorPaths = [{ windingRule: 'NONZERO', data: 'M 0 0 L 10 0 L 10 10 Z' }];
```

The key is `data`, not `d`, and `windingRule` is required. Only `M`, `L`, `C`,
`Q`, `Z` commands are supported -- SVG shorthand like `s`, `h`, `v`, or arcs
fails with "Invalid command". For simple social/placeholder icons prefer an
ELLIPSE or a rounded RECTANGLE over a hand-written path.

## No optional chaining (`?.`) or nullish coalescing (`??`)

The sandbox that runs these scripts rejects them with
"unexpected token in expression: '?'". Write the check out in full:

```javascript
// WRONG
const modeId = collection?.modes?.[0]?.modeId ?? null;

// RIGHT
const modeId = collection && collection.modes.length ? collection.modes[0].modeId : null;
```

## lineHeight and letterSpacing are objects, not numbers

```javascript
// WRONG -- 'Expected object, received number'
textStyle.lineHeight = 56;

// RIGHT
textStyle.lineHeight = { unit: 'PIXELS', value: 56 };
textStyle.letterSpacing = { unit: 'PIXELS', value: 0.5 };
```

`fontSize` *is* a plain number -- only these two are objects.

## createComponentSet does not exist -- use combineAsVariants

```javascript
// WRONG
const set = figma.createComponentSet();

// RIGHT: build the variants first, then combine them
const a = figma.createComponent(); a.name = 'State=Default';
const b = figma.createComponent(); b.name = 'State=Hover';
const set = figma.combineAsVariants([a, b], figma.currentPage);
```

For a static mockup you usually don't need variants at all -- hover/active
states are not visible in a screenshot. Prefer one plain component.

## Canonical script shape

```javascript
// load fonts -> create/mutate -> RETURN ids. Small and atomic.
await figma.loadFontAsync({ family: 'Inter', style: 'Semi Bold' });
const frame = figma.createFrame();
frame.resize(320, 52);           // resize BEFORE sizing modes
figma.currentPage.appendChild(frame);
frame.x = 200; frame.y = 200;    // keep off (0,0)
return { createdNodeIds: [frame.id] };
```
