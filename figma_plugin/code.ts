// figma_plugin/code.js -- the "hands". Runs inside Figma's Plugin API sandbox.
// It receives a Request over postMessage (relayed from ui.html's WebSocket),
// executes it, and posts back a Response. It has NO agent logic -- it just
// evals whatever JS it's told and reports the outcome. See CLAUDE.md section 7.

type BridgeRequest = {
  id: string;
  type: 'exec' | 'screenshot' | 'metadata' | 'ping' | 'hello';
  code?: string;
  node_id?: string;
};

// Messages from ui.html that are NOT bridge requests -- they manage the
// plugin's own saved settings. Prefixed so they can never collide with a
// protocol message coming off the WebSocket.
type UiMessage =
  | { type: '__loadConfig' }
  | { type: '__saveConfig'; bridgeUrl: string };

const CONFIG_KEY = 'bridgeUrl';
const DEFAULT_BRIDGE_URL = 'ws://localhost:9223';

type BridgeResponse = {
  id: string;
  ok: boolean;
  result?: unknown;
  image_base64?: string;
  error?: string;
};

// The plugin preloads only these Inter styles at startup. Any other family --
// or any other Inter style -- must be loaded by the script that uses it.
const PRELOADED_STYLES = ['Regular', 'Medium', 'Semi Bold', 'Bold'];

async function preloadFonts(): Promise<void> {
  for (const style of PRELOADED_STYLES) {
    try {
      await figma.loadFontAsync({ family: 'Inter', style });
    } catch {
      // Not fatal: a script that actually needs this style will load it
      // itself and surface a clear error if the style string is wrong.
    }
  }
}

async function resolveNode(nodeId?: string): Promise<BaseNode> {
  if (!nodeId) return figma.currentPage;
  const node = await figma.getNodeByIdAsync(nodeId);
  if (!node) throw new Error(`No node with id ${nodeId}`);
  return node;
}

async function runExec(code: string): Promise<unknown> {
  // The script is the BODY of an async function: it may `await` Plugin API
  // calls and must `return { createdNodeIds: [...] }`. Atomic by construction --
  // if it throws, nothing after the throw ran.
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor as {
    new (body: string): () => Promise<unknown>;
  };
  const fn = new AsyncFunction(code);
  return await fn();
}

function summarizeNode(node: BaseNode): unknown {
  const summary: Record<string, unknown> = { id: node.id, name: node.name, type: (node as SceneNode).type };
  if ('children' in node) {
    summary.children = (node as ChildrenMixin).children.map((child) => ({
      id: child.id,
      name: child.name,
      type: child.type,
    }));
  }
  return summary;
}

async function runMetadata(nodeId?: string): Promise<unknown> {
  const node = await resolveNode(nodeId);
  return summarizeNode(node);
}

async function runScreenshot(nodeId?: string): Promise<string> {
  const node = await resolveNode(nodeId);
  if (!('exportAsync' in node)) {
    throw new Error('This node type cannot be exported as an image.');
  }
  const bytes = await (node as unknown as ExportMixin).exportAsync({ format: 'PNG' });
  return figma.base64Encode(bytes);
}

async function handleRequest(request: BridgeRequest): Promise<BridgeResponse> {
  try {
    if (request.type === 'ping') {
      return { id: request.id, ok: true, result: 'pong' };
    }
    if (request.type === 'hello') {
      // Requires "enablePrivatePluginApi" in manifest.json -- figma.fileKey is
      // otherwise always undefined. Lets the bridge/web UI tell files apart.
      return {
        id: request.id,
        ok: true,
        result: { fileKey: figma.fileKey ?? null, fileName: figma.root.name },
      };
    }
    if (request.type === 'exec') {
      // Edits only work in the design editor. Dev Mode is read-only --
      // fail clearly instead of letting a confusing Plugin API error surface.
      if ((figma.editorType as string) === 'dev') {
        throw new Error('This file is open in Dev Mode, which is read-only. Open it in the design editor.');
      }
      const result = await runExec(request.code || '');
      return { id: request.id, ok: true, result };
    }
    if (request.type === 'metadata') {
      return { id: request.id, ok: true, result: await runMetadata(request.node_id) };
    }
    if (request.type === 'screenshot') {
      return { id: request.id, ok: true, image_base64: await runScreenshot(request.node_id) };
    }
    return { id: request.id, ok: false, error: `Unknown request type: ${request.type}` };
  } catch (error) {
    return { id: request.id, ok: false, error: error instanceof Error ? error.message : String(error) };
  }
}

// Keep the UI open -- this plugin stays running to keep serving requests
// over the bridge, unlike a typical run-once-then-close plugin.
figma.showUI(__html__, { width: 380, height: 460 });

async function handleUiMessage(msg: UiMessage): Promise<boolean> {
  if (msg.type === '__loadConfig') {
    const saved = await figma.clientStorage.getAsync(CONFIG_KEY);
    figma.ui.postMessage({
      __config: { bridgeUrl: saved || DEFAULT_BRIDGE_URL, isDefault: !saved },
    });
    return true;
  }
  if (msg.type === '__saveConfig') {
    const url = (msg.bridgeUrl || '').trim() || DEFAULT_BRIDGE_URL;
    await figma.clientStorage.setAsync(CONFIG_KEY, url);
    figma.ui.postMessage({ __config: { bridgeUrl: url, isDefault: false, saved: true } });
    return true;
  }
  return false;
}

figma.ui.onmessage = async (msg: BridgeRequest | UiMessage) => {
  if (await handleUiMessage(msg as UiMessage)) return;
  const response = await handleRequest(msg as BridgeRequest);
  figma.ui.postMessage(response);
};

preloadFonts();
