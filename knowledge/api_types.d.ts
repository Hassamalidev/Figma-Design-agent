// Trimmed subset of the Figma Plugin API typings, kept here for RAG grounding
// (knowledge/index.py retrieves relevant slices of this file into context).
// Not exhaustive -- covers the surface this agent actually touches: node
// creation, auto layout, text, paints, and variables. For anything missing,
// prefer the official typings (@figma/plugin-typings) as ground truth.

declare global {
  const figma: PluginAPI;
}

interface PluginAPI {
  readonly editorType: 'figma' | 'figjam' | 'slides' | 'dev';
  readonly mode: 'default' | 'inspect' | 'codegen';
  readonly root: DocumentNode;
  currentPage: PageNode;
  readonly viewport: ViewportAPI;
  readonly variables: VariablesAPI;

  setCurrentPageAsync(page: PageNode): Promise<void>;
  loadAllPagesAsync(): Promise<void>;

  createFrame(): FrameNode;
  createRectangle(): RectangleNode;
  createEllipse(): EllipseNode;
  createText(): TextNode;
  createComponent(): ComponentNode;
  createPage(): PageNode;

  loadFontAsync(fontName: FontName): Promise<void>;
  listAvailableFontsAsync(): Promise<Font[]>;

  getNodeByIdAsync(id: string): Promise<BaseNode | null>;

  /** Throws in this harness's automated context -- never call it. Use `return` instead. */
  notify(message: string, options?: NotificationOptions): NotificationHandler;

  closePlugin(message?: string): void;
}

interface FontName {
  family: string;
  style: string; // exact string, e.g. "Regular", "Semi Bold" (note the space)
}

interface Font {
  fontName: FontName;
}

// ---- Nodes -----------------------------------------------------------------

interface BaseNode {
  readonly id: string;
  name: string;
  removed: boolean;
}

interface SceneNodeMixin {
  visible: boolean;
  locked: boolean;
  x: number;
  y: number;
  resize(width: number, height: number): void; // resets sizing modes to FIXED
  resizeWithoutConstraints(width: number, height: number): void;
}

type SceneNode = FrameNode | RectangleNode | EllipseNode | TextNode | ComponentNode | InstanceNode;

interface DocumentNode extends BaseNode {
  children: PageNode[];
}

interface PageNode extends BaseNode {
  children: SceneNode[];
  appendChild(node: SceneNode): void;
  selection: SceneNode[];
}

interface ChildrenMixin {
  children: SceneNode[];
  appendChild(node: SceneNode): void;
  insertChild(index: number, node: SceneNode): void;
}

interface AutoLayoutMixin {
  layoutMode: 'NONE' | 'HORIZONTAL' | 'VERTICAL';
  primaryAxisSizingMode: 'FIXED' | 'AUTO';
  counterAxisSizingMode: 'FIXED' | 'AUTO';
  primaryAxisAlignItems: 'MIN' | 'CENTER' | 'MAX' | 'SPACE_BETWEEN';
  counterAxisAlignItems: 'MIN' | 'CENTER' | 'MAX' | 'BASELINE';
  paddingLeft: number;
  paddingRight: number;
  paddingTop: number;
  paddingBottom: number;
  itemSpacing: number;
  // Only meaningful once a node is a CHILD of an auto-layout frame.
  layoutSizingHorizontal: 'FIXED' | 'HUG' | 'FILL';
  layoutSizingVertical: 'FIXED' | 'HUG' | 'FILL';
}

interface GeometryMixin {
  fills: ReadonlyArray<Paint> | typeof figma.mixed; // read-only: clone, edit, reassign
  strokes: ReadonlyArray<Paint>;
  strokeWeight: number;
  cornerRadius: number | typeof figma.mixed;
}

interface FrameNode extends BaseNode, SceneNodeMixin, ChildrenMixin, AutoLayoutMixin, GeometryMixin {
  type: 'FRAME';
}

interface RectangleNode extends BaseNode, SceneNodeMixin, GeometryMixin {
  type: 'RECTANGLE';
}

interface EllipseNode extends BaseNode, SceneNodeMixin, GeometryMixin {
  type: 'ELLIPSE';
}

interface TextNode extends BaseNode, SceneNodeMixin, GeometryMixin {
  type: 'TEXT';
  characters: string;
  fontName: FontName; // must loadFontAsync() first
  fontSize: number;
  textAutoResize: 'NONE' | 'WIDTH_AND_HEIGHT' | 'HEIGHT' | 'TRUNCATE';
  textAlignHorizontal: 'LEFT' | 'CENTER' | 'RIGHT' | 'JUSTIFIED';
}

interface ComponentNode extends BaseNode, SceneNodeMixin, ChildrenMixin, AutoLayoutMixin, GeometryMixin {
  type: 'COMPONENT';
  createInstance(): InstanceNode;
}

interface InstanceNode extends BaseNode, SceneNodeMixin, ChildrenMixin, GeometryMixin {
  type: 'INSTANCE';
  mainComponent: ComponentNode | null;
}

// ---- Paints ------------------------------------------------------------------

type Paint = SolidPaint | GradientPaint;

interface SolidPaint {
  type: 'SOLID';
  color: RGB; // 0-1 range, NOT 0-255. Opacity is separate.
  opacity?: number;
  boundVariables?: { color?: VariableAlias };
}

interface GradientPaint {
  type: 'GRADIENT_LINEAR' | 'GRADIENT_RADIAL' | 'GRADIENT_ANGULAR' | 'GRADIENT_DIAMOND';
  gradientStops: { position: number; color: RGBA }[];
}

interface RGB {
  r: number; // 0-1
  g: number; // 0-1
  b: number; // 0-1
}

interface RGBA extends RGB {
  a: number; // 0-1 -- variable VALUES use RGBA even where the paint itself uses RGB
}

// ---- Variables -----------------------------------------------------------------

interface VariablesAPI {
  createVariableCollection(name: string): VariableCollection;
  createVariable(
    name: string,
    collection: VariableCollection,
    type: 'COLOR' | 'FLOAT' | 'STRING' | 'BOOLEAN'
  ): Variable;
  setBoundVariableForPaint(paint: Paint, field: 'color', variable: Variable): Paint; // returns a NEW paint
}

interface VariableCollection {
  readonly id: string;
  name: string;
  modes: { modeId: string; name: string }[];
}

interface Variable {
  readonly id: string;
  name: string;
  resolvedType: 'COLOR' | 'FLOAT' | 'STRING' | 'BOOLEAN';
  setValueForMode(modeId: string, value: RGBA | number | string | boolean): void;
}

interface VariableAlias {
  type: 'VARIABLE_ALIAS';
  id: string;
}

// ---- Viewport --------------------------------------------------------------

interface ViewportAPI {
  scrollAndZoomIntoView(nodes: ReadonlyArray<BaseNode>): void;
}

interface NotificationOptions {
  timeout?: number;
}
interface NotificationHandler {
  cancel: () => void;
}

export {};
