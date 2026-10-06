import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { test } from "node:test";
import * as helpers from "../web/js/minimax_gen_timeline.js";

const read = name => fs.readFileSync(new URL(`../web/js/${name}`, import.meta.url), "utf8");
const timelineCode = read("minimax_timeline.js");
const snapshotCode = read("minimax_snapshots.js");
const refineCode = read("minimax_refine.js");
const samCode = read("minimax_sam31.js");
const plain = value => JSON.parse(JSON.stringify(value));

function top(code, name, context) {
    const start = code.indexOf(`function ${name}(`);
    assert.ok(start >= 0, name);
    const end = code.indexOf("\n}", start) + 2;
    return vm.runInContext(`(${code.slice(start, end)})`, context);
}

function method(name, context) {
    const start = timelineCode.indexOf(`\n    ${name}(`) + 5;
    assert.ok(start > 4, name);
    const end = timelineCode.indexOf("\n    }", start) + 6;
    return vm.runInContext(`({${timelineCode.slice(start, end)}})`, context)[name];
}

class Element {
    constructor(tag = "div") {
        Object.assign(this, {
            tagName: tag, children: [], dataset: {}, style: {}, attributes: {},
            listeners: {}, value: "", type: "", disabled: false, isConnected: true,
        });
        const classes = new Set();
        this.classList = {
            add: name => classes.add(name),
            remove: (...names) => names.forEach(name => classes.delete(name)),
            contains: name => classes.has(name),
            toggle: (name, enabled) => {
                if (enabled ?? !classes.has(name)) classes.add(name);
                else classes.delete(name);
            },
        };
    }
    set innerHTML(html) {
        this.children = [];
        for (const match of html.matchAll(/<(input|select|div|img|span|button)\b([^>]*)>/g)) {
            if (!/data-(role|config|action|compose)=/.test(match[2])) continue;
            const child = new Element(match[1]);
            for (const attr of match[2].matchAll(/([\w-]+)="([^"]*)"/g)) {
                child.setAttribute(attr[1], attr[2]);
                if (["type", "value", "min", "max"].includes(attr[1])) child[attr[1]] = attr[2];
            }
            this.appendChild(child);
        }
    }
    setAttribute(name, value) {
        this.attributes[name] = value;
        if (name.startsWith("data-")) {
            this.dataset[name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = value;
        }
    }
    removeAttribute(name) {
        delete this.attributes[name];
        if (name === "src") delete this.src;
    }
    appendChild(child) { this.children.push(child); child.parent = this; return child; }
    append(...children) { children.forEach(child => this.appendChild(child)); }
    replaceChildren(...children) { this.children = []; this.append(...children); }
    after(child) { this.parent?.appendChild(child); }
    remove() {
        this.isConnected = false;
        if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this);
    }
    querySelectorAll(selector) {
        const attr = selector.match(/^\[([^=\]]+)(?:="([^"]*)")?\]$/);
        const matches = [];
        const visit = node => {
            for (const child of node.children) {
                if ((attr && Object.hasOwn(child.attributes, attr[1])
                    && (attr[2] === undefined || child.attributes[attr[1]] === attr[2]))
                    || (!attr && child.tagName === selector)) matches.push(child);
                visit(child);
            }
        };
        visit(this);
        return matches;
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    addEventListener(type, fn) { (this.listeners[type] ??= []).push(fn); }
    removeEventListener(type, fn) {
        this.listeners[type] = (this.listeners[type] || []).filter(listener => listener !== fn);
    }
    async fire(type) {
        await Promise.all((this.listeners[type] || []).map(fn => fn({
            target: this, stopPropagation() {},
        })));
    }
}

async function flush() {
    for (let i = 0; i < 24; i++) await Promise.resolve();
}

function fixture() {
    const timers = new Map();
    let timerId = 0;
    const clock = {
        set(fn, delay) { const id = ++timerId; timers.set(id, { fn, delay }); return id; },
        clear(id) { timers.delete(id); },
        run(delay) {
            for (const [id, entry] of [...timers]) {
                if (entry.delay === delay) { timers.delete(id); entry.fn(); }
            }
        },
        size: () => timers.size,
    };
    const document = {
        activeElement: null,
        createElement: tag => new Element(tag),
        createTextNode: text => Object.assign(new Element("text"), { textContent: text }),
    };
    const calls = [];
    const previews = [];
    let missing = false;
    let jobStatus = "queued";
    const response = (data, status = 200) => ({
        ok: status < 400, status, text: async () => JSON.stringify(data),
    });
    const api = {
        fetchApi: async (url, options = {}) => {
            calls.push({ url, options });
            if (missing) return { ok: false, status: 404, text: async () => "<html>missing</html>" };
            if (url.endsWith("/capabilities")) {
                return response({ available: true, models: ["sam3.1.safetensors"] });
            }
            if (url.includes("/job?")) return response({ status: jobStatus });
            if (url.endsWith("/preview")) {
                // Permit responses after abort to test our own token guard.
                return new Promise(resolve => previews.push({
                    options, resolve: data => resolve(response(data)),
                }));
            }
            throw new Error(`Unexpected request: ${url}`);
        },
    };
    const context = vm.createContext({
        ...helpers, api, document, AbortController, console,
        setTimeout: clock.set, clearTimeout: clock.clear,
        syncFirstPassButtons() {}, normalizeOutputContinuity: value => value,
        stripTimelineContinuityRootFields() {}, stripTimelineEphemeralFields() {},
        sanitizeSegmentForPayload: value => value, sanitizeRefVideo: value => value,
        normalizeAudioMode: value => value, coerceTimelineFps: value => value,
        uid: () => "new-id", clampLivePreviewSpeed: value => value,
        LIVE_PREVIEW_SPEED_DEFAULT: 1, DEFAULT_CONTINUITY_FRAMES: 4, MIN_SEG: 4, H3_FPS: 24,
        isSegmentContinuityFromPrev: () => false, isSegmentContinuityForcePrevCache: () => false,
        findDirectorWidget: (node, name) => node.widgets.find(widget => widget.name === name),
        DIRECTOR_DOM_WIDGET_NAME: "director_dom", parseStoredFps: () => 24,
        ensureImageBatchTimeline() {}, snapshotDirectorSampleWidgets() {},
        cancelAnimationFrame() {}, teardownPromptImageMentions() {},
        window: { removeEventListener() {} },
    });
    context.closePassPanels = top(refineCode, "closePassPanels", context);
    vm.runInContext(samCode.replace(/^import\s[\s\S]*?;\r?$/gm, "")
        .replace(/\bexport\s+(?=const|function)/g, ""), context);
    for (const name of [
        "sanitizeRefImage", "sanitizeSourceImage", "sanitizeVideoMedia", "sanitizeSegmentForPayload",
    ]) context[name] = top(timelineCode, name, context);
    context.sanitizeRefAudio = value => value;
    const cfg = context.normalizeSam31Config({
        enabled: true, checkpoint: "sam3.1.safetensors",
    });
    const group = id => ({
        id, taskType: "v2v", length: 4, frameCount: 4,
        sourceVideo: {
            mediaKind: "video", rangeStart: 0, rangeEnd: 2, totalFrames: 2,
            video: { videoFile: `${id}.mp4`, sourceFrameCount: 2, width: 100, height: 100 },
        },
        sam31: { version: 1, prompt: `person ${id}`, selected: [0], result: {
            version: 1, key: id.repeat(64), action: "track",
            frames: 2, objects: [{ index: 0, score: 1 }],
        } },
    });
    const root = new Element();
    const bar = new Element();
    root.appendChild(bar);
    const widget = { name: "sam31_config", value: JSON.stringify(cfg) };
    const editor = {
        node: { widgets: [widget], setDirtyCanvas() {} }, root, outputBarEl: bar,
        selectedIndex: 0, timeline: {
            timelineMode: "prompt_batch", frameRate: 24, global: { taskType: "mixed" },
            output: { width: 864, height: 480 }, segments: [group("a"), group("b")],
        },
        scheduleTimelineSync() {}, resizeNodeForContentMinChange() {}, updateDomWidgetHeight() {},
        _persistCurrentBatchWorkspace() {}, getFrameRate: () => 24, _runSelectionPayload: () => ({}),
        _closeBdModal() {}, _clearPreviewVideos() {},
    };
    context.SAM31_WIDGET_NAMES = vm.runInContext("SAM31_WIDGET_NAMES", context);
    editor.buildTimelinePayload = () => method("buildTimelinePayload", context).call(editor);
    context.mountDirectorSam31Panel(editor);
    const el = role => editor.sam31PanelEl.querySelector(`[data-role="${role}"]`);
    const ready = async () => {
        editor._mmxSAM31PanelOpen = true;
        editor.syncSAM31Panel();
        await el("refresh").fire("click");
    };
    return {
        context, editor, widget, document, clock, calls, previews, el, ready,
        setMissing: value => { missing = value; },
        setJobStatus: value => { jobStatus = value; },
    };
}

function preview(frame) {
    return {
        image: `data:image/png;base64,FRAME_${frame}`,
        mapping: { clip: 0, source_frame: frame, frame_rate: 24,
            source_size: [100, 100], analysis_size: [100, 100] },
    };
}

test("config normalization and versioned group selection round-trip", () => {
    const h = fixture();
    const cfg = h.context.normalizeSam31Config({
        enabled: "false", threshold: NaN, max_objects: 999, long_edge: 9999,
    });
    assert.equal(cfg.enabled, false);
    assert.equal(cfg.threshold, 0.5);
    assert.equal(cfg.max_objects, 16);
    assert.equal(cfg.long_edge, 1024);
    assert.deepEqual(plain(h.context.normalizeSam31Config(JSON.stringify(cfg))), plain(cfg));
    assert.throws(() => h.context.normalizeSam31Config({ version: 2 }));
    assert.deepEqual(plain(h.context.normalizeSam31Group({
        selected: [0, 0, 15, -1, 16, true, "2"],
    }).selected), [0, 15]);
    h.editor.disposeSAM31Panel();
});

test("payload, parser, snapshot and named configure preserve SAM", () => {
    const h = fixture();
    const original = plain(h.editor.timeline.segments[0].sam31);
    const payload = h.editor.buildTimelinePayload();
    assert.deepEqual(plain(payload.segments[0].sam31), original);
    const parse = top(timelineCode, "parseTimeline", h.context);
    const restored = parse(JSON.stringify(payload), 8, 24);
    assert.deepEqual(plain(restored.segments[0].sam31), original);
    h.context.pruneHiddenMedia = top(snapshotCode, "pruneHiddenMedia", h.context);
    const snapshot = top(snapshotCode, "buildSnapshotTimeline", h.context)(h.editor);
    assert.deepEqual(plain(snapshot.segments[0].sam31), original);
    vm.runInContext(snapshotCode.match(/const SNAPSHOT_WIDGET_NAMES = .*;/)[0], h.context);
    const collected = top(snapshotCode, "collectSnapshotWidgets", h.context)({
        widget: () => h.widget, timeline: h.editor.timeline,
    });
    assert.equal(collected.sam31_config, h.widget.value);
    const value = h.widget.value;
    h.widget.value = "{}";
    const restore = top(timelineCode, "restoreDirectorWidgetsFromSaved", h.context);
    assert.equal(restore(h.editor.node, { widgets_values_named: { sam31_config: value } }), true);
    assert.equal(h.widget.value, value);
    h.editor.disposeSAM31Panel();
});

test("actual snapshot import restores SAM and old packs reset it disabled", () => {
    const h = fixture();
    h.context.parseTimeline = top(timelineCode, "parseTimeline", h.context);
    Object.assign(h.editor, {
        timelineWidget: { value: "" }, widget: name => name === "sam31_config" ? h.widget : null,
        getDirectorMode: () => "prompt_batch",
    });
    for (const name of [
        "_restoreUserLivePreviewSpeed", "writeImportedTimelineToExternalGroups", "applyTaskLayout",
        "lockH3FrameRate", "populateTaskSelect", "syncLivePreviewSpeedUI", "setEditMode", "commit",
    ]) h.editor[name] = () => {};
    h.editor.updateSelectionUI = () => h.editor.syncSAM31Panel();
    const saved = plain(h.editor.timeline);
    const value = h.widget.value;
    const imported = method("applyImportedTimeline", h.context);
    imported.call(h.editor, saved, { sam31_config: value });
    assert.equal(h.widget.value, value);
    assert.deepEqual(plain(h.editor.timeline.segments[0].sam31), saved.segments[0].sam31);
    imported.call(h.editor, saved, {});
    assert.equal(JSON.parse(h.widget.value).enabled, false);
    h.editor.disposeSAM31Panel();
});

test("external same-node rebuild retains group SAM by stable ID", () => {
    const h = fixture();
    const saved = plain(h.editor.timeline.segments[0].sam31);
    h.context.collectExternalGroupSpecs = () => [{ nodeId: 7, durationSec: 5, prompt: "scene" }];
    h.context.flushBatchPromptInputs = () => {};
    h.context.imageRefFromPath = () => null;
    h.context.normalizeImageBatchSegments = () => {};
    h.context.resolveSegmentPassMode = () => "all";
    h.editor.timeline.segments[0].externalNodeId = 7;
    Object.assign(h.editor, {
        updateExternalGroupsBanner() {}, getDirectorMode: () => "prompt_batch",
        getTaskKey: () => "mixed", isImageBatch: () => false,
    });
    method("syncExternalGroupsTimeline", h.context).call(h.editor);
    assert.equal(h.editor.timeline.segments[0].id, "a");
    assert.deepEqual(plain(h.editor.timeline.segments[0].sam31), saved);
    h.editor.disposeSAM31Panel();
});

test("focused object text follows group switches and same-ID restores", () => {
    const h = fixture();
    h.document.activeElement = h.el("prompt");
    h.editor.selectedIndex = 1;
    h.editor.syncSAM31Panel();
    assert.equal(h.el("prompt").value, "person b");
    h.editor.timeline.segments[1].sam31.prompt = "restored object";
    h.editor.syncSAM31Panel();
    assert.equal(h.el("prompt").value, "restored object");
    h.editor.disposeSAM31Panel();
});

test("slider changes reject old-frame responses during debounce", async () => {
    const h = fixture();
    await h.ready();
    const pending = h.el("preview-button").fire("click");
    await flush();
    assert.equal(h.previews.length, 1);
    h.el("frame").value = "1";
    await h.el("frame").fire("input");
    h.previews[0].resolve(preview(0));
    await pending;
    assert.equal(h.el("preview").src, undefined);
    h.clock.run(200);
    await flush();
    assert.equal(h.previews.length, 2);
    assert.equal(JSON.parse(h.previews[1].options.body).frame, 1);
    h.previews[1].resolve(preview(1));
    await flush();
    assert.match(h.el("preview").src, /FRAME_1/);
    h.editor.disposeSAM31Panel();
});

test("an obsolete request cannot clear a newer request's busy state", async () => {
    const h = fixture();
    await h.ready();
    const old = h.el("preview-button").fire("click");
    await flush();
    h.el("frame").value = "1";
    await h.el("frame").fire("input");
    h.clock.run(200);
    await flush();
    assert.equal(h.previews.length, 2);
    h.previews[0].resolve(preview(0));
    await old;
    assert.equal(h.el("preview-button").disabled, true);
    h.previews[1].resolve(preview(1));
    await flush();
    assert.equal(h.el("preview-button").disabled, false);
    h.editor.disposeSAM31Panel();
});

test("missing backend disables analysis and cache preview buttons", async () => {
    const h = fixture();
    h.setMissing(true);
    await h.ready();
    assert.ok(h.editor.sam31PanelEl.querySelectorAll("[data-action]").every(button => button.disabled));
    assert.equal(h.el("preview-button").disabled, true);
    assert.match(h.el("environment").textContent, /404/);
    h.editor.disposeSAM31Panel();
});

test("closing another pass drawer stops polling without cancelling the queued job", async () => {
    const h = fixture();
    const group = h.editor.timeline.segments[0];
    group.sam31.jobId = "pending";
    group.sam31.jobSignature = h.context.sam31InputSignature(
        h.editor.timeline, group, JSON.parse(h.widget.value),
    );
    h.editor._mmxSAM31PanelOpen = true;
    h.editor.syncSAM31Panel();
    const refresh = h.el("refresh").fire("click");
    await flush();
    assert.equal(h.clock.size(), 1);
    h.context.closePassPanels(h.editor, "refine");
    await refresh;
    assert.equal(h.clock.size(), 0);
    assert.equal(group.sam31.jobId, "pending");
    assert.equal(h.editor._mmxSAM31PanelOpen, false);
    assert.ok(!h.calls.some(call => /cancel|interrupt/.test(call.url)));
    h.editor.disposeSAM31Panel();
});

test("actual editor destroy clears debounce, aborts preview and rejects its late response", async () => {
    const h = fixture();
    await h.ready();
    const pending = h.el("preview-button").fire("click");
    await flush();
    h.el("frame").value = "1";
    await h.el("frame").fire("input");
    assert.equal(h.clock.size(), 1);
    method("destroy", h.context).call(h.editor);
    assert.equal(h.clock.size(), 0);
    assert.equal(h.editor.sam31PanelEl, null);
    assert.equal(h.editor.root, null);
    assert.equal(h.previews[0].options.signal.aborted, true);
    h.previews[0].resolve(preview(0));
    await pending;
    assert.equal(h.clock.size(), 0);
});

test("SAM input signatures match real media serialization", () => {
    const h = fixture();
    for (const taskType of ["i2v", "fl2v", "r2v", "v2v", "rv2v"]) {
        const group = h.editor.timeline.segments[0];
        Object.assign(group, {
            taskType,
            genImage: { imageFile: "a.png", width: 123 },
            startImage: { imageFile: "first.png" },
            refs: [{ imageFile: "ref.png", width: 123, height: 456 }],
        });
        const cfg = JSON.parse(h.widget.value);
        const payload = h.editor.buildTimelinePayload();
        assert.equal(
            h.context.sam31InputSignature(h.editor.timeline, group, cfg),
            h.context.sam31InputSignature(payload, payload.segments[0], cfg),
            taskType,
        );
    }
    h.editor.disposeSAM31Panel();
});

test("composite controls are per-group and preserve tracking identity", async () => {
    const h = fixture();
    const group = h.editor.timeline.segments[0];
    const cfg = JSON.parse(h.widget.value);
    const before = h.context.sam31InputSignature(h.editor.timeline, group, cfg);
    const checkbox = h.editor.sam31PanelEl.querySelector('[data-compose="enabled"]');
    const grow = h.editor.sam31PanelEl.querySelector('[data-compose="grow"]');
    assert.equal(checkbox.checked, false);
    checkbox.checked = true;
    await checkbox.fire("change");
    grow.value = "12";
    await grow.fire("change");
    assert.equal(group.sam31.composite.enabled, true);
    assert.equal(group.sam31.composite.grow, 12);
    assert.equal(h.context.sam31InputSignature(h.editor.timeline, group, cfg), before);
    const payload = h.editor.buildTimelinePayload();
    assert.deepEqual(plain(payload.segments[0].sam31.composite),
        { enabled: true, grow: 12, feather: 4, blend: 1 });
    h.context.pruneHiddenMedia = top(snapshotCode, "pruneHiddenMedia", h.context);
    const snapshot = top(snapshotCode, "buildSnapshotTimeline", h.context)(h.editor);
    assert.equal(snapshot.segments[0].sam31.composite.enabled, true);
    h.editor.selectedIndex = 1;
    h.editor.syncSAM31Panel();
    assert.equal(checkbox.checked, false);
    assert.equal(grow.value, 0);
    h.editor.selectedIndex = 0;
    h.editor.syncSAM31Panel();
    assert.equal(checkbox.checked, true);
    assert.equal(grow.value, 12);
    assert.match(h.el("compose-status").textContent, /过期蒙版|已绑定/);
    h.editor.disposeSAM31Panel();
});

test("SAM's local dependency graph does not import SAM recursively", () => {
    const localEntry = path.resolve("web/js/minimax_sam31.js");
    const edges = new Map();
    function scan(file) {
        if (edges.has(file)) return;
        const source = fs.readFileSync(file, "utf8");
        const dependencies = [...source.matchAll(/\bfrom\s+["'](\.\/[^"']+\.js)["']/g)]
            .map(match => path.resolve(path.dirname(file), match[1]));
        edges.set(file, dependencies);
        dependencies.forEach(scan);
    }
    function reaches(file, target, seen = new Set()) {
        if (file === target) return true;
        if (seen.has(file)) return false;
        seen.add(file);
        return (edges.get(file) || []).some(next => reaches(next, target, seen));
    }
    scan(localEntry);
    assert.ok(edges.get(localEntry).every(dependency => !reaches(dependency, localEntry)));
    assert.ok(edges.size >= 3);
});
