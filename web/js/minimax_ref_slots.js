/** Shared reference-slot helpers: numbering, loading, same-kind DnD, click preview. */

import { t } from "./minimax_i18n.js";
import {
    MAX_REFERENCE_AUDIOS,
    MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_VIDEOS,
    refAudioLabel,
    refImageLabel,
    refVideoLabel,
} from "./minimax_gen_timeline.js";

let activeClipboardImageTarget = null;
let clipboardPasteBound = false;

function clipboardImageFile(event) {
    const items = [...(event.clipboardData?.items || [])];
    const item = items.find((entry) => entry.kind === "file" && entry.type.startsWith("image/"));
    if (item) return item.getAsFile();
    return [...(event.clipboardData?.files || [])].find((file) => file.type.startsWith("image/")) || null;
}

export function bindImageClipboardPaste(el, onFile) {
    if (!el || typeof onFile !== "function") return;
    const target = { el, onFile };
    const activate = () => { activeClipboardImageTarget = target; };
    el.addEventListener("pointerenter", activate);
    el.addEventListener("focusin", activate);
    el.addEventListener("pointerdown", activate, true);
    if (clipboardPasteBound) return;
    clipboardPasteBound = true;
    document.addEventListener("paste", (event) => {
        const active = activeClipboardImageTarget;
        if (!active?.el?.isConnected) return;
        const file = clipboardImageFile(event);
        if (!file) return;
        event.preventDefault();
        event.stopPropagation();
        void active.onFile(file);
    }, true);
}

export const SLOT_DND_MIME = "application/x-minimax-media-slot";

export const SLOT_UI_STYLES = `
.bd-slot-load{position:absolute;inset:0;z-index:8;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;background:rgba(0,0,0,.78);color:#e8e8e8;font-size:10px;line-height:1.35;text-align:center;padding:8px 6px;pointer-events:auto;box-sizing:border-box}
.bd-slot-load-status{max-width:100%;overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical}
.bd-slot-load-bar{width:82%;height:3px;background:#222;border-radius:99px;overflow:hidden;flex-shrink:0}
.bd-slot-load-bar>i{display:block;height:100%;width:0;background:linear-gradient(90deg,#2a6b4a,#4fff8f);border-radius:99px;transition:width .12s linear}
.bd-slot-load.indeterminate .bd-slot-load-bar>i{width:42%;transition:none;animation:bd-slot-load-slide 1.05s ease-in-out infinite}
@keyframes bd-slot-load-slide{0%{transform:translateX(-110%)}100%{transform:translateX(240%)}}
.bd-batch-ref.drag-over,.bd-batch-audio.drag-over,.bd-batch-video.drag-over,
.bd-ref.drag-over,.bd-ref-audio.drag-over,.bd-ref-video.drag-over{border-color:#4fff8f!important;border-style:solid;background:#152018}
.bd-batch-ref.dragging,.bd-batch-audio.dragging,.bd-batch-video.dragging,
.bd-ref.dragging,.bd-ref-audio.dragging,.bd-ref-video.dragging{opacity:.45}
.bd-batch-audio.has-audio,.bd-batch-video.has-video,.bd-ref-audio.has-audio,.bd-ref-video.has-video{cursor:grab}
.bd-batch-audio.has-audio:active,.bd-batch-video.has-video:active,.bd-ref-audio.has-audio:active,.bd-ref-video.has-video:active{cursor:grabbing}
.bd-slot-preview{position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.82);display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box}
.bd-slot-preview-box{position:relative;max-width:min(920px,96vw);max-height:92vh;display:flex;flex-direction:column;gap:10px;min-width:0}
.bd-slot-preview-box img,.bd-slot-preview-box video{max-width:100%;max-height:min(78vh,820px);object-fit:contain;background:#000;border-radius:8px;display:block}
.bd-slot-preview-box audio{width:min(480px,90vw)}
.bd-slot-preview-head{display:flex;align-items:center;justify-content:space-between;gap:10px;color:#eee;font-size:13px;font-weight:650}
.bd-slot-preview-actions{display:flex;gap:8px;flex-shrink:0}
.bd-slot-preview-actions button{background:#222;color:#ddd;border:1px solid #444;border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer}
.bd-slot-preview-actions button:hover{border-color:#4fff8f;color:#4fff8f}
`;

export function refHasImage(r) {
    return !!(r?.imageFile || r?.imageB64);
}

export function refHasAudio(r) {
    return !!(r?.audioFile || r?.fileName);
}

export function refHasVideo(r) {
    return !!(r?.videoFile || r?.fileName || r?.previewImageFile || r?.previewImageUrl || r?.linked);
}

export const SLOT_KINDS = {
    image: {
        kind: "image",
        listKey: "refs",
        max: MAX_REFERENCE_IMAGES,
        hasFn: refHasImage,
        labelFn: refImageLabel,
    },
    audio: {
        kind: "audio",
        listKey: "refAudios",
        max: MAX_REFERENCE_AUDIOS,
        hasFn: refHasAudio,
        labelFn: refAudioLabel,
    },
    video: {
        kind: "video",
        listKey: "refVideos",
        max: MAX_REFERENCE_VIDEOS,
        hasFn: refHasVideo,
        labelFn: refVideoLabel,
    },
};

export function usedRefIndices(items, hasFn, maxSlots) {
    const used = new Set();
    for (const r of items || []) {
        if (!hasFn(r)) continue;
        const idx = Number(r.index ?? r.slot);
        if (Number.isFinite(idx) && idx >= 0 && idx < maxSlots) used.add(idx);
    }
    return used;
}

export function freeRefIndices(used, maxSlots) {
    const free = [];
    for (let i = 0; i < maxSlots; i++) {
        if (!used.has(i)) free.push(i);
    }
    return free;
}

export function slotKindSpec(kind) {
    return SLOT_KINDS[kind] || null;
}

export function groupFreeIndices(_editor, kind) {
    const spec = slotKindSpec(kind);
    if (!spec) return [];
    return freeRefIndices(new Set(), spec.max);
}

export function swapIndexedMedia(list, fromSlot, toSlot) {
    if (!Array.isArray(list) || fromSlot === toSlot) return list;
    const fromRef = list.find((r) => Number(r.index ?? r.slot) === fromSlot);
    if (!fromRef) return list;
    const toRef = list.find((r) => Number(r.index ?? r.slot) === toSlot);
    const next = list.filter((r) => {
        const idx = Number(r.index ?? r.slot);
        return idx !== fromSlot && idx !== toSlot;
    });
    next.push({ ...fromRef, index: toSlot, slot: undefined });
    if (toRef) next.push({ ...toRef, index: fromSlot, slot: undefined });
    return next;
}

export function countFilledOnIndices(items, indices, hasFn) {
    const set = new Set(indices);
    let n = 0;
    for (const r of items || []) {
        const idx = Number(r.index ?? r.slot);
        if (set.has(idx) && hasFn(r)) n += 1;
    }
    return n;
}

export function slotLoadKey({ scope, segId, kind, index }) {
    return `${scope}:${segId ?? ""}:${kind}:${index}`;
}

function ensureSlotLoads(editor) {
    if (!editor._slotLoads) editor._slotLoads = new Map();
    return editor._slotLoads;
}

export function isSlotBusy(editor, scope, segId, kind, index) {
    return !!editor?._slotLoads?.has(slotLoadKey({ scope, segId, kind, index }));
}

function cssEscape(value) {
    const s = String(value ?? "");
    if (typeof CSS !== "undefined" && CSS.escape) return CSS.escape(s);
    return s.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

export function findSlotEl(editor, key) {
    const parts = String(key || "").split(":");
    if (parts.length < 4) return null;
    const [scope, segId, kind, index] = parts;
    const root = editor?.root || editor?.batchList || document;
    if (scope === "global" || scope === "seg") {
        return root.querySelector?.(
            `[data-ref-scope="${cssEscape(scope)}"][data-ref-kind="${cssEscape(kind)}"][data-ref-index="${cssEscape(index)}"]`,
        );
    }
    const list = editor?.batchList || root;
    const card = (segId
        ? list.querySelector?.(`.bd-batch-card[data-batch-seg-id="${cssEscape(segId)}"]`)
        : null)
        || list.querySelector?.(`.bd-batch-card[data-batch-index="${cssEscape(segId)}"]`);
    const host = card || list;
    return host?.querySelector?.(
        `[data-ref-kind="${cssEscape(kind)}"][data-ref-index="${cssEscape(index)}"]`,
    );
}

function loadLabel(state) {
    const status = state?.status || t("slot.loading.prepare");
    const ratio = Number(state?.ratio);
    if (Number.isFinite(ratio) && ratio > 0 && ratio < 1) {
        return t("slot.loading.progress", { status, pct: Math.round(ratio * 100) });
    }
    if (Number.isFinite(state?.cur) && Number.isFinite(state?.total) && state.total > 1 && state.cur < state.total) {
        return t("slot.loading.chunk", { status, cur: state.cur, total: state.total });
    }
    return status;
}

export function paintSlotLoading(el, state) {
    if (!el) return;
    el.classList.add("is-loading");
    let overlay = el.querySelector(":scope > .bd-slot-load");
    if (!overlay) {
        overlay = document.createElement("div");
        overlay.className = "bd-slot-load";
        overlay.innerHTML = `<span class="bd-slot-load-status"></span><div class="bd-slot-load-bar"><i></i></div>`;
        el.appendChild(overlay);
    }
    const ratio = Number(state?.ratio);
    const determinate = Number.isFinite(ratio) && ratio > 0;
    overlay.classList.toggle("indeterminate", !determinate);
    const statusEl = overlay.querySelector(".bd-slot-load-status");
    if (statusEl) statusEl.textContent = loadLabel(state);
    const bar = overlay.querySelector(".bd-slot-load-bar > i");
    if (bar && determinate) bar.style.width = `${Math.max(0, Math.min(100, Math.round(ratio * 100)))}%`;
}

export function clearSlotLoading(el) {
    if (!el) return;
    el.classList.remove("is-loading");
    el.querySelector(":scope > .bd-slot-load")?.remove();
}

export function beginSlotLoad(editor, key, status) {
    const loads = ensureSlotLoads(editor);
    const state = { status: status || t("slot.loading.prepare"), ratio: 0 };
    loads.set(key, state);
    paintSlotLoading(findSlotEl(editor, key), state);
}

export function updateSlotLoad(editor, key, patch) {
    const loads = ensureSlotLoads(editor);
    const prev = loads.get(key) || { status: t("slot.loading.prepare"), ratio: 0 };
    const state = { ...prev, ...patch };
    loads.set(key, state);
    paintSlotLoading(findSlotEl(editor, key), state);
}

export function endSlotLoad(editor, key) {
    editor?._slotLoads?.delete(key);
    clearSlotLoading(findSlotEl(editor, key));
}

export function restoreSlotLoadOverlays(editor) {
    const loads = editor?._slotLoads;
    if (!loads?.size) return;
    for (const [key, state] of loads) {
        paintSlotLoading(findSlotEl(editor, key), state);
    }
}

export function nextEmptySlot(items, freeIndices, hasFn, editor, scope, segId, kind, reserved = null) {
    for (const abs of freeIndices) {
        if (reserved?.has(abs)) continue;
        if (isSlotBusy(editor, scope, segId, kind, abs)) continue;
        const hit = (items || []).find((r) => Number(r.index ?? r.slot) === abs);
        if (!hasFn(hit)) return abs;
    }
    return -1;
}

function parseSlotPayload(raw) {
    if (!raw) return null;
    try {
        const data = JSON.parse(raw);
        if (!data || !data.kind) return null;
        return data;
    } catch (_) {
        return null;
    }
}

export function isSlotDnD(types) {
    const list = [...(types || [])];
    return list.includes(SLOT_DND_MIME);
}

export function bindKindSlotDnD(el, {
    editor,
    kind,
    scope,
    segIndex,
    slotIndex,
    hasMedia,
    onMove,
    onDropFile,
}) {
    el.draggable = !!hasMedia;
    el.addEventListener("dragstart", (e) => {
        if (!hasMedia) {
            e.preventDefault();
            return;
        }
        editor._slotDragMoved = false;
        const payload = JSON.stringify({
            kind,
            scope,
            segIndex: scope === "global" ? -1 : segIndex,
            from: slotIndex,
        });
        e.dataTransfer.setData(SLOT_DND_MIME, payload);
        e.dataTransfer.setData("text/plain", payload);
        e.dataTransfer.effectAllowed = "move";
        el.classList.add("dragging");
    });
    el.addEventListener("dragend", () => {
        el.classList.remove("dragging");
        setTimeout(() => { editor._slotDragMoved = false; }, 0);
    });
    el.addEventListener("dragover", (e) => {
        const types = [...(e.dataTransfer?.types || [])];
        if (isSlotDnD(types) || types.includes("Files")) {
            e.preventDefault();
            e.stopPropagation();
            e.dataTransfer.dropEffect = isSlotDnD(types) ? "move" : "copy";
            if (isSlotDnD(types)) el.classList.add("drag-over");
        }
    });
    el.addEventListener("dragleave", () => el.classList.remove("drag-over"));
    el.addEventListener("drop", (e) => {
        el.classList.remove("drag-over");
        e.preventDefault();
        e.stopPropagation();
        const raw = e.dataTransfer.getData(SLOT_DND_MIME)
            || e.dataTransfer.getData("text/plain");
        const data = parseSlotPayload(raw);
        if (data?.kind === kind && data.scope === scope) {
            if (scope !== "global" && Number(data.segIndex) !== Number(segIndex)) return;
            if (isSlotBusy(editor, scope, scope === "global" ? "" : editor.timeline?.segments?.[segIndex]?.id ?? segIndex, kind, slotIndex)) {
                return;
            }
            editor._slotDragMoved = true;
            editor._batchRefDragMoved = true;
            editor._refDragMoved = true;
            onMove?.(Number(data.from), slotIndex);
            return;
        }
        const file = e.dataTransfer.files?.[0];
        if (file) onDropFile?.(file);
    });
}

export function bindSlotActivate(el, {
    editor,
    hasMedia,
    onPreview,
    onPick,
    ignoreSelector = ".x, .bd-r2v-play, .bd-r2v-progress, .bd-r2v-dur, video, audio, .bd-slot-load",
}) {
    el.addEventListener("click", (e) => {
        if (e.target.closest?.(ignoreSelector)) return;
        if (editor._slotDragMoved || editor._batchRefDragMoved || editor._refDragMoved) {
            editor._slotDragMoved = false;
            editor._batchRefDragMoved = false;
            editor._refDragMoved = false;
            return;
        }
        if (el.classList.contains("is-loading")) return;
        if (hasMedia) {
            onPreview?.();
            return;
        }
        onPick?.();
    });
}

let _previewEl = null;

export function closeSlotPreview() {
    _previewEl?.remove();
    _previewEl = null;
}

export function openSlotPreview({ kind, src, label, onReplace }) {
    closeSlotPreview();
    if (!src) return;
    const overlay = document.createElement("div");
    overlay.className = "bd-slot-preview";
    const box = document.createElement("div");
    box.className = "bd-slot-preview-box";
    const head = document.createElement("div");
    head.className = "bd-slot-preview-head";
    const title = document.createElement("span");
    title.textContent = label || t("slot.previewTitle");
    const actions = document.createElement("div");
    actions.className = "bd-slot-preview-actions";
    if (onReplace) {
        const replaceBtn = document.createElement("button");
        replaceBtn.type = "button";
        replaceBtn.textContent = t("slot.previewReplace");
        replaceBtn.onclick = (e) => {
            e.stopPropagation();
            closeSlotPreview();
            onReplace();
        };
        actions.appendChild(replaceBtn);
    }
    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.textContent = t("dialog.cancel");
    closeBtn.onclick = (e) => {
        e.stopPropagation();
        closeSlotPreview();
    };
    actions.appendChild(closeBtn);
    head.appendChild(title);
    head.appendChild(actions);
    box.appendChild(head);
    if (kind === "audio") {
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.autoplay = true;
        audio.src = src;
        box.appendChild(audio);
    } else if (kind === "video") {
        const video = document.createElement("video");
        video.controls = true;
        video.autoplay = true;
        video.playsInline = true;
        video.src = src;
        box.appendChild(video);
    } else {
        const img = document.createElement("img");
        img.src = src;
        img.alt = label || "";
        box.appendChild(img);
    }
    overlay.appendChild(box);
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) closeSlotPreview();
    });
    const onKey = (e) => {
        if (e.key === "Escape") {
            closeSlotPreview();
            window.removeEventListener("keydown", onKey);
        }
    };
    window.addEventListener("keydown", onKey);
    document.body.appendChild(overlay);
    _previewEl = overlay;
}

export function mediaSrcFromRef(ref, kind) {
    if (!ref) return "";
    if (kind === "audio") return ref.audioFile || ref.fileName || "";
    if (kind === "video") return ref.videoFile || "";
    return ref.imageFile || ref.imageB64 || "";
}
