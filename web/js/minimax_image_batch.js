/** Multi prompt-group UI for t2i / i2i / r2i / t2v / i2v / r2v / mixed (prompt batch mode). */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    DEFAULT_ASPECT_RATIO,
    DEFAULT_MEGAPIXELS,
    defaultDurationSec,
    defaultFrameCount,
    durationToClampedMiniMaxFrames,
    durationToMiniMaxFrames,
    floorMiniMaxFrameCount,
    framesToDurationSec,
    imageBatchVariant,
    isContinuityMasterEnabled,
    isSegmentContinuityForcePrevCache,
    isSegmentContinuityFromPrev,
    isMixedTask,
    isVideoBatchTask,
    MIXED_GROUP_TASKS,
    MAX_GEN_FRAMES,
    MAX_REFERENCE_AUDIOS,
    MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_VIDEOS,
    maxDurationSec,
    MINIMAX_CANVAS_MULTIPLE,
    minDurationSec,
    minFrameCount,
    newBatchSegment,
    normalizeSegmentSeed,
    normalizeSegmentSeedMode,
    normalizeVideoFit,
    preferredDurationSecFromFrames,
    refAudioLabel,
    refImageLabel,
    refVideoLabel,
    REF_IMAGE_SIZE_OPTIONS,
    resolveMixedGroupKey,
    resolveSegmentRefImageSize,
    resolveSegmentPassMode,
    resolveTaskKey,
    roundDurationSec,
    sumFrameCounts,
    fileForComfyUpload,
    safeUploadFilename,
} from "./minimax_gen_timeline.js";
import {
    promptVideosFor,
    refreshPromptTokenEditors,
    teardownPromptImageMentions,
    wirePromptImageMentions,
} from "./minimax_prompt_mentions.js";
import { mountGroupVideoTimeline } from "./minimax_group_video_timeline.js";
import { t } from "./minimax_i18n.js";
import {
    hasDuplicateReferenceAudio,
    isReferenceAudioSourceFile,
    isReferenceAudioVideoFile,
    prepareLocalReferenceAudio,
} from "./minimax_ref_audio.js";
import {
    SLOT_UI_STYLES,
    beginSlotLoad,
    bindImageClipboardPaste,
    bindKindSlotDnD,
    bindSlotActivate,
    countFilledOnIndices,
    endSlotLoad,
    groupFreeIndices,
    isSlotBusy,
    isSlotDnD,
    mediaSrcFromRef,
    nextEmptySlot,
    openSlotPreview,
    refHasAudio,
    refHasImage,
    refHasVideo,
    restoreSlotLoadOverlays,
    slotLoadKey,
    swapIndexedMedia,
    updateSlotLoad,
} from "./minimax_ref_slots.js";

const _players = new WeakMap();
/** r2v picture grid: 9 slots in 3×3; reveal 3 → 6 → 9. */
const R2V_PICTURE_SLOTS = MAX_REFERENCE_IMAGES;
const R2V_PICTURE_STEP = 3;
let _activeR2vMedia = null;

function clamp(n, lo, hi) {
    return Math.max(lo, Math.min(hi, n));
}

export function formatMediaDuration(sec) {
    if (!Number.isFinite(sec) || sec < 0) return "--:--";
    const total = Math.max(0, Math.round(sec));
    const m = Math.floor(total / 60);
    const s = total % 60;
    return `${m}:${String(s).padStart(2, "0")}`;
}

function pauseActiveR2vMedia(except = null) {
    if (_activeR2vMedia && _activeR2vMedia !== except) {
        try {
            _activeR2vMedia.pause();
        } catch (_) { /* ignore */ }
        const btn = _activeR2vMedia._r2vPlayBtn;
        if (btn) btn.textContent = "▶";
    }
    if (_activeR2vMedia !== except) _activeR2vMedia = null;
}

export function bindR2vMediaPlayback(mediaEl, playBtn, progressWrap = null) {
    mediaEl.classList.add("bd-r2v-media");
    mediaEl._r2vPlayBtn = playBtn;
    const fill = progressWrap?.querySelector?.(".bd-r2v-progress-fill");
    const syncBtn = () => {
        playBtn.textContent = mediaEl.paused ? "▶" : "⏸";
    };
    const syncProgress = () => {
        if (!progressWrap || !fill) return;
        const dur = mediaEl.duration;
        const pct = Number.isFinite(dur) && dur > 0
            ? Math.min(100, Math.max(0, (mediaEl.currentTime / dur) * 100))
            : 0;
        fill.style.width = `${pct}%`;
        progressWrap.classList.toggle("active", !mediaEl.paused);
        progressWrap.classList.toggle("playing", !mediaEl.paused);
    };
    playBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (mediaEl.paused) {
            pauseActiveR2vMedia(mediaEl);
            mediaEl.play().catch(() => {});
            _activeR2vMedia = mediaEl;
        } else {
            mediaEl.pause();
            if (_activeR2vMedia === mediaEl) _activeR2vMedia = null;
        }
        syncBtn();
        syncProgress();
    });
    mediaEl.addEventListener("play", () => {
        pauseActiveR2vMedia(mediaEl);
        _activeR2vMedia = mediaEl;
        syncBtn();
        syncProgress();
    });
    mediaEl.addEventListener("pause", () => {
        syncBtn();
        syncProgress();
    });
    mediaEl.addEventListener("timeupdate", syncProgress);
    mediaEl.addEventListener("ended", () => {
        if (_activeR2vMedia === mediaEl) _activeR2vMedia = null;
        mediaEl.currentTime = 0;
        syncBtn();
        syncProgress();
        progressWrap?.classList.remove("active", "playing");
        if (fill) fill.style.width = "0%";
    });
    if (progressWrap && fill) {
        progressWrap.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const dur = mediaEl.duration;
            if (!Number.isFinite(dur) || dur <= 0) return;
            const rect = progressWrap.getBoundingClientRect();
            const ratio = rect.width > 0 ? (e.clientX - rect.left) / rect.width : 0;
            mediaEl.currentTime = Math.min(dur, Math.max(0, ratio * dur));
            syncProgress();
        });
    }
}

export function wireMediaDuration(mediaEl, durEl, onReady) {
    const apply = () => {
        if (!Number.isFinite(mediaEl.duration) || mediaEl.duration === Infinity) return;
        durEl.textContent = formatMediaDuration(mediaEl.duration);
        onReady?.(mediaEl.duration);
    };
    // Chromium may deliver duration after loadedmetadata for Range-served
    // local MP4 files.  Listen to all metadata-ready transitions instead of
    // leaving a filled slot stuck at --:--.
    mediaEl.addEventListener("loadedmetadata", apply);
    mediaEl.addEventListener("durationchange", apply);
    mediaEl.addEventListener("loadeddata", apply);
    mediaEl.addEventListener("canplay", apply);
    if (mediaEl.readyState >= 1) apply();
}

/** User-facing seconds (1 decimal). */
function resolveSegmentDurationSec(seg) {
    if (seg.durationSec != null && Number.isFinite(Number(seg.durationSec))) {
        const { durationSec } = durationToClampedMiniMaxFrames(seg.durationSec, 24);
        return durationSec;
    }
    return defaultDurationSec(resolveTaskKey(seg.taskType || ""));
}

function resolveVideoSegmentDuration(taskKey, seg, rawSec = resolveSegmentDurationSec(seg), { enforceSourceFrames = false } = {}) {
    const segTaskKey = taskKey === "mixed" ? resolveMixedGroupKey(seg) : taskKey;
    const heldImage = seg.sourceVideo?.mediaKind === "image";
    const sourceFrames = ["v2v", "rv2v"].includes(segTaskKey)
        ? (heldImage ? 0 : sourceVideoFrameMap(seg.sourceVideo).length)
        : 0;
    const clamped = clamp(
        Number(rawSec) || defaultDurationSec(taskKey),
        minDurationSec(),
        maxDurationSec(),
    );
    const requested = durationToClampedMiniMaxFrames(clamped, 24);
    const frames = enforceSourceFrames ? Math.max(sourceFrames, requested.frames) : requested.frames;
    if (heldImage) {
        seg.sourceVideo.totalFrames = frames;
        seg.sourceVideo.rangeStart = 0;
        seg.sourceVideo.rangeEnd = frames;
    }
    return {
        frames,
        durationSec: preferredDurationSecFromFrames(frames, 24),
        sourceFrames,
    };
}

/** Apply seconds to a segment by index (avoids stale closures after normalize). */
function applyBatchSegmentDuration(editor, index, rawSec) {
    const taskKey = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    const seg = editor.timeline.segments?.[index];
    if (!seg || !isVideoBatchTask(taskKey)) return null;
    const { frames, durationSec } = resolveVideoSegmentDuration(taskKey, seg, rawSec);
    seg.durationSec = durationSec;
    seg.frameCount = frames;
    seg.length = frames;
    seg._videoFrameCount = frames;
    // Stale drag preview must not override batch totals.
    if (editor._previewSegments) editor._previewSegments = null;
    normalizeImageBatchSegments(editor);
    return editor.timeline.segments[index] || null;
}

/**
 * Resolve a live segment from a batch card control.
 * Prefer segment id — after splice/reorder, DOM indices no longer match the array.
 */
function liveBatchSegmentFromEl(editor, el, indexAttr) {
    const segs = editor?.timeline?.segments;
    if (!el || !Array.isArray(segs) || !segs.length) return null;
    const id = el.getAttribute("data-batch-seg-id");
    if (id) {
        const index = segs.findIndex((s) => s?.id && s.id === id);
        if (index >= 0) return { seg: segs[index], index };
        return null;
    }
    const index = parseInt(el.getAttribute(indexAttr), 10);
    if (!Number.isFinite(index) || index < 0 || index >= segs.length) return null;
    const cardIdx = parseInt(el.closest?.(".bd-batch-card")?.dataset?.batchIndex, 10);
    if (Number.isFinite(cardIdx) && cardIdx >= 0 && cardIdx < segs.length) {
        return { seg: segs[cardIdx], index: cardIdx };
    }
    const nCards = editor.batchList?.querySelectorAll(".bd-batch-card")?.length ?? 0;
    if (nCards !== segs.length) return null;
    return { seg: segs[index], index };
}

/**
 * Pull prompt textareas into timeline.segments.
 * Must run before normalize / re-render / timeline sync — otherwise edits sit on
 * stale segment objects (or only in the DOM) and get wiped.
 */
export function flushBatchPromptInputs(editor) {
    const list = editor?.batchList;
    if (!list) return;
    const segs = editor?.timeline?.segments;
    if (!Array.isArray(segs) || !segs.length) return;
    list.querySelectorAll("textarea[data-batch-prompt-index]").forEach((el) => {
        el.__bdTokenApi?.sync?.();
        const live = liveBatchSegmentFromEl(editor, el, "data-batch-prompt-index");
        if (!live?.seg) return;
        live.seg.prompt = el.value || "";
        live.seg.negativePrompt = live.seg.negativePrompt ?? "";
    });
}

/** Flush visible 秒数 inputs into segments before a full card re-render. */
export function flushBatchDurationInputs(editor) {
    const list = editor?.batchList;
    if (!list) return;
    // Persist prompts first — duration apply/normalize must not drop textarea drafts.
    flushBatchPromptInputs(editor);
    const taskKey = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    if (!isVideoBatchTask(taskKey)) return;
    for (const input of list.querySelectorAll("input[data-batch-sec-index]")) {
        if (input.readOnly) continue;
        const live = liveBatchSegmentFromEl(editor, input, "data-batch-sec-index");
        if (!live?.seg) continue;
        clearTimeout(input._t);
        input._t = null;
        const displayed = parseFloat(input.value);
        if (!Number.isFinite(displayed)) continue;
        const current = Number(live.seg.durationSec);
        // Skip if already in sync (avoid churn while typing the same committed value).
        if (Number.isFinite(current) && roundDurationSec(displayed) === roundDurationSec(current)
            && input !== document.activeElement) {
            continue;
        }
        applyBatchSegmentDuration(editor, live.index, displayed);
    }
}

function formatPreviewFps(value) {
    const fps = Math.round(Number(value) * 100) / 100;
    if (Number.isInteger(fps)) return String(fps);
    return fps.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function stopPlayer(el) {
    const st = _players.get(el);
    if (!st) return;
    st.playing = false;
    if (st.timer) {
        clearInterval(st.timer);
        st.timer = null;
    }
}

function stopAllPlayers(root) {
    root?.querySelectorAll(".bd-batch-vpreview")?.forEach((wrap) => stopPlayer(wrap));
    pauseActiveR2vMedia(null);
    root?.querySelectorAll("video.bd-r2v-media, audio.bd-r2v-media")?.forEach((m) => {
        try { m.pause(); } catch (_) { /* ignore */ }
    });
}

export function refreshBatchVideoFrames(editor) {
    const videos = editor?.batchList?.querySelectorAll?.(
        ".bd-group-video-player video, video.bd-r2v-media",
    ) || [];
    for (const video of videos) {
        if (!video.src || !video.paused) continue;
        const restoreTime = Number(video.currentTime) || 0;
        const restoreFrame = () => {
            const duration = Number(video.duration);
            const target = restoreTime > 0
                ? restoreTime
                : (Number.isFinite(duration) ? Math.min(0.05, duration * 0.01) : 0.05);
            try { video.currentTime = target; } catch (_) { /* metadata not ready */ }
        };
        video.addEventListener("loadedmetadata", restoreFrame, { once: true });
        video.load();
    }
}

export const IMAGE_BATCH_STYLES = `
.bd-btn.bd-disabled,.bd-btn:disabled{opacity:.38;cursor:not-allowed;pointer-events:none}
.bd-mode button.bd-disabled,.bd-mode button:disabled{opacity:.38;cursor:not-allowed;pointer-events:none}
.bd-batch{width:100%;box-sizing:border-box;display:flex;flex-direction:column;gap:8px}
.bd-batch-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.bd-batch-picker{display:none;flex-wrap:wrap;gap:6px;width:100%;box-sizing:border-box;padding:2px 0 6px;flex-shrink:0}
.bd-batch-picker.visible{display:flex}
.bd-batch-pick{display:flex;flex-direction:column;gap:2px;min-width:92px;max-width:140px;padding:6px 8px;border:1px solid #333;border-radius:8px;background:#161616;cursor:pointer;color:#ccc;user-select:none}
.bd-batch-pick:hover{border-color:#4a7a5a}
.bd-batch-pick.selected{border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.35);color:#eafff0}
.bd-batch-pick.running{border-color:#4fff8f}
.bd-batch-pick.run-skipped{opacity:.45}
.bd-batch-pick-title{display:flex;align-items:center;gap:4px;font-size:11px;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bd-batch-pick-meta{font-size:10px;color:#8aa}
.bd-batch-pick-thumb{width:100%;height:40px;object-fit:cover;border-radius:4px;background:#0d0d0d;margin-top:2px}
.bd-batch-run-select.active{background:#1a3a2a;color:#4fff8f;border-color:#4fff8f}
.bd-batch-run-all{display:inline-flex;align-items:center;gap:4px;font-size:11px;color:#aaa;cursor:pointer;user-select:none}
.bd-batch-run-all.hidden{display:none!important}
.bd-batch-run-all input{width:14px;height:14px;margin:0;cursor:pointer;accent-color:#4fff8f}
/* Default cap; batch-fill mode overrides via .bd-wrap.bd-batch-fill + JS max-height. */
.bd-batch-list{display:flex;flex-direction:column;gap:8px;width:100%;max-height:none;overflow-y:visible;padding-right:2px;min-height:0}
.bd-batch-card{background:linear-gradient(165deg,#1a1a1a 0%,#141414 55%,#111 100%);border:1px solid #2c2c2c;border-radius:10px;padding:12px 14px;display:grid;grid-template-rows:auto minmax(0,1fr);gap:10px;align-items:stretch;box-shadow:inset 0 1px 0 rgba(255,255,255,.03);flex:0 0 auto;box-sizing:border-box;min-height:195px;overflow:hidden;resize:vertical}
.bd-batch-card.bd-batch-task-t2v{background:linear-gradient(165deg,#1d231f 0%,#171b18 55%,#121412 100%);border-color:#344039}
.bd-batch-card.bd-batch-task-i2v{background:linear-gradient(165deg,#1b2229 0%,#161b20 55%,#111416 100%);border-color:#34434f}
.bd-batch-card.bd-batch-task-fl2v{background:linear-gradient(165deg,#25231d 0%,#1d1b17 55%,#151411 100%);border-color:#494436}
.bd-batch-card.bd-batch-task-r2v{background:linear-gradient(165deg,#261f21 0%,#1e181a 55%,#151213 100%);border-color:#4b393e}
.bd-batch-card.bd-batch-task-v2v{background:linear-gradient(165deg,#192525 0%,#151d1d 55%,#101515 100%);border-color:#324949}
.bd-batch-card.bd-batch-task-rv2v{background:linear-gradient(165deg,#232129 0%,#1b1920 55%,#131216 100%);border-color:#423d4d}
/* t2v: 提示词 | 预览（开实时预览才出第三列） */
.bd-batch-card.bd-batch-plain{grid-template-columns:minmax(0,1fr)}
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-plain{grid-template-columns:minmax(0,1fr) minmax(180px,.55fr)}
/* i2v: 源图 | 提示词 | 预览 */
.bd-batch-card.bd-batch-source,.bd-batch-card.bd-batch-fl2v,.bd-batch-card.bd-batch-refs:not(.bd-batch-r2v){grid-template-columns:auto minmax(0,1fr)}
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-source,
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-fl2v,
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-refs:not(.bd-batch-r2v){grid-template-columns:auto minmax(0,1fr) minmax(180px,.55fr)}
.bd-batch-card.bd-batch-v2v{grid-template-columns:minmax(180px,280px) minmax(0,1fr)}
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-v2v{grid-template-columns:minmax(180px,280px) minmax(0,1fr) minmax(180px,.55fr)}
/* mixed: 素材区与预览区等宽，提示词区稍宽；t2v 无素材区。 */
.bd-batch-card.bd-batch-mixed:not(.bd-batch-r2v){grid-template-columns:minmax(0,1fr) minmax(0,1.3fr)}
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-mixed:not(.bd-batch-r2v){grid-template-columns:minmax(0,1fr) minmax(0,1.3fr) minmax(0,1fr)}
.bd-batch-card.bd-batch-mixed.bd-batch-plain{grid-template-columns:minmax(0,1.3fr)}
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-mixed.bd-batch-plain{grid-template-columns:minmax(0,1fr) minmax(0,1.3fr) minmax(0,1fr)}
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-mixed.bd-batch-plain>.bd-batch-prompts{grid-column:1/3}
.bd-batch-type{max-width:168px;min-width:108px;font-size:11px;padding:2px 6px;height:24px}
.bd-batch-fl2v-media{display:flex;flex-direction:column;gap:6px;min-width:220px;max-width:280px;width:100%}
.bd-batch-mixed>.bd-batch-fl2v-media{min-width:0;max-width:none}
.bd-batch-fl2v-slots{display:grid;grid-template-columns:1fr 1fr;gap:6px;width:100%;min-width:0}
.bd-batch-fl2v-slots .bd-batch-src{width:100%;height:auto;aspect-ratio:var(--batch-fl2v-slot-ar,16/9);min-height:72px;font-size:10px;line-height:1.35;padding:6px}
.bd-batch-fl2v-slots .bd-batch-src .ph{color:#666;font-size:10px;text-align:center;line-height:1.35;pointer-events:none}
.bd-batch-fl2v-slots .bd-batch-src .tag{position:absolute;top:4px;padding:1px 5px;border-radius:2px;font-size:9px;font-weight:700;line-height:1.4;pointer-events:none;z-index:2}
.bd-batch-fl2v-slots .bd-batch-src .tag.start{left:4px;background:rgba(79,255,143,.92);color:#111}
.bd-batch-fl2v-slots .bd-batch-src .tag.end{right:4px;left:auto;background:rgba(240,160,48,.92);color:#111}
.bd-batch-video-source{width:100%;max-width:320px;height:auto;aspect-ratio:16/9;padding:0}
.bd-batch-video-source video{width:100%;height:100%;object-fit:contain;background:#000;pointer-events:none}
.bd-batch-video-source .ph{padding:12px;color:#777;font-size:11px}
.bd-batch-r2v-assets .bd-batch-video-source{max-width:none}
.bd-batch-v2v .bd-batch-media{width:100%;max-width:none;min-width:0;min-height:0;height:100%;align-self:stretch;overflow:auto;background:#0c0c0c;border:1px solid #262626;border-radius:10px;padding:10px 12px;box-sizing:border-box}
.bd-batch-source-video-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:2px}
.bd-batch-source-video-title{color:#eaeaea;font-size:11px;font-weight:700}
.bd-batch-source-video-actions{display:flex;align-items:center;gap:6px;margin-left:auto;flex-wrap:wrap;justify-content:flex-end}
.bd-batch-video-resolution,.bd-batch-video-fit{min-width:0;max-width:118px;background:#161616;border:1px solid #3a3a3a;border-radius:6px;color:#eee;padding:3px 6px;font-size:11px}
.bd-batch-video-fit.hidden{display:none!important}
.bd-batch-source-video-restore{background:#181818;border:1px solid #444;color:#ccc;border-radius:4px;padding:3px 7px;font-size:10px;cursor:pointer;white-space:nowrap}
.bd-batch-source-video-restore:disabled{opacity:.45;cursor:not-allowed}
.bd-batch-source-video-restore:not(:disabled):hover{border-color:#666;color:#fff}
.bd-batch-source-video-delete{background:transparent;border:1px solid #553;color:#f88;border-radius:4px;padding:3px 7px;font-size:10px;cursor:pointer;white-space:nowrap}
.bd-batch-source-video-delete:disabled{border-color:#3a3a3a;color:#777;opacity:.55;cursor:not-allowed}
.bd-batch-source-video-delete:not(:disabled):hover{background:#3a1515}

.bd-preview-col{display:flex;flex-direction:column;gap:6px;min-width:0;height:0;min-height:100%;max-height:100%;overflow:hidden;align-self:stretch;background:#0c0c0c;border:1px solid #262626;border-radius:10px;padding:10px 12px;box-sizing:border-box}
.bd-preview-col-head{display:flex;align-items:center;justify-content:flex-start;gap:6px;flex-shrink:0}
.bd-preview-tab{background:#222;color:#bbb;border:1px solid #333;border-radius:4px;padding:2px 8px;font-size:10px;line-height:1.4;cursor:pointer;white-space:nowrap;flex-shrink:0}
.bd-preview-tab:hover{border-color:#555;color:#eee}
.bd-preview-tab.active{background:#1a3a2a;color:#4fff8f;border-color:#4fff8f}
.bd-preview-col>.bd-label{color:#eaeaea;font-size:11px;font-weight:700;letter-spacing:.02em;flex-shrink:0}
.bd-preview-col .bd-batch-preview{flex:1 1 auto;width:100%;max-width:none;min-height:0;justify-self:stretch;border:0;background:transparent;padding:0;border-radius:0;overflow:hidden}
.bd-preview-col.exec-open .bd-batch-preview{display:none}
.bd-exec-prompt-wrap{display:none;flex:1 1 auto;min-height:0;flex-direction:column;gap:4px;min-width:0}
.bd-preview-col.exec-open .bd-exec-prompt-wrap{display:flex}
.bd-exec-prompt{flex:1 1 auto;min-height:0;overflow:auto;background:#101010;border:1px solid #2e2e2e;border-radius:8px;padding:8px 10px;color:#c8c8c8;font-size:11px;line-height:1.45;white-space:pre-wrap;word-break:break-word}
.bd-exec-prompt:empty::before{content:attr(data-empty);color:#555}
.bd-wrap:not(.bd-live-preview-on) .bd-preview-col{display:none!important}
.bd-batch-plain .bd-batch-head,.bd-batch-source .bd-batch-head,.bd-batch-fl2v .bd-batch-head,.bd-batch-v2v .bd-batch-head,.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-head{padding-bottom:2px;border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:2px}
.bd-batch-plain .bd-batch-prompts,.bd-batch-source .bd-batch-prompts,.bd-batch-fl2v .bd-batch-prompts,.bd-batch-v2v .bd-batch-prompts,.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-prompts,.bd-batch-r2v .bd-batch-prompts{background:#0c0c0c;border:1px solid #262626;border-radius:10px;padding:10px 12px;gap:6px;height:100%;min-height:0;align-self:stretch;display:flex;flex-direction:column;overflow:hidden}
.bd-batch-plain .bd-batch-prompts .bd-label,.bd-batch-source .bd-batch-prompts .bd-label,.bd-batch-fl2v .bd-batch-prompts .bd-label,.bd-batch-v2v .bd-batch-prompts .bd-label,.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-prompts .bd-label,.bd-batch-r2v .bd-batch-prompts .bd-label{color:#eaeaea;font-size:11px;font-weight:700;letter-spacing:.02em;flex-shrink:0}
.bd-batch-plain .bd-batch-prompts textarea,.bd-batch-source .bd-batch-prompts textarea,.bd-batch-fl2v .bd-batch-prompts textarea,.bd-batch-v2v .bd-batch-prompts textarea,.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-prompts textarea,.bd-batch-r2v .bd-batch-prompts textarea{background:#101010;border-color:#2e2e2e;border-radius:8px;padding:10px;font-size:12px;line-height:1.45}
.bd-batch-plain .bd-batch-preview,.bd-batch-source .bd-batch-preview,.bd-batch-fl2v .bd-batch-preview,.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-preview{border-radius:10px;border-color:#262626;background:#0c0c0c}
/* ——— r2v asset stage (polished) ——— */
.bd-batch-card.bd-batch-r2v{display:flex;flex-direction:column;align-items:stretch;min-height:490px}
.bd-batch-card.bd-batch-collapsed{display:flex;flex:0 0 auto!important;flex-direction:column;height:auto!important;min-height:0;resize:none}
.bd-batch-card.bd-batch-collapsed>:not(.bd-batch-head){display:none!important}
.bd-batch-card.running{border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.25)}
.bd-batch-card.done{border-color:#3a5080}
.bd-batch-card.run-skipped{opacity:.42}
/* selected / run-on must win over .done so timeline ↔ card selection stays visible */
.bd-batch-card.selected,.bd-batch-card.selected.done{border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.35)}
.bd-batch-card.run-on:not(.run-skipped){border-color:#3a7a55}
.bd-batch-card.selected.run-on,.bd-batch-card.selected.run-on.done{border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.4)}
.bd-batch-head{grid-column:1/-1;display:flex;align-items:center;align-self:start;justify-content:flex-start;gap:8px;flex-wrap:nowrap;flex:0 0 auto;width:100%;min-width:0}
.bd-batch-r2v .bd-batch-head{padding-bottom:2px;border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:2px;flex-shrink:0}
.bd-batch-head b{color:#f0f0f0;font-size:12px;font-weight:650}
.bd-batch-head>b{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bd-batch-collapse{appearance:none;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;padding:0;border:0;background:transparent;color:#aaa;font-size:13px;line-height:1;cursor:pointer;flex-shrink:0}
.bd-batch-collapse:hover,.bd-batch-collapse:focus-visible{color:#eee;background:#252525;border-radius:4px;outline:none}
.bd-batch-title-input{min-width:100px;max-width:220px;height:24px;box-sizing:border-box;background:#181818;border:1px solid #4a7a5a;border-radius:4px;color:#eee;padding:2px 6px;font:inherit;font-size:12px}
.bd-batch-run-check{width:14px;height:14px;margin:0;cursor:pointer;accent-color:#4fff8f;flex-shrink:0}
.bd-batch-continuity{display:inline-flex;align-items:center;gap:4px;font-size:11px;color:#9ab;cursor:pointer;user-select:none;flex-shrink:0}
.bd-batch-continuity input{width:14px;height:14px;margin:0;cursor:pointer;accent-color:#6ab0ff;flex-shrink:0}
.bd-batch-continuity span{white-space:nowrap}
.bd-batch-head-meta{display:flex;align-items:center;gap:8px;flex:0 0 auto;flex-wrap:nowrap;margin-left:auto}
.bd-batch-seed{display:inline-flex;align-items:center;gap:4px;color:#aaa;font-size:11px;white-space:nowrap}
.bd-batch-seed select,.bd-batch-seed input{box-sizing:border-box;height:25px;background:#181818;border:1px solid #444;border-radius:5px;color:#eee;padding:3px 6px;font-size:11px}
.bd-batch-seed select{max-width:74px}
.bd-batch-seed input{width:116px;font-variant-numeric:tabular-nums}
.bd-batch-fc{display:flex;align-items:center;gap:6px;color:#aaa;font-size:12px}
.bd-batch-fc input{width:72px;background:#181818;border:1px solid #444;border-radius:5px;color:#eee;padding:5px 8px;font-size:13px}
.bd-batch-pass{display:inline-flex;align-items:center;gap:4px;flex-shrink:0}
.bd-batch-force-resample{display:inline-flex;align-items:center;gap:3px;color:#aaa;font-size:10px;white-space:nowrap;cursor:pointer}
.bd-batch-force-resample input{width:12px;height:12px;margin:0;accent-color:#4fff8f;cursor:pointer}
.bd-batch-pass-btn{background:#181818;border:1px solid #444;color:#ccc;border-radius:5px;padding:3px 7px;font-size:11px;cursor:pointer;line-height:1.2}
.bd-batch-pass-btn.active{border-color:#4fff8f;color:#4fff8f;background:#163022}
.bd-batch-pass-btn:disabled{opacity:.4;cursor:not-allowed}
.bd-batch-pass-status{appearance:none;-webkit-appearance:none;display:inline-block;box-sizing:border-box;width:12px;height:12px;min-width:12px;min-height:12px;padding:0;margin:0;border:1px solid #2a2a2a;border-radius:50%;line-height:0;font-size:0;overflow:hidden;vertical-align:middle;cursor:pointer;flex-shrink:0;background:#666}
.bd-batch-pass-status.missing{background:#aaa}
.bd-batch-pass-status.valid{background:#65d68a}
.bd-batch-pass-status.mismatch{background:#f0bd58}
.bd-batch-pass-status.error{background:#ef7777}
.bd-batch-pass-status.pending{background:#666}
.bd-batch-pass-status.unchecked{background:#444}
.bd-batch-pass-status.checking{background:#55aaff;box-shadow:0 0 0 0 rgba(85,170,255,.65);animation:bd-pass-checking 1.1s ease-in-out infinite}
@keyframes bd-pass-checking{50%{box-shadow:0 0 0 4px rgba(85,170,255,0)}}
.bd-batch-pass-clear{background:transparent;border:1px solid #553;color:#f88;border-radius:4px;padding:3px 6px;font-size:10px;cursor:pointer}
.bd-batch-pass-clear:hover{background:#3a1515}
.bd-batch-pass-pop{position:fixed;z-index:10000;max-width:300px;padding:8px 10px;background:#1a1a1a;border:1px solid #444;border-radius:8px;color:#ddd;font:12px/1.45 sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.45);white-space:pre-wrap;word-break:break-word}
.bd-batch-refsize{display:flex;align-items:center;gap:6px;color:#aaa;font-size:12px;white-space:nowrap}
.bd-batch-refsize select{background:#161616;border:1px solid #3a3a3a;border-radius:6px;color:#eee;padding:5px 6px;font-size:12px;max-width:132px}
.bd-batch-del{background:transparent;border:1px solid #553;color:#f88;border-radius:4px;padding:3px 8px;font-size:10px;cursor:pointer}
.bd-batch-del:disabled{border-color:#3a3a3a;color:#777;opacity:.55;cursor:not-allowed}
.bd-batch-del:not(:disabled):hover{background:#3a1515}
.bd-batch-media{display:flex;flex-direction:column;gap:4px;min-width:88px;max-width:140px}
.bd-batch-mixed.bd-batch-source>.bd-batch-media{width:100%;max-width:none;min-width:0}
.bd-batch-mixed.bd-batch-source>.bd-batch-media>.bd-batch-src{width:100%;height:100%;min-height:88px}

/* Left = 参考素材 · Middle = 提示词 · Right = 预览 */
.bd-batch-r2v-body{display:grid;grid-template-columns:minmax(240px,.85fr) minmax(0,1.4fr);gap:12px;width:100%;align-items:stretch;min-height:420px;flex:1 1 auto;overflow:hidden}
.bd-wrap.bd-live-preview-on .bd-batch-r2v-body{grid-template-columns:minmax(220px,.8fr) minmax(0,1.3fr) minmax(200px,.7fr)}
.bd-batch-mixed .bd-batch-r2v-body{grid-template-columns:minmax(0,1fr) minmax(0,1.3fr)}
.bd-wrap.bd-live-preview-on .bd-batch-mixed .bd-batch-r2v-body{grid-template-columns:minmax(0,1fr) minmax(0,1.3fr) minmax(0,1fr)}
.bd-batch-r2v-body>.bd-batch-r2v-main,.bd-batch-r2v-body>.bd-preview-col{height:0;min-height:100%;max-height:100%;overflow:hidden;align-self:stretch}
.bd-batch-r2v-assets{display:flex;flex-direction:column;gap:10px;min-width:0;min-height:0;overflow:auto}
.bd-batch-r2v-main{display:flex;flex-direction:column;gap:6px;min-width:0}
.bd-r2v-section{background:#0c0c0c;border:1px solid #262626;border-radius:10px;padding:10px 12px;display:flex;flex-direction:column;gap:8px;min-width:0;box-sizing:border-box}
.bd-r2v-section-head{display:flex;align-items:center;justify-content:space-between;gap:8px}
.bd-r2v-section-title{font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#eaeaea;min-width:0}
.bd-r2v-section-actions{display:flex;align-items:center;gap:8px;flex-shrink:0}
.bd-r2v-section-count{font-size:11px;color:#7d7d7d;font-variant-numeric:tabular-nums;letter-spacing:.02em}
.bd-r2v-section-actions .bd-batch-refsize{font-size:11px;gap:4px}
.bd-r2v-section-actions .bd-batch-refsize select{padding:2px 5px;font-size:11px;max-width:132px}

.bd-batch-src{position:relative;width:88px;height:88px;border:1px dashed #555;border-radius:4px;background:#111;display:flex;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;color:#666;font-size:9px;text-align:center;padding:4px;box-sizing:border-box}
.bd-batch-src .x{position:absolute;top:1px;right:1px;width:18px;height:18px;border-radius:4px;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.78);color:#ff8a8a;font-size:14px;font-weight:700;z-index:3;line-height:1}
.bd-batch-src.has-img:hover .x{display:flex}
.bd-batch-src.has-img{border-style:solid;border-color:#444;padding:0}
.bd-batch-src img{width:100%;height:100%;object-fit:contain;background:#000}
.bd-batch-refs{display:grid;grid-template-columns:repeat(3,1fr);gap:3px;width:108px}
.bd-batch-r2v .bd-batch-refs{grid-template-columns:repeat(3,minmax(0,1fr));width:100%;max-width:none;gap:6px}
.bd-batch-r2v .bd-batch-ref.bd-r2v-pic-hidden{display:none!important}
.bd-r2v-pics-toggle{align-self:stretch;margin-top:2px;background:transparent;border:1px dashed #333;border-radius:8px;color:#9a9a9a;font-size:11px;padding:6px 8px;cursor:pointer;transition:border-color .15s,color .15s,background .15s}
.bd-r2v-pics-toggle:hover{border-color:#555;color:#ddd;background:#121212}
.bd-batch-ref{position:relative;aspect-ratio:1;border:1px dashed #555;border-radius:3px;background:#111;display:flex;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;font-size:8px;color:#666}
.bd-batch-r2v .bd-batch-ref{aspect-ratio:1;min-height:0;border-radius:8px;border:1px dashed #333;background:#080808;color:#555;font-size:10px;transition:border-color .15s,background .15s,transform .12s}
.bd-batch-r2v .bd-batch-ref:hover{border-color:#5a5a5a;background:#101010}
.bd-batch-ref.has-img{border-style:solid}
.bd-batch-r2v .bd-batch-ref.has-img{border-color:#3a3a3a;background:#000}
.bd-batch-ref img{width:100%;height:100%;object-fit:cover}
.bd-batch-r2v .bd-batch-ref img{width:100%;height:100%;object-fit:contain;object-position:center;background:#000}
.bd-batch-r2v .bd-batch-ref .dot{position:absolute;left:6px;top:6px;width:7px;height:7px;border-radius:50%;background:#4fff8f;box-shadow:0 0 0 2px rgba(0,0,0,.5);z-index:2}
.bd-batch-r2v .bd-batch-ref .cap{position:absolute;left:0;right:0;bottom:0;padding:14px 6px 5px;background:linear-gradient(180deg,transparent,rgba(0,0,0,.78));color:#ddd;font-size:10px;font-weight:600;text-align:center;pointer-events:none;z-index:2}
.bd-batch-r2v .bd-batch-ref:not(.has-img) .cap{position:static;padding:0;background:none;color:#666;font-weight:500}
.bd-batch-ref .x{position:absolute;top:0;right:2px;color:#f88;font-size:10px;display:none;line-height:1}
.bd-batch-r2v .bd-batch-ref .x{top:4px;right:4px;width:20px;height:20px;border-radius:6px;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.72);color:#ff9a9a;font-size:14px;font-weight:700;z-index:3}
.bd-batch-ref:hover .x{display:block}
.bd-batch-r2v .bd-batch-ref:hover .x,.bd-batch-r2v .bd-batch-ref:focus-within .x{display:flex}
.bd-batch-media-block{display:flex;flex-direction:column;gap:4px;min-width:0}
.bd-batch-media-block .bd-label{color:#888;font-size:10px}
.bd-batch-audios,.bd-batch-videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px;width:100%;max-width:420px}
.bd-batch-r2v .bd-batch-videos,.bd-batch-r2v .bd-batch-audios{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));max-width:none;gap:7px;width:100%}
.bd-batch-audio,.bd-batch-video{position:relative;min-height:44px;border:1px dashed #555;border-radius:4px;background:#111;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;cursor:pointer;padding:6px 4px;box-sizing:border-box;font-size:9px;color:#666;text-align:center;line-height:1.25}
.bd-batch-r2v .bd-batch-audio,.bd-batch-r2v .bd-batch-video{min-height:0;height:auto;flex-direction:column;align-items:stretch;justify-content:flex-start;gap:6px;padding:6px;border-radius:8px;border:1px dashed #333;background:#080808;text-align:left;font-size:11px;color:#777;transition:border-color .15s,background .15s}
.bd-batch-r2v .bd-batch-audio:hover,.bd-batch-r2v .bd-batch-video:hover{border-color:#555;background:#101010}
.bd-batch-audio.has-audio,.bd-batch-video.has-video{border-style:solid;border-color:#4a6a4a;color:#cfe;background:#152015}
.bd-batch-r2v .bd-batch-audio.has-audio,.bd-batch-r2v .bd-batch-video.has-video{border-color:#2f4a38;background:#101812;color:#d8ebe0}
.bd-batch-audio:hover,.bd-batch-video:hover{border-color:#7a9cff}
.bd-r2v-thumb{position:relative;width:38px;height:38px;border-radius:7px;background:#1a1a1a;border:1px solid #2e2e2e;flex-shrink:0;display:flex;align-items:center;justify-content:center;color:#666;font-size:14px;overflow:hidden}
.bd-batch-r2v .bd-batch-video .bd-r2v-thumb,.bd-r2v-thumb-video{width:100%;height:auto;aspect-ratio:16/9;border-radius:6px}
.bd-batch-r2v .bd-batch-audio .bd-r2v-thumb{width:100%;height:44px;border-radius:6px}
.bd-r2v-thumb-video video{width:100%;height:100%;object-fit:cover;display:block;background:#000;pointer-events:none}
.bd-r2v-play{position:absolute;inset:0;margin:auto;width:28px;height:28px;border:0;border-radius:50%;background:rgba(0,0,0,.62);color:#fff;font-size:12px;line-height:1;cursor:pointer;display:flex;align-items:center;justify-content:center;padding:0;z-index:2}
.bd-r2v-play:hover{background:rgba(20,20,20,.82);color:#4fff8f}
.bd-batch-r2v .has-audio .bd-r2v-thumb,.bd-batch-r2v .has-video .bd-r2v-thumb{border-color:#3a5a45;color:#8fdfb0;background:#152018}
.bd-r2v-meta{min-width:0;flex:1;display:flex;flex-direction:column;gap:2px}
.bd-media-file-name{display:none!important}
.bd-media-file-name.can-scroll:hover{text-overflow:clip;animation:bd-media-file-marquee var(--bd-file-scroll,2.8s) linear infinite alternate}
@keyframes bd-media-file-marquee{from{transform:translateX(0)}to{transform:translateX(calc(-1 * var(--bd-file-overflow,0px)))}}
.bd-batch-r2v .bd-batch-video .bd-r2v-meta,.bd-batch-r2v .bd-batch-audio .bd-r2v-meta{flex-direction:row;align-items:center;justify-content:space-between;gap:4px}
.bd-r2v-meta .tag{color:#cfcfcf;font-size:11px;font-weight:650}
.bd-r2v-dur{flex-shrink:0;min-width:2.6em;text-align:right;font-size:11px;color:#8a9;font-variant-numeric:tabular-nums}
.bd-r2v-paired-audio{flex-shrink:0;color:#7dbaff;font-size:12px;line-height:1;padding:0 2px}
.bd-batch-r2v .bd-batch-audio .name,.bd-batch-r2v .bd-batch-video .name{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#8aa;font-size:10px;padding:0}
.bd-batch-r2v .bd-batch-video.has-video .name,.bd-batch-r2v .bd-batch-audio.has-audio .name{display:none}
.bd-batch-r2v .bd-batch-video:not(.has-video) .name,.bd-batch-r2v .bd-batch-audio:not(.has-audio) .name{display:block;color:#666}
.bd-batch-r2v .bd-batch-audio audio.bd-r2v-media{position:absolute;width:0;height:0;opacity:0;pointer-events:none}
.bd-r2v-progress{display:none;width:100%;height:3px;border-radius:99px;background:#222;overflow:hidden;cursor:pointer}
.bd-r2v-progress.active{display:block}
.bd-r2v-progress-fill{height:100%;width:0;background:linear-gradient(90deg,#2a6b4a,#4fff8f);border-radius:99px;transition:width .08s linear}
.bd-r2v-progress.playing .bd-r2v-progress-fill{transition:none}
.bd-batch-audio .name,.bd-batch-video .name{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#9ad;font-size:9px;padding:0 2px}
.bd-batch-audio .x,.bd-batch-video .x{position:absolute;top:1px;right:3px;color:#f88;font-size:12px;display:none;line-height:1}
.bd-batch-r2v .bd-batch-audio .x,.bd-batch-r2v .bd-batch-video .x{position:absolute;top:8px;right:8px;width:20px;height:20px;border-radius:6px;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.72);color:#ff9a9a;font-size:14px;font-weight:700;z-index:3}
.bd-batch-audio:hover .x,.bd-batch-video:hover .x{display:block}
.bd-batch-r2v .bd-batch-audio:hover .x,.bd-batch-r2v .bd-batch-video:hover .x{display:flex}
.bd-batch-prompts{display:flex;flex-direction:column;gap:4px;min-width:0}
.bd-batch-prompts .bd-label{color:#888;font-size:10px}
.bd-batch-prompts textarea,.bd-batch-prompts .bd-token-wrap{width:100%;min-height:88px;box-sizing:border-box}
.bd-batch-prompts textarea{background:#181818;border:1px solid #333;border-radius:4px;color:#eee;padding:6px;resize:none;overflow-y:auto;font-size:11px;font-family:inherit;line-height:1.35}
.bd-batch-plain .bd-batch-prompts textarea,.bd-batch-source .bd-batch-prompts textarea,.bd-batch-fl2v .bd-batch-prompts textarea,.bd-batch-v2v .bd-batch-prompts textarea,.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-prompts textarea,.bd-batch-r2v .bd-batch-prompts textarea,
.bd-batch-plain .bd-batch-prompts .bd-token-wrap,.bd-batch-source .bd-batch-prompts .bd-token-wrap,.bd-batch-fl2v .bd-batch-prompts .bd-token-wrap,.bd-batch-v2v .bd-batch-prompts .bd-token-wrap,.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-prompts .bd-token-wrap,.bd-batch-r2v .bd-batch-prompts .bd-token-wrap{flex:1 1 auto;min-height:88px;height:auto;overflow:auto}
.bd-batch-r2v .bd-batch-prompts textarea{background:#101010;border-color:#2e2e2e;border-radius:8px;padding:10px;font-size:12px;line-height:1.45}
.bd-batch-preview{background:#0d0d0d;border:1px solid #333;border-radius:4px;min-height:100px;display:flex;flex-direction:column;align-items:stretch;justify-content:center;overflow:hidden;color:#555;font-size:10px;text-align:center;padding:4px;box-sizing:border-box}
.bd-batch-plain .bd-preview-col .bd-batch-preview,.bd-batch-source .bd-preview-col .bd-batch-preview,.bd-batch-refs:not(.bd-batch-r2v) .bd-preview-col .bd-batch-preview{width:100%;max-width:none;min-height:0;justify-self:stretch}
.bd-batch-r2v .bd-preview-col .bd-batch-preview{min-height:0;max-height:none;height:auto;flex:1 1 auto}
.bd-batch-preview img{width:100%;max-width:100%;max-height:200px;object-fit:contain;display:block;margin:0 auto}
.bd-preview-col .bd-batch-preview img,.bd-preview-col .bd-batch-live-preview img{max-height:100%;height:100%}
.bd-batch-plain .bd-batch-preview img,.bd-batch-source .bd-batch-preview img{max-height:180px}
.bd-preview-col .bd-batch-preview img,.bd-preview-col .bd-batch-live-preview img{max-height:100%!important;height:100%;width:auto;margin:0 auto}
.bd-batch-r2v .bd-batch-preview img{width:100%;max-height:280px}
.bd-batch-vpreview{width:100%;height:100%;display:flex;flex-direction:column;align-items:stretch;gap:4px;min-height:0}
.bd-batch-vpreview canvas{width:100%;flex:1 1 auto;min-height:96px;max-height:200px;background:#000;border-radius:3px;display:block;object-fit:contain}
.bd-batch-r2v .bd-batch-vpreview canvas{border-radius:8px;max-height:280px;min-height:160px}
.bd-preview-col .bd-batch-vpreview{flex:1 1 auto;min-height:0;height:100%}
.bd-batch-plain .bd-batch-vpreview canvas,.bd-batch-source .bd-batch-vpreview canvas{max-height:180px;min-height:96px}
.bd-preview-col .bd-batch-vpreview canvas,
.bd-batch-plain .bd-preview-col .bd-batch-vpreview canvas,
.bd-batch-source .bd-preview-col .bd-batch-vpreview canvas,
.bd-batch-v2v .bd-preview-col .bd-batch-vpreview canvas,
.bd-batch-r2v .bd-preview-col .bd-batch-vpreview canvas{max-height:none;min-height:0;flex:1 1 auto;height:100%}
.bd-preview-col .bd-batch-preview,
.bd-preview-col .bd-batch-vpreview,
.bd-preview-col .bd-batch-live-preview{min-height:0}
.bd-batch-vpreview-ctrl{display:flex;align-items:center;justify-content:center;gap:6px;flex-shrink:0}
.bd-batch-vpreview-ctrl button{font-size:10px;padding:2px 8px}
.bd-batch-vpreview-meta{color:#666;font-size:9px;text-align:center;flex-shrink:0}
.bd-batch-live-preview{position:relative;width:100%;height:100%;min-height:0;flex:1 1 auto;display:flex;align-items:center;justify-content:center;box-sizing:border-box}
.bd-batch-r2v .bd-batch-live-preview{min-height:0}
.bd-batch-live-preview img{width:100%;height:100%;max-width:100%;max-height:100%;object-fit:contain;display:block;border-radius:6px}
.bd-batch-r2v .bd-batch-live-preview img{max-height:100%}
.bd-batch-live-badge{position:absolute;left:8px;bottom:8px;padding:2px 7px;border-radius:999px;background:rgba(0,0,0,.72);color:#cfcfcf;font-size:10px;pointer-events:none}
@media(max-width:860px){
.bd-batch-r2v-body,.bd-batch-r2v-foot,.bd-wrap.bd-live-preview-on .bd-batch-r2v-body{grid-template-columns:1fr}
.bd-batch-card.bd-batch-mixed.bd-batch-r2v .bd-batch-r2v-body,
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-mixed.bd-batch-r2v .bd-batch-r2v-body{grid-template-columns:1fr}
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-plain{grid-template-columns:minmax(0,1fr) minmax(140px,180px)}
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-source,
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-fl2v,
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-refs:not(.bd-batch-r2v){grid-template-columns:auto minmax(0,1fr) minmax(140px,180px)}
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-v2v{grid-template-columns:minmax(160px,220px) minmax(0,1fr) minmax(140px,180px)}
}
@media(max-width:720px){
.bd-batch-card,.bd-batch-card.bd-batch-plain,.bd-batch-card.bd-batch-source,.bd-batch-card.bd-batch-fl2v,.bd-batch-card.bd-batch-v2v,.bd-batch-card.bd-batch-refs:not(.bd-batch-r2v),
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-plain,
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-source,
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-fl2v,
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-v2v,
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-refs:not(.bd-batch-r2v){grid-template-columns:1fr}
.bd-batch-card.bd-batch-mixed:not(.bd-batch-r2v),
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-mixed:not(.bd-batch-r2v),
.bd-batch-card.bd-batch-mixed.bd-batch-plain,
.bd-wrap.bd-live-preview-on .bd-batch-card.bd-batch-mixed.bd-batch-plain{grid-template-columns:1fr}
.bd-batch-r2v .bd-batch-refs{grid-template-columns:repeat(3,minmax(0,1fr))}
}
` + SLOT_UI_STYLES;

const BATCH_CHUNK_SIZE = 8 * 1024 * 1024;
const BATCH_UPLOAD_SOFT_LIMIT = 95 * 1024 * 1024;

async function uploadImage(file) {
    const uploadFile = fileForComfyUpload(file);
    const body = new FormData();
    body.append("image", uploadFile, uploadFile.name);
    body.append("type", "input");
    body.append("subfolder", "H3_D_NEO/references");
    body.append("overwrite", "false");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!resp.ok) throw new Error(await resp.text() || `Upload failed (${resp.status})`);
    return resp.json();
}

async function uploadChunked(file, onProgress) {
    const filename = safeUploadFilename(file?.name, file?.type);
    const uploadId = crypto.randomUUID();
    const totalChunks = Math.ceil(file.size / BATCH_CHUNK_SIZE);
    for (let i = 0; i < totalChunks; i++) {
        onProgress?.(i / totalChunks, i, totalChunks);
        const start = i * BATCH_CHUNK_SIZE;
        const end = Math.min(start + BATCH_CHUNK_SIZE, file.size);
        const body = new FormData();
        body.append("upload_id", uploadId);
        body.append("chunk_index", String(i));
        body.append("total_chunks", String(totalChunks));
        body.append("filename", filename);
        body.append("chunk", file.slice(start, end), `${filename}.part`);
        const resp = await api.fetchApi("/minimax/director/upload_chunk", { method: "POST", body });
        if (!resp.ok) throw new Error(await resp.text() || t("upload.chunkFailed", { status: resp.status }));
        const data = await resp.json();
        onProgress?.((i + 1) / totalChunks, i + 1, totalChunks);
        if (data.name) return data;
    }
    throw new Error(t("upload.chunkIncomplete"));
}

async function uploadMedia(file, onProgress) {
    if (file.size <= BATCH_UPLOAD_SOFT_LIMIT) {
        try {
            onProgress?.(0, 0, 1);
            const uploaded = await uploadImage(file);
            onProgress?.(1, 1, 1);
            return uploaded;
        } catch (err) {
            const msg = String(err?.message || err || "");
            if (!/too large|size|413/i.test(msg)) throw err;
        }
    }
    return uploadChunked(file, onProgress);
}

function relPath(upload) {
    const name = upload.name || upload.filename;
    const sub = (upload.subfolder || "").replace(/\\/g, "/").replace(/\/$/, "");
    return sub ? `${sub}/${name}` : name;
}

function viewUrl(imageFile) {
    const norm = String(imageFile || "").replace(/\\/g, "/");
    const slash = norm.lastIndexOf("/");
    const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
    const subfolder = slash >= 0 ? norm.slice(0, slash) : "";
    const params = new URLSearchParams({ filename, type: "input" });
    if (subfolder) params.set("subfolder", subfolder);
    return api.apiURL(`/view?${params.toString()}`);
}

export function mountImageBatchPanel(root) {
    const panel = document.createElement("div");
    panel.className = "bd-batch hidden";
    panel.dataset.r = "batch-panel";
    panel.innerHTML = `
        <div class="bd-batch-toolbar">
            <button type="button" class="bd-btn bd-btn-primary hidden" data-a="batch-add" data-i18n="toolbar.addShot">添加一组</button>
            <button type="button" class="bd-btn bd-batch-run-select hidden" data-a="batch-run-select" data-i18n="toolbar.runSelect" data-i18n-title="tooltip.batchRunSelect">选择运行</button>
            <label class="bd-batch-run-all hidden" data-r="batch-run-all-wrap" data-i18n-title="tooltip.runSelectAll">
                <input type="checkbox" data-r="batch-run-all-cb">
                <span data-i18n="toolbar.selectAll">全选</span>
            </label>
            <button type="button" class="bd-btn" data-a="batch-detail-mode" data-i18n="toolbar.batchDetailSolo" data-i18n-title="tooltip.batchDetailSolo">单显模式</button>
            <span class="bd-meta" data-r="batch-hint" data-i18n="batch.hint.defaultImage">每组生成 1 张图片</span>
        </div>
        <div class="bd-batch-picker" data-r="batch-picker"></div>
        <div class="bd-batch-list" data-r="batch-list"></div>`;
    root.appendChild(panel);
    return {
        panel,
        list: panel.querySelector('[data-r="batch-list"]'),
        picker: panel.querySelector('[data-r="batch-picker"]'),
        hint: panel.querySelector('[data-r="batch-hint"]'),
        addBtn: panel.querySelector('[data-a="batch-add"]'),
        runSelectBtn: panel.querySelector('[data-a="batch-run-select"]'),
        runSelectAllWrap: panel.querySelector('[data-r="batch-run-all-wrap"]'),
        runSelectAllCb: panel.querySelector('[data-r="batch-run-all-cb"]'),
        detailModeBtn: panel.querySelector('[data-a="batch-detail-mode"]'),
    };
}

export function wireBatchRunSelectControls(editor, batchUi) {
    editor.batchRunSelectBtn = batchUi.runSelectBtn;
    editor.batchRunSelectAllWrap = batchUi.runSelectAllWrap;
    editor.batchRunSelectAllCb = batchUi.runSelectAllCb;
    editor.batchDetailModeBtn = batchUi.detailModeBtn;
    editor.batchPicker = batchUi.picker;
    batchUi.runSelectBtn?.addEventListener("click", (e) => {
        e.stopPropagation();
        editor.toggleRunSelectMode?.();
    });
    batchUi.detailModeBtn?.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleBatchDetailMode(editor);
    });
    batchUi.runSelectAllCb?.addEventListener("change", (e) => {
        e.stopPropagation();
        if (!editor.isRunSelectEnabled?.()) return;
        editor.setRunSelectionAll?.(batchUi.runSelectAllCb.checked);
    });
}

function cloneRefs(refs) {
    if (!Array.isArray(refs) || !refs.length) return [];
    try {
        return JSON.parse(JSON.stringify(refs));
    } catch {
        return refs.map((r) => ({ ...r }));
    }
}

function cloneValue(value) {
    if (value == null) return value;
    try { return JSON.parse(JSON.stringify(value)); } catch { return value; }
}

function sourceSelectionFrameCount(sourceVideo) {
    const source = sourceVideo || {};
    if (source.mediaKind === "image") {
        return Math.max(0, Math.round(Number(source.rangeEnd) || Number(source.totalFrames) || 0));
    }
    return sourceVideoFrameMap(source).length;
}

function syncSourceMaterialFromFirstGroup(editor, index) {
    const segments = editor?.timeline?.segments;
    const target = segments?.[index];
    const first = segments?.[0];
    if (!target || !first?.sourceVideo) return false;
    target.sourceVideo = cloneValue(first.sourceVideo);
    target.previewB64 = "";
    target.previewFrames = [];
    editor.renderImageBatchGroups?.();
    editor.commit?.(false, { syncTimeline: true });
    editor.updateVideoNameLabel?.();
    editor.scheduleRender?.();
    return true;
}

function syncSegmentDurationFromSource(editor, index) {
    const live = editor?.timeline?.segments?.[index];
    const count = sourceSelectionFrameCount(live?.sourceVideo);
    if (!live || count <= 0) return false;
    const updated = applyBatchSegmentDuration(editor, index, preferredDurationSecFromFrames(count, 24));
    if (!updated) return false;
    // flushTimelineSync reads the visible 秒数 input. Update it first so the
    // old typed value cannot overwrite the range sync.
    syncSourceRangeDurationInput(editor, updated, index, updated.frameCount);
    editor.commit?.(false, { syncTimeline: true });
    editor.flushTimelineSync?.();
    editor.updateVideoNameLabel?.();
    editor.renderImageBatchGroups?.();
    return true;
}

/** Copy global.refs into batch segments that have no refs (r2i only). */
export function migrateGlobalRefsIntoBatchSegments(editor, taskKey) {
    const key = resolveTaskKey(taskKey || editor.getTaskKey?.() || "");
    if (key !== "r2i") return false;
    const globalRefs = editor.timeline?.global?.refs;
    if (!Array.isArray(globalRefs) || !globalRefs.length) return false;
    let moved = false;
    for (const seg of editor.timeline.segments || []) {
        if ((seg.refs || []).length) continue;
        seg.refs = cloneRefs(globalRefs);
        moved = true;
    }
    return moved;
}

export function ensureImageBatchTimeline(editor) {
    editor.timeline.editMode = "segment";
    editor.timeline.output = editor.timeline.output || {};
    const taskKey = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    editor.timeline.output.mode = "fixed";
    if (!editor.timeline.output.aspectRatio) editor.timeline.output.aspectRatio = DEFAULT_ASPECT_RATIO;
    if (editor.timeline.output.megapixels == null) editor.timeline.output.megapixels = DEFAULT_MEGAPIXELS;
    if (editor.timeline.output.multiple == null) editor.timeline.output.multiple = MINIMAX_CANVAS_MULTIPLE;
    if (!isVideoBatchTask(taskKey)) {
        editor.timeline.output.exportMode = "all";
    }
    const defFc = defaultFrameCount(taskKey);
    if (taskKey === "i2v") {
        editor.timeline.video = {
            fileName: "",
            videoFile: "",
            subfolder: "",
            type: "input",
            frames: [],
            frameMap: [],
        };
        editor.timeline.videoClips = [];
    }
    if (!editor.timeline.segments?.length) {
        editor.timeline.segments = [newBatchSegment({
            durationSec: defaultDurationSec(taskKey === "mixed" ? "t2v" : taskKey),
            taskType: taskKey === "mixed" ? "t2v" : "",
            passMode: defaultBatchPassMode(editor.node),
        })];
    }
    // r2i/r2v need per-group refs. If the user came from rv2v (global refs) or left
    // refs only on global, copy them into empty batch groups so generation actually
    // receives reference_image_* — otherwise it silently behaves like t2v/t2i.
    migrateGlobalRefsIntoBatchSegments(editor, taskKey);
    for (const seg of editor.timeline.segments) {
        if (isVideoBatchTask(taskKey)) {
            const { frames, durationSec } = resolveVideoSegmentDuration(taskKey, seg);
            seg.durationSec = durationSec;
            seg.frameCount = frames;
            seg.length = frames;
            seg._videoFrameCount = frames;
        } else {
            const prevFc = parseInt(seg.frameCount ?? seg.length, 10) || 0;
            if (prevFc > 1) seg._videoFrameCount = prevFc;
            seg.frameCount = 1;
            seg.length = 1;
        }
        seg.negativePrompt = seg.negativePrompt ?? "";
        seg.genImage = seg.genImage || { imageFile: seg.imageFile || "" };
        // Do NOT copy r2v refs into i2v/t2v here — each task keeps its own workspace.
        // Backend ignores refs on i2v/t2v; r2v snapshots restore them on switch-back.
        seg.refs = seg.refs || [];
        seg.refAudios = seg.refAudios || [];
        seg.refVideos = seg.refVideos || [];
        seg.previewB64 = seg.previewB64 || "";
        seg.previewFrames = seg.previewFrames || [];
        seg.previewFps = seg.previewFps || parseFloat(editor.frameRateWidget?.value || 24);
        if (!seg.id) seg.id = newBatchSegment().id;
    }
    normalizeImageBatchSegments(editor);
}

export function normalizeImageBatchSegments(editor) {
    // Keep textarea drafts on the live segment objects before touching them.
    flushBatchPromptInputs(editor);
    const taskKey = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    const isVideo = isVideoBatchTask(taskKey);
    const defFc = defaultFrameCount(taskKey);
    const defSec = defaultDurationSec(taskKey);
    let start = 0;
    const segs = editor.timeline.segments || [];
    // Mutate in place so card closures (prompt oninput) stay attached to the
    // same objects. Replacing with `{ ...seg }` orphans DOM writes and can
    // wipe group 5/6 prompts on the next sync/re-render.
    if (!segs.length) {
        editor.timeline.segments = [newBatchSegment({
            durationSec: defSec,
            passMode: defaultBatchPassMode(editor.node),
        })];
    }
    for (const seg of editor.timeline.segments) {
        let fc = 1;
        let durationSec;
        if (isVideo) {
            const resolved = resolveVideoSegmentDuration(taskKey, seg, resolveSegmentDurationSec(seg) || defSec);
            fc = resolved.frames;
            durationSec = resolved.durationSec;
            seg.durationSec = durationSec;
            seg._videoFrameCount = fc;
        }
        seg.start = start;
        seg.length = fc;
        seg.frameCount = fc;
        seg.negativePrompt = seg.negativePrompt ?? "";
        seg.genImage = seg.genImage || { imageFile: "" };
        seg.refs = seg.refs || [];
        seg.refAudios = seg.refAudios || [];
        seg.refVideos = seg.refVideos || [];
        seg.previewB64 = seg.previewB64 || "";
        seg.previewFrames = seg.previewFrames || [];
        seg.previewFps = seg.previewFps || parseFloat(editor.frameRateWidget?.value || 24);
        seg.seedMode = normalizeSegmentSeedMode(seg.seedMode);
        seg.seed = normalizeSegmentSeed(seg.seed);
        if (taskKey === "r2v") {
            seg.refImageSize = resolveSegmentRefImageSize(seg, editor.timeline?.output);
        }
        if (!seg.id) seg.id = newBatchSegment().id;
        start += fc;
    }
    const out = editor.timeline.segments;
    editor.timeline.totalFrames = start || out[0]?.frameCount || defFc;
}

export function addImageBatchGroup(editor) {
    if (editor.hasExternalI2vGroups?.() || editor.hasExternalR2vGroups?.()) return;
    const taskKey = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    const mixed = taskKey === "mixed";
    const last = (editor.timeline.segments || [])[(editor.timeline.segments || []).length - 1];
    const groupType = mixed ? resolveMixedGroupKey(last) : "";
    editor.timeline.segments.push(newBatchSegment({
        durationSec: defaultDurationSec(groupType || taskKey),
        negativePrompt: "",
        taskType: groupType,
        passMode: defaultBatchPassMode(editor.node),
    }));
    normalizeImageBatchSegments(editor);
    editor.selectedIndex = Math.max(0, editor.timeline.segments.length - 1);
    editor.renderImageBatchGroups();
    editor.syncMixedCommonLayout?.();
    editor.commit();
    editor.updateVideoNameLabel?.();
    editor.resizeNodeForContentMinChange?.();
}

export function deleteImageBatchGroup(editor, index) {
    if (editor.hasExternalI2vGroups?.() || editor.hasExternalR2vGroups?.()) return;
    if (editor.timeline.segments.length <= 1) return;
    // Persist drafts while DOM still matches the current array, then splice.
    flushBatchPromptInputs(editor);
    flushBatchDurationInputs(editor);
    editor.timeline.segments.splice(index, 1);
    normalizeImageBatchSegments(editor);
    editor.selectedIndex = clamp(
        editor.selectedIndex > index ? editor.selectedIndex - 1 : editor.selectedIndex,
        0,
        editor.timeline.segments.length - 1,
    );
    editor.renderImageBatchGroups();
    editor.syncMixedCommonLayout?.();
    editor.commit();
    editor.updateVideoNameLabel?.();
    editor.resizeNodeForContentMinChange?.();
}

function pickFile(accept, onFile) {
    // Keep input in DOM until change/cancel — otherwise some Chromium builds
    // drop the dialog result when the element is GC'd.
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;
    input.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0;pointer-events:none";
    const cleanup = () => {
        input.remove();
    };
    input.onchange = () => {
        const file = input.files?.[0];
        cleanup();
        if (file) onFile(file);
    };
    input.addEventListener("cancel", cleanup);
    document.body.appendChild(input);
    input.click();
}

function isBatchImageFile(file) {
    return !!file && (
        String(file.type || "").startsWith("image/")
        || /\.(jpe?g|png|webp|bmp|gif|tiff?)$/i.test(file.name || "")
    );
}

function isBatchVideoFile(file) {
    return !!file && (
        String(file.type || "").startsWith("video/")
        || /\.(mp4|mov|webm|mkv|avi|m4v|mpg|mpeg|mts|ts)$/i.test(file.name || "")
    );
}

function bindOsFileDrop(el, onFiles) {
    el.addEventListener("dragover", (e) => {
        const types = [...(e.dataTransfer?.types || [])];
        if (isSlotDnD(types)) {
            e.preventDefault();
            e.stopPropagation();
            return;
        }
        e.preventDefault();
        e.stopPropagation();
        e.dataTransfer.dropEffect = "copy";
    });
    el.addEventListener("drop", (e) => {
        const types = [...(e.dataTransfer?.types || [])];
        if (isSlotDnD(types)) {
            e.preventDefault();
            e.stopPropagation();
            return;
        }
        e.preventDefault();
        e.stopPropagation();
        const files = [...(e.dataTransfer?.files || [])];
        if (!files.length) return;
        void onFiles(files, e);
    });
}

function applySegSourceImage(editor, index, imageFile, width = 0, height = 0) {
    const segId = editor.timeline.segments[index]?.id;
    const seg = (editor.timeline.segments || []).find((s) => s.id === segId)
        || editor.timeline.segments[index];
    if (!seg) return;
    seg.genImage = { imageFile, width: width || 0, height: height || 0 };
    seg.imageFile = imageFile;
    editor.renderImageBatchGroups();
    editor.updateOutputPreview?.();
    editor.commit(false, { syncTimeline: true });
    editor.scheduleRender?.();
    editor.scheduleTimelineSync?.();
}

async function assignSegSourceFromFile(editor, index, file) {
    const segId = editor.timeline.segments[index]?.id;
    try {
        if (!isBatchImageFile(file)) throw new Error("Not an image file");
        const uploaded = await uploadImage(file);
        const imageFile = relPath(uploaded);
        if (!imageFile) throw new Error("Upload returned empty filename");
        applySegSourceImage(editor, index, imageFile, 0, 0);
        try {
            const dims = await readImageDimensions(file);
            const live = (editor.timeline.segments || []).find((s) => s.id === segId) || editor.timeline.segments[index];
            if (live?.genImage?.imageFile === imageFile) {
                live.genImage = { imageFile, width: dims.width, height: dims.height };
                editor.updateOutputPreview?.();
                editor.scheduleTimelineSync?.();
            }
        } catch (dimErr) {
            console.warn("[MiniMax H3Director] batch source dims skipped:", dimErr);
        }
    } catch (err) {
        console.error("[MiniMax H3Director] batch source upload failed:", err);
        void editor.showBdMessage?.(t("snapshot.errorTitle"), t("upload.alertFailed", { err: err?.message || err }));
    }
}

async function uploadSegSource(editor, index) {
    pickFile("image/*,.jpg,.jpeg,.png,.webp,.bmp,.gif", (file) => {
        void assignSegSourceFromFile(editor, index, file);
    });
}

function clearSegSourceImage(editor, index) {
    const seg = editor.timeline.segments[index];
    if (!seg) return;
    if (seg.genImage) seg.genImage = { imageFile: "" };
    seg.imageFile = "";
    editor.renderImageBatchGroups();
    editor.updateOutputPreview?.();
    editor.commit(false, { syncTimeline: true });
    editor.scheduleRender?.();
}

async function assignSegSourceVideoFromFile(editor, index, file) {
    if (!isBatchVideoFile(file)) return false;
    const key = groupSlotKey(editor, index, "source-video", 0);
    beginSlotLoad(editor, key, t("slot.loading.upload"));
    try {
        const uploaded = await uploadMedia(file, (ratio, cur, total) => {
            updateSlotLoad(editor, key, { status: t("slot.loading.upload"), ratio, cur, total });
        });
        const videoFile = relPath(uploaded);
        const prep = await editor._prepareVideoFrames({
            fileName: uploaded?.name || file.name,
            relPath: videoFile,
            subfolder: uploaded?.subfolder || "",
            type: uploaded?.type || "input",
            statusPrefix: t("parse.prefix"),
        });
        const clip = editor._buildClipRecord(prep);
        const seg = editor.timeline.segments[index];
        if (!seg) return false;
        seg.sourceVideo = {
            mediaKind: "video",
            totalFrames: prep.totalFrames,
            video: { ...clip, frameMap: [], deletedSourceRanges: [] },
            videoClips: [clip],
        };
        const frameCount = floorMiniMaxFrameCount(prep.totalFrames);
        if (frameCount <= 0) throw new Error("MiniMax H3 source video requires at least 5 frames.");
        if (frameCount !== prep.totalFrames) {
            console.warn(
                `[MiniMax H3 Director] Source video aligned down: frames=${prep.totalFrames} -> ${frameCount} `
                + `(cropped ${prep.totalFrames - frameCount} tail frame(s)).`,
            );
        }
        seg.sourceVideo.rangeStart = 0;
        seg.sourceVideo.rangeEnd = frameCount;
        seg.durationSec = preferredDurationSecFromFrames(frameCount, 24);
        seg.frameCount = frameCount;
        seg.length = frameCount;
        seg._videoFrameCount = frameCount;
        endSlotLoad(editor, key);
        editor.renderImageBatchGroups();
        editor.commit(false, { syncTimeline: true });
        editor.updateVideoNameLabel?.();
        return true;
    } catch (err) {
        endSlotLoad(editor, key);
        console.error("[MiniMax H3Director] mixed source video upload failed:", err);
        void editor.showBdMessage?.(t("snapshot.errorTitle"), t("upload.alertFailed", { err: err?.message || err }));
        return false;
    }
}

async function assignSegSourceImageFromFile(editor, index, file) {
    if (!isBatchImageFile(file)) return false;
    const key = groupSlotKey(editor, index, "source-video", 0);
    beginSlotLoad(editor, key, t("slot.loading.upload"));
    try {
        const uploaded = await uploadMedia(file, (ratio, cur, total) => {
            updateSlotLoad(editor, key, { status: t("slot.loading.upload"), ratio, cur, total });
        });
        const imageFile = relPath(uploaded);
        const dims = await editor.probeInputImageDimensions(imageFile, uploaded?.type || "input");
        const seg = editor.timeline.segments[index];
        if (!seg) return false;
        const resolved = durationToClampedMiniMaxFrames(
            Number(seg.durationSec) || defaultDurationSec(resolveMixedGroupKey(seg)),
            24,
        );
        seg.sourceVideo = {
            mediaKind: "image",
            totalFrames: resolved.frames,
            rangeStart: 0,
            rangeEnd: resolved.frames,
            image: {
                imageFile,
                fileName: uploaded?.name || file.name,
                type: uploaded?.type || "input",
                subfolder: uploaded?.subfolder || "",
                width: dims.width || 0,
                height: dims.height || 0,
            },
            video: { frameMap: [], deletedSourceRanges: [] },
            videoClips: [],
        };
        seg.durationSec = resolved.durationSec;
        seg.frameCount = resolved.frames;
        seg.length = resolved.frames;
        seg._videoFrameCount = resolved.frames;
        endSlotLoad(editor, key);
        editor.renderImageBatchGroups();
        editor.commit(false, { syncTimeline: true });
        editor.updateVideoNameLabel?.();
        return true;
    } catch (err) {
        endSlotLoad(editor, key);
        console.error("[MiniMax H3Director] mixed source image upload failed:", err);
        void editor.showBdMessage?.(t("snapshot.errorTitle"), t("upload.alertFailed", { err: err?.message || err }));
        return false;
    }
}

function uploadSegSourceVideo(editor, index) {
    pickFile("image/*,video/*,.jpg,.jpeg,.png,.webp,.bmp,.gif,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts", (file) => {
        if (isBatchImageFile(file)) void assignSegSourceImageFromFile(editor, index, file);
        else void assignSegSourceVideoFromFile(editor, index, file);
    });
}

function sourceVideoFullFrameMap(sourceVideo) {
    const source = sourceVideo || {};
    if (source.mediaKind === "image" && source.image?.imageFile) {
        const count = Math.max(0, Math.round(Number(source.totalFrames) || 0));
        return Array.from({ length: count }, () => ({ clip: 0, frame: 0 }));
    }
    const video = source.video || source;
    if (Array.isArray(video.frameMap) && video.frameMap.length) {
        return video.frameMap.map((entry) => (
            entry && typeof entry === "object" ? { ...entry } : entry
        ));
    }
    const clips = Array.isArray(source.videoClips) && source.videoClips.length
        ? source.videoClips
        : ((video.videoFile || video.fileName) ? [video] : []);
    return clips.flatMap((clip, clipIndex) => {
        const count = Math.max(0, Number(clip.sourceFrameCount) || 0);
        return Array.from({ length: count }, (_, frame) => ({ clip: clipIndex, frame }));
    });
}

function sourceVideoFrameMap(sourceVideo) {
    const source = sourceVideo || {};
    const frameMap = sourceVideoFullFrameMap(source);
    const start = Math.max(0, Math.min(frameMap.length, Math.round(Number(source.rangeStart) || 0)));
    const rawEnd = Number(source.rangeEnd);
    const end = Number.isFinite(rawEnd)
        ? Math.max(start, Math.min(frameMap.length, Math.round(rawEnd)))
        : frameMap.length;
    const alignedEnd = start + floorMiniMaxFrameCount(end - start);
    return frameMap.slice(start, alignedEnd);
}

function resizeSourceRangeToFrameCount(sourceVideo, frameCount) {
    const source = sourceVideo;
    if (!source || typeof source !== "object") return false;
    const total = sourceVideoFullFrameMap(source).length;
    if (total <= 0) return false;
    const start = Math.max(0, Math.min(total - 1, Math.round(Number(source.rangeStart) || 0)));
    const requested = Math.max(5, Math.round(Number(frameCount) || 5));
    const end = Math.min(total, start + requested);
    const alignedEnd = start + floorMiniMaxFrameCount(end - start);
    if (alignedEnd <= start) return false;
    source.rangeStart = start;
    source.rangeEnd = alignedEnd;
    return true;
}

function appendSourceVideoResolutionControl(container, editor, seg, { showTitle = true } = {}) {
    const head = document.createElement("div");
    head.className = "bd-batch-source-video-head";
    const actions = document.createElement("span");
    actions.className = "bd-batch-source-video-actions";
    const select = document.createElement("select");
    select.className = "bd-batch-video-resolution";
    const current = seg.videoResolution === "source" ? "source" : "target";
    seg.videoResolution = current;
    for (const value of ["source", "target"]) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = t(`batch.videoResolution.${value}`);
        option.selected = value === current;
        select.appendChild(option);
    }
    const fitSelect = document.createElement("select");
    fitSelect.className = "bd-batch-video-fit";
    fitSelect.setAttribute("data-i18n-title", "tooltip.videoFit");
    fitSelect.title = t("tooltip.videoFit");
    const currentFit = normalizeVideoFit(seg.videoFit);
    seg.videoFit = currentFit;
    for (const value of ["contain", "crop"]) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = t(`batch.videoFit.${value}`);
        option.selected = value === currentFit;
        fitSelect.appendChild(option);
    }
    fitSelect.classList.toggle("hidden", current !== "target");
    select.onchange = () => {
        seg.videoResolution = select.value === "source" ? "source" : "target";
        fitSelect.classList.toggle("hidden", seg.videoResolution !== "target");
        editor.commit(false, { syncTimeline: true });
        editor.flushTimelineSync?.();
    };
    fitSelect.onchange = () => {
        seg.videoFit = normalizeVideoFit(fitSelect.value);
        editor.commit(false, { syncTimeline: true });
        editor.flushTimelineSync?.();
    };
    const hasVideo = sourceVideoFrameMap(seg.sourceVideo).length > 0;
    const restoreButton = document.createElement("button");
    restoreButton.type = "button";
    restoreButton.className = "bd-batch-source-video-restore";
    restoreButton.textContent = t("batch.restoreVideo");
    restoreButton.disabled = !hasVideo || seg.sourceVideo?.mediaKind === "image";
    restoreButton.onclick = (event) => {
        event.stopPropagation();
        restoreSegSourceVideo(editor, seg);
    };
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "bd-batch-source-video-delete";
    deleteButton.textContent = t("batch.delete");
    deleteButton.disabled = !hasVideo;
    deleteButton.onclick = (event) => {
        event.stopPropagation();
        delete seg.sourceVideo;
        editor.renderImageBatchGroups();
        editor.commit(false, { syncTimeline: true });
        editor.updateVideoNameLabel?.();
        editor.scheduleRender?.();
    };
    actions.append(select, fitSelect, restoreButton, deleteButton);
    if (showTitle) {
        const title = document.createElement("span");
        title.className = "bd-batch-source-video-title";
        title.textContent = t("batch.sourceMedia");
        head.append(title, actions);
        container.appendChild(head);
    } else {
        container.appendChild(actions);
    }
}

function syncSourceRangeDurationInput(editor, seg, index, frameCount) {
    const durationSec = preferredDurationSecFromFrames(frameCount, 24);
    const playSec = framesToDurationSec(frameCount, 24);
    for (const input of editor.batchList?.querySelectorAll("input[data-batch-sec-index]") || []) {
        const sameId = seg.id && input.getAttribute("data-batch-seg-id") === String(seg.id);
        const sameIndex = Number(input.getAttribute("data-batch-sec-index")) === index;
        if (!sameId && !sameIndex) continue;
        input.value = String(durationSec);
        input.title = t("batch.videoEditDurationTooltip", { frames: frameCount, play: playSec });
        break;
    }
}

function mountSegSourceVideoTimeline(container, editor, seg, index, { showHeader = true } = {}) {
    if (showHeader) appendSourceVideoResolutionControl(container, editor, seg);
    const source = seg.sourceVideo || {};
    if (source.mediaKind === "image" && source.image?.imageFile) {
        const src = document.createElement("div");
        src.className = "bd-batch-src";
        renderSourceSlot(src, source.image.imageFile);
        src.title = t("source.imageTitleFilled", {
            label: t("batch.sourceMedia"),
            file: source.image.imageFile,
        });
        bindSlotActivate(src, {
            editor,
            hasMedia: true,
            onPick: () => uploadSegSourceVideo(editor, index),
            onPreview: () => openSlotPreview({
                kind: "image",
                src: viewUrl(source.image.imageFile),
                label: t("batch.sourceMedia"),
                onReplace: () => uploadSegSourceVideo(editor, index),
            }),
        });
        bindOsFileDrop(src, (files) => {
            const file = files.find((item) => isBatchImageFile(item) || isBatchVideoFile(item));
            if (isBatchImageFile(file)) void assignSegSourceImageFromFile(editor, index, file);
            else if (file) void assignSegSourceVideoFromFile(editor, index, file);
        });
        bindImageClipboardPaste(src, (file) => assignSegSourceImageFromFile(editor, index, file));
        container.appendChild(src);
        return;
    }
    mountGroupVideoTimeline(container, {
        editor,
        seg,
        sourceLabel: t("batch.sourceMedia"),
        onUpload: () => uploadSegSourceVideo(editor, index),
        onDropFile: (file) => {
            if (isBatchImageFile(file)) void assignSegSourceImageFromFile(editor, index, file);
            else if (isBatchVideoFile(file)) void assignSegSourceVideoFromFile(editor, index, file);
        },
        onSyncMaterial: () => syncSourceMaterialFromFirstGroup(editor, index),
        onSyncSeconds: () => syncSegmentDurationFromSource(editor, index),
        onRangePreview: (start, end) => {
            void start;
            void end;
        },
        onRangeChange: (start, end) => {
            const source = seg.sourceVideo;
            if (!source) return;
            source.rangeStart = start;
            source.rangeEnd = end;
            seg.previewB64 = "";
            seg.previewFrames = [];
            editor.commit(false, { syncTimeline: true });
            editor.updateVideoNameLabel?.();
            editor.scheduleRender?.();
        },
    });
    const emptySource = container.querySelector(".bd-group-video-player");
    if (emptySource) {
        bindImageClipboardPaste(emptySource, (file) => assignSegSourceImageFromFile(editor, index, file));
    }
}

function applySegFl2vImage(editor, index, kind, imageFile, width = 0, height = 0) {
    const segId = editor.timeline.segments[index]?.id;
    const seg = (editor.timeline.segments || []).find((s) => s.id === segId)
        || editor.timeline.segments[index];
    if (!seg) return;
    const ref = { imageFile, width: width || 0, height: height || 0 };
    if (kind === "end") {
        seg.endImage = ref;
    } else {
        seg.startImage = ref;
        if (!seg.genImage?.imageFile) {
            seg.genImage = { imageFile, width: width || 0, height: height || 0 };
            seg.imageFile = imageFile;
        }
    }
    editor.renderImageBatchGroups();
    editor.updateOutputPreview?.();
    editor.commit(false, { syncTimeline: true });
    editor.scheduleRender?.();
    editor.scheduleTimelineSync?.();
}

async function assignSegFl2vFromFile(editor, index, kind, file) {
    try {
        if (!isBatchImageFile(file)) throw new Error("Not an image file");
        const uploaded = await uploadImage(file);
        const imageFile = relPath(uploaded);
        if (!imageFile) throw new Error("Upload returned empty filename");
        applySegFl2vImage(editor, index, kind, imageFile, 0, 0);
        try {
            const dims = await readImageDimensions(file);
            const live = editor.timeline.segments[index];
            const slot = kind === "end" ? live?.endImage : live?.startImage;
            if (slot?.imageFile === imageFile) {
                slot.width = dims.width;
                slot.height = dims.height;
                editor.scheduleTimelineSync?.();
            }
        } catch (dimErr) {
            console.warn("[MiniMax H3Director] mixed fl2v dims skipped:", dimErr);
        }
    } catch (err) {
        console.error("[MiniMax H3Director] mixed fl2v upload failed:", err);
        void editor.showBdMessage?.(t("snapshot.errorTitle"), t("upload.alertFailed", { err: err?.message || err }));
    }
}

function uploadSegFl2v(editor, index, kind) {
    pickFile("image/*,.jpg,.jpeg,.png,.webp,.bmp,.gif", (file) => {
        void assignSegFl2vFromFile(editor, index, kind, file);
    });
}

function clearSegFl2vImage(editor, index, kind) {
    const seg = editor.timeline.segments[index];
    if (!seg) return;
    if (kind === "end") {
        seg.endImage = null;
    } else {
        seg.startImage = null;
        // genImage mirrors the i2v first frame. Drop it so
        // last-only groups cannot resurrect the same picture as image0.
        if (seg.genImage) seg.genImage = { imageFile: "" };
        seg.imageFile = "";
    }
    editor.renderImageBatchGroups();
    editor.commit(false, { syncTimeline: true });
    editor.scheduleRender?.();
}

function applyMixedGroupType(editor, index, nextKey) {
    const key = MIXED_GROUP_TASKS.includes(nextKey) ? nextKey : "t2v";
    const seg = editor.timeline.segments[index];
    if (!seg) return;
    seg.taskType = key;
    delete seg.uiCardHeight;
    seg.uiCollapsed = false;
    if (key === "fl2v" && !seg.startImage?.imageFile && seg.genImage?.imageFile) {
        seg.startImage = {
            imageFile: seg.genImage.imageFile,
            width: seg.genImage.width || 0,
            height: seg.genImage.height || 0,
        };
    }
    if (key === "i2v" && !seg.genImage?.imageFile && seg.startImage?.imageFile) {
        seg.genImage = {
            imageFile: seg.startImage.imageFile,
            width: seg.startImage.width || 0,
            height: seg.startImage.height || 0,
        };
        seg.imageFile = seg.startImage.imageFile;
    }
    editor.renderImageBatchGroups();
    editor.syncMixedCommonLayout?.();
    editor.updateModeUI?.();
    editor.commit?.(false, { syncTimeline: true });
    editor.updateDomWidgetHeight?.();
}

function appendMixedFl2vSlots(card, editor, seg, index) {
    const media = document.createElement("div");
    media.className = "bd-batch-fl2v-media";
    const slots = document.createElement("div");
    slots.className = "bd-batch-fl2v-slots";
    const output = editor.timeline?.output || {};
    const outputWidth = parseInt(output.width, 10)
        || parseInt(editor.timeline?.width, 10)
        || parseInt(editor.widthWidget?.value, 10)
        || 864;
    const outputHeight = parseInt(output.height, 10)
        || parseInt(editor.timeline?.height, 10)
        || parseInt(editor.heightWidget?.value, 10)
        || 480;
    slots.style.setProperty("--batch-fl2v-slot-ar", `${Math.max(1, outputWidth)} / ${Math.max(1, outputHeight)}`);
    for (const kind of ["start", "end"]) {
        const wrap = document.createElement("div");
        const src = document.createElement("div");
        src.className = "bd-batch-src";
        const file = kind === "end" ? seg.endImage?.imageFile : seg.startImage?.imageFile;
        const label = t(kind === "end" ? "fl2v.tag.end" : "fl2v.tag.start");
        renderSourceSlot(src, file);
        if (!file) {
            src.innerHTML = `<span class="ph">${t(kind === "end" ? "panel.fl2v.endOptional" : "panel.fl2v.startRequired")}</span>`;
        } else {
            const tag = document.createElement("span");
            tag.className = `tag ${kind}`;
            tag.textContent = t(kind === "end" ? "fl2v.tag.end" : "fl2v.tag.start");
            src.appendChild(tag);
        }
        src.title = file
            ? t("source.imageTitleFilled", { label, file })
            : t(kind === "end" ? "tooltip.fl2vEndSlot" : "tooltip.fl2vStartSlot");
        bindSlotActivate(src, {
            editor,
            hasMedia: !!file,
            onPick: () => uploadSegFl2v(editor, index, kind),
            onPreview: () => openSlotPreview({
                kind: "image",
                src: viewUrl(file),
                label,
                onReplace: () => uploadSegFl2v(editor, index, kind),
            }),
        });
        bindOsFileDrop(src, (files) => {
            const image = files.find(isBatchImageFile);
            if (image) void assignSegFl2vFromFile(editor, index, kind, image);
        });
        bindImageClipboardPaste(src, (file) => assignSegFl2vFromFile(editor, index, kind, file));
        wrap.appendChild(src);
        if (file) {
            const x = document.createElement("span");
            x.className = "x";
            x.textContent = "×";
            x.onclick = (e) => {
                e.stopPropagation();
                clearSegFl2vImage(editor, index, kind);
            };
            src.appendChild(x);
        }
        slots.appendChild(wrap);
    }
    media.appendChild(slots);
    card.appendChild(media);
}

function readImageDimensions(file) {
    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(file);
        const img = new Image();
        const done = (fn, arg) => {
            clearTimeout(timer);
            URL.revokeObjectURL(url);
            fn(arg);
        };
        const timer = setTimeout(() => done(reject, new Error("Image dimension timeout")), 8000);
        img.onload = () => done(resolve, { width: img.naturalWidth, height: img.naturalHeight });
        img.onerror = () => done(reject, new Error("Failed to read image dimensions"));
        img.src = url;
    });
}

function groupSlotKey(editor, index, kind, slot) {
    const seg = editor.timeline?.segments?.[index];
    return slotLoadKey({
        scope: "group",
        segId: seg?.id ?? index,
        kind,
        index: slot,
    });
}

function audioLoadProgress(editor, key) {
    return (ratio, cur, total, phase) => {
        updateSlotLoad(editor, key, {
            status: phase === "extract" ? t("slot.loading.extractAudio") : t("slot.loading.upload"),
            ratio,
            cur,
            total,
        });
    };
}

async function assignSegRefFromFile(editor, index, slot, file) {
    if (!isBatchImageFile(file)) return false;
    const key = groupSlotKey(editor, index, "image", slot);
    beginSlotLoad(editor, key, t("slot.loading.upload"));
    try {
        const uploaded = await uploadImage(file);
        const seg = editor.timeline.segments[index];
        if (!seg) return false;
        const imageFile = relPath(uploaded);
        if (hasDuplicateGroupMedia(seg.refs, imageFile, slot)) {
            endSlotLoad(editor, key);
            void editor.showBdMessage?.(t("snapshot.errorTitle"), t("ref.mediaDuplicate"));
            return false;
        }
        seg.refs = (seg.refs || []).filter((r) => Number(r.index ?? r.slot) !== slot);
        seg.refs.push({ index: slot, imageFile, imageB64: "" });
        endSlotLoad(editor, key);
        editor.renderImageBatchGroups();
        editor.commit();
        return true;
    } catch (err) {
        endSlotLoad(editor, key);
        console.error("[MiniMax H3Director] batch ref upload failed:", err);
        return false;
    }
}

async function uploadSegRef(editor, index, slot) {
    pickFile("image/*", (file) => assignSegRefFromFile(editor, index, slot, file));
}

function moveBatchKindSlot(editor, kind, segIndex, fromSlot, toSlot) {
    if (fromSlot === toSlot) return;
    const seg = editor.timeline.segments[segIndex];
    if (!seg) return;
    const listKey = kind === "image" ? "refs" : kind === "audio" ? "refAudios" : "refVideos";
    seg[listKey] = swapIndexedMedia(seg[listKey] || [], fromSlot, toSlot);
    editor.renderImageBatchGroups();
    editor.commit();
}

function bindBatchKindSlot(slotEl, editor, index, kind, slotIndex, hasMedia, onDropFile, onPick, previewSrc, label) {
    bindKindSlotDnD(slotEl, {
        editor,
        kind,
        scope: "group",
        segIndex: index,
        slotIndex,
        hasMedia,
        onMove: (from, to) => moveBatchKindSlot(editor, kind, index, from, to),
        onDropFile,
    });
    bindSlotActivate(slotEl, {
        editor,
        hasMedia,
        onPick,
        onPreview: () => openSlotPreview({
            kind,
            src: previewSrc,
            label,
            onReplace: onPick,
        }),
    });
    if (kind === "image") bindImageClipboardPaste(slotEl, onDropFile);
}

function removeSegRef(editor, index, slot) {
    const seg = editor.timeline.segments[index];
    if (!seg) return;
    seg.refs = (seg.refs || []).filter((r) => Number(r.index ?? r.slot) !== slot);
    editor.renderImageBatchGroups();
    editor.commit();
}

async function assignSegAudioFromFile(editor, index, slot, file) {
    if (!isReferenceAudioSourceFile(file)) return false;
    const key = groupSlotKey(editor, index, "audio", slot);
    const extracting = isReferenceAudioVideoFile(file);
    beginSlotLoad(editor, key, extracting ? t("slot.loading.extractAudio") : t("slot.loading.upload"));
    try {
        const prepared = await prepareLocalReferenceAudio(file, audioLoadProgress(editor, key));
        const seg = editor.timeline.segments[index];
        if (!seg) return false;
        if (hasDuplicateReferenceAudio(seg.refAudios, prepared.relPath, slot)) {
            endSlotLoad(editor, key);
            void editor.showBdMessage?.(t("snapshot.errorTitle"), t("ref.audioDuplicate"));
            return false;
        }
        seg.refAudios = (seg.refAudios || []).filter((r) => Number(r.index ?? r.slot) !== slot);
        seg.refAudios.push({
            index: slot,
            audioFile: prepared.relPath,
            fileName: prepared.fileName || file.name,
            type: prepared.type || "input",
            subfolder: prepared.subfolder || "",
        });
        endSlotLoad(editor, key);
        editor.renderImageBatchGroups();
        editor.commit();
        return true;
    } catch (err) {
        endSlotLoad(editor, key);
        console.error("[MiniMax H3Director] batch audio upload failed:", err);
        void editor.showBdMessage?.(t("snapshot.errorTitle"), t("upload.refAudioFailed", { err: err?.message || err }));
        return false;
    }
}

async function uploadSegAudio(editor, index, slot) {
    pickFile("audio/*,video/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.wma,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts", (file) => {
        void assignSegAudioFromFile(editor, index, slot, file);
    });
}

function removeSegAudio(editor, index, slot) {
    const seg = editor.timeline.segments[index];
    if (!seg) return;
    seg.refAudios = (seg.refAudios || []).filter((r) => Number(r.index ?? r.slot) !== slot);
    editor.renderImageBatchGroups();
    editor.commit();
}

async function assignSegVideoFromFile(editor, index, slot, file) {
    if (!isBatchVideoFile(file)) return false;
    const key = groupSlotKey(editor, index, "video", slot);
    beginSlotLoad(editor, key, t("slot.loading.upload"));
    try {
        const uploaded = await uploadMedia(file, (ratio, cur, total) => {
            updateSlotLoad(editor, key, {
                status: t("slot.loading.upload"),
                ratio,
                cur,
                total,
            });
        });
        const seg = editor.timeline.segments[index];
        if (!seg) return false;
        const videoFile = relPath(uploaded);
        if (hasDuplicateGroupMedia(seg.refVideos, videoFile, slot)) {
            endSlotLoad(editor, key);
            void editor.showBdMessage?.(t("snapshot.errorTitle"), t("ref.mediaDuplicate"));
            return false;
        }
        seg.refVideos = (seg.refVideos || []).filter((r) => Number(r.index ?? r.slot) !== slot);
        seg.refVideos.push({
            index: slot,
            videoFile,
            fileName: uploaded?.name || file.name,
            type: "input",
            subfolder: uploaded?.subfolder || "",
        });
        endSlotLoad(editor, key);
        editor.renderImageBatchGroups();
        editor.commit();
        return true;
    } catch (err) {
        endSlotLoad(editor, key);
        console.error("[MiniMax H3Director] batch video upload failed:", err);
        void editor.showBdMessage?.(t("snapshot.errorTitle"), t("upload.refVideoBatchFailed", { err: err?.message || err }));
        return false;
    }
}

async function uploadSegVideo(editor, index, slot) {
    pickFile("video/*,.mp4,.mov,.webm,.mkv", (file) => {
        void assignSegVideoFromFile(editor, index, slot, file);
    });
}

function removeSegVideo(editor, index, slot) {
    const seg = editor.timeline.segments[index];
    if (!seg) return;
    seg.refVideos = (seg.refVideos || []).filter((r) => Number(r.index ?? r.slot) !== slot);
    editor.renderImageBatchGroups();
    editor.commit();
}

function fileBaseName(path) {
    const s = String(path || "").replace(/\\/g, "/");
    return s.split("/").pop() || s;
}

function makeMediaFileName(path) {
    const name = document.createElement("span");
    name.className = "bd-media-file-name";
    name.textContent = fileBaseName(path);
    name.title = String(path || "");
    requestAnimationFrame(() => {
        const overflow = Math.max(0, name.scrollWidth - name.clientWidth);
        if (overflow > 2) {
            name.style.setProperty("--bd-file-overflow", `${overflow}px`);
            name.style.setProperty("--bd-file-scroll", `${Math.max(2.4, overflow / 24)}s`);
            name.classList.add("can-scroll");
        }
    });
    return name;
}

function countFilledRefs(seg, { picFree = [], audFree = [], vidFree = [] } = {}) {
    return {
        imgs: countFilledOnIndices(seg.refs, picFree, refHasImage),
        videos: countFilledOnIndices(seg.refVideos, vidFree, refHasVideo),
        audios: countFilledOnIndices(seg.refAudios, audFree, refHasAudio),
    };
}

function dropFilesIntoGroupSlots(editor, index, files, e, {
    isFile,
    slotSelector,
    freeIndices,
    itemsKey,
    hasFn,
    kind,
    assignFile,
}) {
    const matching = files.filter(isFile);
    if (!matching.length) return;
    const hit = e.target.closest?.(slotSelector);
    const hitIndex = hit ? Number(hit.dataset.refIndex) : NaN;
    const freeSet = new Set(freeIndices);
    const replaceFirst = Number.isFinite(hitIndex) && freeSet.has(hitIndex);
    const reserved = new Set();
    const jobs = [];
    for (let i = 0; i < matching.length; i++) {
        const seg = editor.timeline.segments[index];
        if (!seg) return;
        const target = (i === 0 && replaceFirst && !reserved.has(hitIndex)
            && !isSlotBusy(editor, "group", seg.id ?? index, kind, hitIndex))
            ? hitIndex
            : nextEmptySlot(
                seg[itemsKey],
                freeIndices,
                hasFn,
                editor,
                "group",
                seg.id ?? index,
                kind,
                reserved,
            );
        if (target < 0) {
            if (i === 0) void editor.showBdMessage?.(t("snapshot.errorTitle"), t("mediaPicker.slotsFull"));
            break;
        }
        reserved.add(target);
        jobs.push(assignFile(editor, index, target, matching[i]));
    }
    void Promise.all(jobs);
}

function createR2vSection(title, countText) {
    const section = document.createElement("div");
    section.className = "bd-r2v-section";
    const head = document.createElement("div");
    head.className = "bd-r2v-section-head";
    const titleEl = document.createElement("span");
    titleEl.className = "bd-r2v-section-title";
    titleEl.textContent = title;
    const actions = document.createElement("span");
    actions.className = "bd-r2v-section-actions";
    const countEl = document.createElement("span");
    countEl.className = "bd-r2v-section-count";
    countEl.textContent = countText;
    actions.appendChild(countEl);
    head.appendChild(titleEl);
    head.appendChild(actions);
    section.appendChild(head);
    return section;
}

function renderAudioSlot(el, ref, slot, index, editor, { r2v = false } = {}) {
    const label = refAudioLabel(slot);
    const file = ref?.audioFile || ref?.fileName || "";
    el.className = `bd-batch-audio${file ? " has-audio" : ""}`;
    el.dataset.refKind = "audio";
    el.dataset.refIndex = String(slot);
    el.title = file
        ? t("ref.audioTitleFilled", { label, file })
        : t("ref.clickUpload", { label });
    el.innerHTML = "";
    if (r2v) {
        const thumb = document.createElement("div");
        thumb.className = "bd-r2v-thumb";
        const meta = document.createElement("div");
        meta.className = "bd-r2v-meta";
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = label;
        meta.appendChild(tag);
        if (file) meta.appendChild(makeMediaFileName(file));
        el.appendChild(thumb);
        el.appendChild(meta);
        if (file) {
            const playBtn = document.createElement("button");
            playBtn.type = "button";
            playBtn.className = "bd-r2v-play";
            playBtn.title = t("batch.r2v.play");
            playBtn.textContent = "▶";
            thumb.appendChild(playBtn);
            const dur = document.createElement("span");
            dur.className = "bd-r2v-dur";
            dur.textContent = ref?.durationSec != null
                ? formatMediaDuration(ref.durationSec)
                : "--:--";
            meta.appendChild(dur);
            const progress = document.createElement("div");
            progress.className = "bd-r2v-progress";
            progress.title = t("batch.r2v.seek");
            progress.innerHTML = `<div class="bd-r2v-progress-fill"></div>`;
            el.appendChild(progress);
            const audio = document.createElement("audio");
            audio.preload = "metadata";
            audio.src = viewUrl(file);
            audio.className = "bd-r2v-media";
            el.appendChild(audio);
            bindR2vMediaPlayback(audio, playBtn, progress);
            wireMediaDuration(audio, dur, (sec) => {
                if (ref) ref.durationSec = sec;
            });
            const x = document.createElement("span");
            x.className = "x";
            x.textContent = "×";
            x.onclick = (e) => { e.stopPropagation(); removeSegAudio(editor, index, slot); };
            el.appendChild(x);
        } else {
            thumb.textContent = "♪";
            const hint = document.createElement("span");
            hint.className = "name";
            hint.textContent = t("batch.r2v.uploadHint");
            meta.appendChild(hint);
        }
        return;
    }
    if (file) {
        const tag = document.createElement("span");
        tag.textContent = label;
        el.appendChild(tag);
        const name = document.createElement("span");
        name.className = "name bd-media-file-name";
        name.textContent = fileBaseName(file);
        name.title = file;
        el.appendChild(name);
        const x = document.createElement("span");
        x.className = "x";
        x.textContent = "×";
        x.onclick = (e) => { e.stopPropagation(); removeSegAudio(editor, index, slot); };
        el.appendChild(x);
    } else {
        el.textContent = t("ref.audioUpload", { label });
    }
}

function renderVideoSlot(el, ref, slot, index, editor, { r2v = false } = {}) {
    const label = refVideoLabel(slot);
    const file = ref?.videoFile || "";
    const posterSrc = ref?.previewImageUrl
        || (ref?.previewImageFile ? viewUrl(ref.previewImageFile) : "");
    const hasMedia = !!(file || posterSrc || ref?.linked);
    const titleFile = file || ref?.fileName || ref?.previewImageFile || "";
    el.className = `bd-batch-video${hasMedia ? " has-video" : ""}`;
    el.dataset.refKind = "video";
    el.dataset.refIndex = String(slot);
    el.title = hasMedia
        ? t("ref.videoTitleFilled", { label, file: titleFile || label })
        : t("ref.videoTitleEmpty", { label });
    el.innerHTML = "";
    if (r2v) {
        const thumb = document.createElement("div");
        thumb.className = "bd-r2v-thumb bd-r2v-thumb-video";
        const meta = document.createElement("div");
        meta.className = "bd-r2v-meta";
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = label;
        meta.appendChild(tag);
        if (file) meta.appendChild(makeMediaFileName(file));
        el.appendChild(thumb);
        el.appendChild(meta);
        if (file) {
            const video = document.createElement("video");
            video.preload = "metadata";
            video.muted = true;
            video.playsInline = true;
            video.src = viewUrl(file);
            video.className = "bd-r2v-media";
            thumb.appendChild(video);
            const playBtn = document.createElement("button");
            playBtn.type = "button";
            playBtn.className = "bd-r2v-play";
            playBtn.title = t("batch.r2v.play");
            playBtn.textContent = "▶";
            thumb.appendChild(playBtn);
            const dur = document.createElement("span");
            dur.className = "bd-r2v-dur";
            dur.textContent = ref?.durationSec != null
                ? formatMediaDuration(ref.durationSec)
                : "--:--";
            meta.appendChild(dur);
            bindR2vMediaPlayback(video, playBtn);
            playBtn.addEventListener("click", () => {
                video.muted = false;
            });
            wireMediaDuration(video, dur, (sec) => {
                if (ref) ref.durationSec = sec;
            });
            video.addEventListener("loadeddata", () => {
                if (video.readyState >= 2 && video.currentTime < 0.05) {
                    try { video.currentTime = Math.min(0.1, (video.duration || 1) * 0.05); } catch (_) { /* ignore */ }
                }
            }, { once: true });
            if (ref?.pairedAudioFile) {
                const audioBadge = document.createElement("span");
                audioBadge.className = "bd-r2v-paired-audio";
                audioBadge.title = ref.pairedAudioFile;
                audioBadge.textContent = "♪";
                meta.appendChild(audioBadge);
            }
            const x = document.createElement("span");
            x.className = "x";
            x.textContent = "×";
            x.onclick = (e) => { e.stopPropagation(); removeSegVideo(editor, index, slot); };
            el.appendChild(x);
        } else if (posterSrc) {
            // External IMAGE-batch video: show upstream first-frame poster (no file path).
            const img = document.createElement("img");
            img.className = "bd-r2v-media";
            img.src = posterSrc;
            img.alt = label;
            thumb.appendChild(img);
            const hint = document.createElement("span");
            hint.className = "name";
            hint.textContent = t("batch.r2v.externalPoster");
            meta.appendChild(hint);
            if (ref?.pairedAudioFile) {
                const audioBadge = document.createElement("span");
                audioBadge.className = "bd-r2v-paired-audio";
                audioBadge.title = ref.pairedAudioFile;
                audioBadge.textContent = "♪";
                meta.appendChild(audioBadge);
            }
        } else if (ref?.linked) {
            thumb.textContent = "▶";
            const hint = document.createElement("span");
            hint.className = "name";
            hint.textContent = t("batch.r2v.externalLinked");
            meta.appendChild(hint);
        } else {
            thumb.textContent = "▶";
            const hint = document.createElement("span");
            hint.className = "name";
            hint.textContent = t("batch.r2v.uploadHint");
            meta.appendChild(hint);
        }
        return;
    }
    if (file) {
        const tag = document.createElement("span");
        tag.textContent = label;
        el.appendChild(tag);
        const name = document.createElement("span");
        name.className = "name bd-media-file-name";
        name.textContent = fileBaseName(file);
        name.title = file;
        el.appendChild(name);
        const x = document.createElement("span");
        x.className = "x";
        x.textContent = "×";
        x.onclick = (e) => { e.stopPropagation(); removeSegVideo(editor, index, slot); };
        el.appendChild(x);
    } else {
        el.textContent = t("ref.videoUpload", { label });
    }
}

function slotPreviewSrc(ref, kind) {
    const raw = mediaSrcFromRef(ref, kind);
    if (!raw) return "";
    if (kind === "image" && String(raw).startsWith("data:")) return raw;
    return viewUrl(raw);
}

/**
 * r2v layout: left = pictures/videos/audio · right = prompt + preview (returned).
 * @returns {HTMLElement} main column for prompt/preview
 */
function appendR2vMediaSections(
    card,
    seg,
    index,
    editor,
    { sourceVideo = false, referenceVideos = true } = {},
) {
    const picFree = groupFreeIndices(editor, "image");
    const audFree = groupFreeIndices(editor, "audio");
    const vidFree = groupFreeIndices(editor, "video");
    const picSlots = picFree.length;
    const audSlots = audFree.length;
    const vidSlots = vidFree.length;
    const counts = countFilledRefs(seg, { picFree, audFree, vidFree });
    const body = document.createElement("div");
    body.className = "bd-batch-r2v-body";

    const assets = document.createElement("div");
    assets.className = "bd-batch-r2v-assets";

    if (sourceVideo) {
        const sourceSection = createR2vSection(t("batch.sourceMedia"), "");
        const actions = sourceSection.querySelector(".bd-r2v-section-actions");
        appendSourceVideoResolutionControl(actions, editor, seg, { showTitle: false });
        mountSegSourceVideoTimeline(sourceSection, editor, seg, index, { showHeader: false });
        assets.appendChild(sourceSection);
    }

    const imgSection = createR2vSection(
        t("batch.r2v.sectionPictures"),
        `${counts.imgs}/${picSlots}`,
    );
    const imageActions = imgSection.querySelector(".bd-r2v-section-actions");
    const sizeRow = document.createElement("label");
    sizeRow.className = "bd-batch-refsize";
    sizeRow.title = t("tooltip.refImageSize");
    const sizeLabel = document.createElement("span");
    sizeLabel.setAttribute("data-i18n", "output.refImageSize.label");
    sizeLabel.textContent = t("output.refImageSize.label");
    const sizeSelect = document.createElement("select");
    sizeSelect.className = "bd-select";
    const currentSize = resolveSegmentRefImageSize(seg, editor.timeline?.output);
    seg.refImageSize = currentSize;
    for (const optionKey of REF_IMAGE_SIZE_OPTIONS) {
        const option = document.createElement("option");
        option.value = optionKey;
        option.setAttribute("data-i18n", `output.refImageSize.${optionKey}`);
        option.textContent = t(`output.refImageSize.${optionKey}`);
        option.selected = optionKey === currentSize;
        sizeSelect.appendChild(option);
    }
    sizeSelect.onchange = (event) => {
        event.stopPropagation();
        const live = (editor.timeline.segments || []).find((item) => item?.id && item.id === seg.id)
            || editor.timeline.segments?.[index];
        if (!live) return;
        live.refImageSize = resolveSegmentRefImageSize({ refImageSize: sizeSelect.value });
        editor.scheduleTimelineSync?.();
        editor.flushTimelineSync?.();
    };
    sizeSelect.onclick = (event) => event.stopPropagation();
    sizeRow.appendChild(sizeLabel);
    sizeRow.appendChild(sizeSelect);
    imageActions?.insertBefore(sizeRow, imageActions.firstChild);
    const refs = document.createElement("div");
    refs.className = "bd-batch-refs";
    bindOsFileDrop(refs, (files, e) => dropFilesIntoGroupSlots(editor, index, files, e, {
        isFile: isBatchImageFile,
        slotSelector: ".bd-batch-ref",
        freeIndices: picFree,
        itemsKey: "refs",
        hasFn: refHasImage,
        kind: "image",
        assignFile: assignSegRefFromFile,
    }));
    if (!editor._r2vPicsVisible) editor._r2vPicsVisible = {};
    const segKey = String(seg.id ?? index);
    let highestPos = -1;
    picFree.forEach((abs, pos) => {
        const ref = (seg.refs || []).find((r) => Number(r.index ?? r.slot) === abs);
        if (refHasImage(ref)) highestPos = pos;
    });
    const minVisible = highestPos >= 0
        ? Math.min(picSlots, Math.ceil((highestPos + 1) / R2V_PICTURE_STEP) * R2V_PICTURE_STEP)
        : Math.min(picSlots, R2V_PICTURE_STEP);
    let visible = Number(editor._r2vPicsVisible[segKey]) || Math.min(picSlots, R2V_PICTURE_STEP);
    visible = Math.max(Math.min(picSlots, R2V_PICTURE_STEP), Math.min(picSlots, visible));
    if (visible < minVisible) visible = minVisible;
    editor._r2vPicsVisible[segKey] = visible;

    const applyPicVisibility = () => {
        refs.querySelectorAll(".bd-batch-ref").forEach((el, i) => {
            el.classList.toggle("bd-r2v-pic-hidden", i >= visible);
        });
    };

    picFree.forEach((abs, local) => {
        const ref = (seg.refs || []).find((r) => Number(r.index ?? r.slot) === abs);
        const slot = document.createElement("div");
        slot.className = "bd-batch-ref";
        slot.dataset.refKind = "image";
        slot.dataset.refIndex = String(abs);
        slot.dataset.refScope = "group";
        if (local >= visible) slot.classList.add("bd-r2v-pic-hidden");
        renderR2vRefSlot(slot, ref, abs, index, editor);
        bindBatchKindSlot(
            slot,
            editor,
            index,
            "image",
            abs,
            refHasImage(ref),
            (file) => void assignSegRefFromFile(editor, index, abs, file),
            () => uploadSegRef(editor, index, abs),
            slotPreviewSrc(ref, "image"),
            refImageLabel(abs),
        );
        refs.appendChild(slot);
    });
    imgSection.appendChild(refs);

    if (picSlots > R2V_PICTURE_STEP) {
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "bd-r2v-pics-toggle";
        const syncToggleLabel = () => {
            if (visible < picSlots) {
                const next = Math.min(R2V_PICTURE_STEP, picSlots - visible);
                toggle.textContent = t("batch.r2v.expandPics", { n: next });
            } else {
                toggle.textContent = t("batch.r2v.collapsePics");
            }
        };
        syncToggleLabel();
        toggle.onclick = (e) => {
            e.stopPropagation();
            if (visible < picSlots) {
                visible = Math.min(picSlots, visible + R2V_PICTURE_STEP);
            } else {
                visible = Math.max(Math.min(picSlots, R2V_PICTURE_STEP), minVisible);
            }
            editor._r2vPicsVisible[segKey] = visible;
            applyPicVisibility();
            syncToggleLabel();
            editor.updateDomWidgetHeight?.();
        };
        imgSection.appendChild(toggle);
    }
    assets.appendChild(imgSection);

    if (referenceVideos) {
        const videoSection = createR2vSection(
            t("batch.r2v.sectionVideos"),
            `${counts.videos}/${vidSlots}`,
        );
        const videos = document.createElement("div");
        videos.className = "bd-batch-videos";
        bindOsFileDrop(videos, (files, e) => dropFilesIntoGroupSlots(editor, index, files, e, {
            isFile: isBatchVideoFile,
            slotSelector: ".bd-batch-video",
            freeIndices: vidFree,
            itemsKey: "refVideos",
            hasFn: refHasVideo,
            kind: "video",
            assignFile: assignSegVideoFromFile,
        }));
        for (const abs of vidFree) {
            const ref = (seg.refVideos || []).find((r) => Number(r.index ?? r.slot) === abs);
            const slot = document.createElement("div");
            slot.dataset.refScope = "group";
            renderVideoSlot(slot, ref, abs, index, editor, { r2v: true });
            bindBatchKindSlot(
                slot,
                editor,
                index,
                "video",
                abs,
                refHasVideo(ref),
                (file) => void assignSegVideoFromFile(editor, index, abs, file),
                () => uploadSegVideo(editor, index, abs),
                slotPreviewSrc(ref, "video"),
                refVideoLabel(abs),
            );
            videos.appendChild(slot);
        }
        videoSection.appendChild(videos);
        assets.appendChild(videoSection);
    }

    const audioSection = createR2vSection(
        t("batch.r2v.sectionAudios"),
        `${counts.audios}/${audSlots}`,
    );
    const audios = document.createElement("div");
    audios.className = "bd-batch-audios";
    bindOsFileDrop(audios, (files, e) => dropFilesIntoGroupSlots(editor, index, files, e, {
        isFile: isReferenceAudioSourceFile,
        slotSelector: ".bd-batch-audio",
        freeIndices: audFree,
        itemsKey: "refAudios",
        hasFn: refHasAudio,
        kind: "audio",
        assignFile: assignSegAudioFromFile,
    }));
    for (const abs of audFree) {
        const ref = (seg.refAudios || []).find((r) => Number(r.index ?? r.slot) === abs);
        const slot = document.createElement("div");
        slot.dataset.refScope = "group";
        renderAudioSlot(slot, ref, abs, index, editor, { r2v: true });
        bindBatchKindSlot(
            slot,
            editor,
            index,
            "audio",
            abs,
            refHasAudio(ref),
            (file) => void assignSegAudioFromFile(editor, index, abs, file),
            () => uploadSegAudio(editor, index, abs),
            slotPreviewSrc(ref, "audio"),
            refAudioLabel(abs),
        );
        audios.appendChild(slot);
    }
    audioSection.appendChild(audios);
    assets.appendChild(audioSection);

    const main = document.createElement("div");
    main.className = "bd-batch-r2v-main";

    body.appendChild(assets);
    body.appendChild(main);
    card.appendChild(body);
    return main;
}

function renderR2vRefSlot(el, ref, slot, index, editor) {
    const label = refImageLabel(slot);
    const has = !!ref?.imageFile;
    el.classList.toggle("has-img", has);
    el.innerHTML = "";
    el.title = ref?.imageFile
        ? t("ref.imageTitleFilled", { label, file: ref.imageFile })
        : t("ref.imageTitleEmpty", { label });
    if (has) {
        const img = document.createElement("img");
        img.src = viewUrl(ref.imageFile);
        img.draggable = false;
        el.appendChild(img);
        const dot = document.createElement("span");
        dot.className = "dot";
        el.appendChild(dot);
        const cap = document.createElement("span");
        cap.className = "cap";
        cap.textContent = label;
        el.appendChild(cap);
        el.appendChild(makeMediaFileName(ref.imageFile));
        const x = document.createElement("span");
        x.className = "x";
        x.textContent = "×";
        x.onclick = (e) => { e.stopPropagation(); removeSegRef(editor, index, slot); };
        el.appendChild(x);
    } else {
        const cap = document.createElement("span");
        cap.className = "cap";
        cap.textContent = label;
        el.appendChild(cap);
    }
}

function renderSourceSlot(el, imageFile) {
    el.classList.toggle("has-img", !!imageFile);
    if (imageFile) {
        el.innerHTML = `<img src="${viewUrl(imageFile)}" alt="">`;
    } else {
        el.textContent = t("batch.uploadSource");
    }
}

function renderRefSlot(el, ref, slot, index, editor) {
    const label = refImageLabel(slot);
    el.classList.toggle("has-img", !!ref?.imageFile);
    el.innerHTML = "";
    el.title = ref?.imageFile
        ? t("ref.imageTitleFilled", { label, file: ref.imageFile })
        : t("ref.imageTitleEmpty", { label });
    if (ref?.imageFile) {
        const img = document.createElement("img");
        img.src = viewUrl(ref.imageFile);
        img.draggable = false;
        el.appendChild(img);
        el.appendChild(makeMediaFileName(ref.imageFile));
        const x = document.createElement("span");
        x.className = "x";
        x.textContent = "×";
        x.onclick = (e) => { e.stopPropagation(); removeSegRef(editor, index, slot); };
        el.appendChild(x);
    } else {
        el.textContent = label;
    }
}

function frameSrc(b64) {
    if (!b64) return "";
    return b64.startsWith("data:") ? b64 : `data:image/jpeg;base64,${b64}`;
}

function loadFrameImages(frames) {
    return Promise.all(frames.map((b64) => new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = frameSrc(b64);
    })));
}

function drawFrame(canvas, img) {
    const ctx = canvas.getContext("2d");
    if (!ctx || !img) return;
    const cw = canvas.clientWidth || 160;
    const ch = canvas.clientHeight || 90;
    if (canvas.width !== cw) canvas.width = cw;
    if (canvas.height !== ch) canvas.height = ch;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, cw, ch);
    const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight);
    const dw = img.naturalWidth * scale;
    const dh = img.naturalHeight * scale;
    ctx.drawImage(img, (cw - dw) / 2, (ch - dh) / 2, dw, dh);
}

export const LIVE_PREVIEW_SPEED_MIN = 0;
export const LIVE_PREVIEW_SPEED_MAX = 1;
export const LIVE_PREVIEW_SPEED_DEFAULT = 0.25;
export const LIVE_PREVIEW_SPEED_STEP = 0.05;

export function clampLivePreviewSpeed(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return LIVE_PREVIEW_SPEED_DEFAULT;
    const clamped = clamp(n, LIVE_PREVIEW_SPEED_MIN, LIVE_PREVIEW_SPEED_MAX);
    const steps = Math.round((clamped - LIVE_PREVIEW_SPEED_MIN) / LIVE_PREVIEW_SPEED_STEP);
    return clamp(
        Number((LIVE_PREVIEW_SPEED_MIN + steps * LIVE_PREVIEW_SPEED_STEP).toFixed(2)),
        LIVE_PREVIEW_SPEED_MIN,
        LIVE_PREVIEW_SPEED_MAX,
    );
}

export function formatLivePreviewSpeed(v) {
    const n = clampLivePreviewSpeed(v);
    return n.toFixed(2).replace(/\.?0+$/, "") || "0";
}

function livePreviewSpeedFromEl(el) {
    return clampLivePreviewSpeed(el?.closest?.(".bd-wrap")?.dataset?.livePreviewSpeed);
}

function stopLiveImgAnim(img) {
    if (!img?._mmxLiveTimer) return;
    clearInterval(img._mmxLiveTimer);
    img._mmxLiveTimer = null;
}

function playLiveImgFrames(img, frames, fps) {
    const srcs = (Array.isArray(frames) ? frames : []).map(frameSrc).filter(Boolean);
    if (!img || !srcs.length) return;
    const same = Array.isArray(img._mmxLiveSrcs)
        && srcs.length === img._mmxLiveSrcs.length
        && srcs.every((s, i) => s === img._mmxLiveSrcs[i]);
    const keepIdx = same && img._mmxLiveTimer ? (img._mmxLiveFrameIdx || 0) : 0;
    stopLiveImgAnim(img);
    img._mmxLiveSrcs = srcs;
    img._mmxLiveBaseFps = Number(fps) || 16;
    let idx = srcs.length ? keepIdx % srcs.length : 0;
    img._mmxLiveFrameIdx = idx;
    const speed = livePreviewSpeedFromEl(img);
    if (speed <= 0) {
        img.removeAttribute("src");
        img.classList.add("hidden");
        return;
    }
    img.classList.remove("hidden");
    img.src = srcs[idx];
    if (srcs.length <= 1) return;
    const playFps = img._mmxLiveBaseFps * speed;
    const interval = Math.max(40, 1000 / Math.max(0.05, playFps));
    img._mmxLiveTimer = setInterval(() => {
        idx = (idx + 1) % srcs.length;
        img._mmxLiveFrameIdx = idx;
        img.src = srcs[idx];
    }, interval);
}

export function restartLivePreviewAnims(root) {
    const imgs = root?.querySelectorAll?.("img.bd-live-preview") || [];
    for (const img of imgs) {
        if (img._mmxLiveSrcs?.length > 1) {
            playLiveImgFrames(img, img._mmxLiveSrcs, img._mmxLiveBaseFps);
        }
    }
}

function mountLivePreview(el, seg, badgeText) {
    stopLiveImgAnim(el.querySelector("img.bd-live-preview"));
    stopPlayer(el);
    el.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "bd-batch-live-preview";
    const img = document.createElement("img");
    img.className = "bd-live-preview";
    img.alt = "live preview";
    const badge = document.createElement("div");
    badge.className = "bd-batch-live-badge";
    badge.textContent = badgeText || t("batch.generating");
    wrap.appendChild(img);
    wrap.appendChild(badge);
    el.appendChild(wrap);
    playLiveImgFrames(
        img,
        (seg.previewFrames?.length ? seg.previewFrames : [seg.previewB64]),
        seg.previewFps || 16,
    );
}

/** Patch a per-group preview cell in place (t2v cards, fl2v shots, v2v strips). */
export function patchGroupLivePreview(el, rec = {}, placeholder = "") {
    if (!el) return;
    const b64 = rec.b64 || rec.previewB64 || rec.image_b64 || rec.imageB64 || "";
    const frames = Array.isArray(rec.frames)
        ? rec.frames
        : (Array.isArray(rec.previewFrames) ? rec.previewFrames : []);
    const live = !!(rec.live || rec.previewLive);
    const fps = rec.fps || rec.previewFps || 16;
    const step = rec.step ?? rec.previewStep;
    const total = rec.total_steps ?? rec.previewTotalSteps ?? rec.totalSteps;
    if (!b64 && !frames.length) {
        stopLiveImgAnim(el.querySelector("img.bd-live-preview"));
        stopPlayer(el);
        el.textContent = placeholder || t("batch.previewVideoAfterRun");
        return;
    }
    const badge = (step && total)
        ? t("batch.generatingStep", { step, total })
        : (live ? t("batch.generating") : "");
    const fake = {
        previewB64: b64,
        previewFrames: frames.length ? frames : [b64],
        previewFps: fps,
    };
    let img = el.querySelector("img.bd-live-preview");
    if (!img) {
        mountLivePreview(el, fake, badge);
        const bdg = el.querySelector(".bd-batch-live-badge");
        if (bdg) bdg.classList.toggle("hidden", !badge);
        return;
    }
    playLiveImgFrames(img, fake.previewFrames, fps);
    let bdg = el.querySelector(".bd-batch-live-badge");
    if (bdg) {
        bdg.textContent = badge;
        bdg.classList.toggle("hidden", !badge);
    }
}

function mountVideoPreview(el, seg, running, fps) {
    stopPlayer(el);
    el.innerHTML = "";
    if (running) {
        if (seg.previewB64) {
            const step = seg.previewStep;
            const total = seg.previewTotalSteps;
            const badge = (step && total)
                ? t("batch.generatingStep", { step, total })
                : t("batch.generating");
            mountLivePreview(el, seg, badge);
            return;
        }
        el.textContent = t("batch.generating");
        return;
    }
    const frames = (seg.previewFrames?.length ? seg.previewFrames : null)
        || (seg.previewB64 ? [seg.previewB64] : null);
    if (!frames?.length) {
        el.textContent = t("batch.previewVideoAfterRun");
        return;
    }
    const wrap = document.createElement("div");
    wrap.className = "bd-batch-vpreview";
    const canvas = document.createElement("canvas");
    canvas.height = 90;
    const ctrl = document.createElement("div");
    ctrl.className = "bd-batch-vpreview-ctrl";
    const playBtn = document.createElement("button");
    playBtn.type = "button";
    playBtn.className = "bd-btn";
    playBtn.textContent = t("batch.play");
    const meta = document.createElement("div");
    meta.className = "bd-batch-vpreview-meta";
    meta.textContent = t("batch.previewMeta", { n: frames.length, fps: formatPreviewFps(fps) });
    ctrl.appendChild(playBtn);
    wrap.appendChild(canvas);
    wrap.appendChild(ctrl);
    wrap.appendChild(meta);
    el.appendChild(wrap);

    const state = { playing: false, timer: null, idx: 0, images: null };
    _players.set(wrap, state);

    loadFrameImages(frames).then((images) => {
        state.images = images;
        drawFrame(canvas, images[0]);
    }).catch(() => {
        meta.textContent = t("batch.previewLoadFailed");
    });

    playBtn.onclick = (e) => {
        e.stopPropagation();
        if (!state.images?.length) return;
        if (state.playing) {
            state.playing = false;
            if (state.timer) clearInterval(state.timer);
            state.timer = null;
            playBtn.textContent = t("batch.play");
            return;
        }
        state.playing = true;
        playBtn.textContent = t("batch.pause");
        const interval = Math.max(20, 1000 / Math.max(1, fps));
        state.timer = setInterval(() => {
            if (!state.images?.length) return;
            state.idx = (state.idx + 1) % state.images.length;
            drawFrame(canvas, state.images[state.idx]);
        }, interval);
    };
}

function renderImagePreview(el, seg, running) {
    stopPlayer(el);
    el.innerHTML = "";
    if (running) {
        if (seg.previewB64) {
            const step = seg.previewStep;
            const total = seg.previewTotalSteps;
            const badge = (step && total)
                ? t("batch.generatingStep", { step, total })
                : t("batch.generating");
            mountLivePreview(el, seg, badge);
            return;
        }
        el.textContent = t("batch.generating");
        return;
    }
    if (seg.previewB64) {
        const img = document.createElement("img");
        img.src = frameSrc(seg.previewB64);
        img.alt = "preview";
        el.appendChild(img);
        return;
    }
    el.textContent = t("batch.previewAfterRun");
}

function renderPreview(el, seg, running, isVideo, fps) {
    if (isVideo) mountVideoPreview(el, seg, running, fps);
    else renderImagePreview(el, seg, running);
}

const LORA_TRIGGER_WIDGET_NAMES = new Set([
    "filtered_trigger_words",
    "trigger_words",
    "toggle_trigger_words",
    "orinalmessage",
    "originalmessage",
    "text",
    "string",
    "value",
    "prompt",
]);

function textFromLoraTriggerValue(value) {
    if (value == null || typeof value === "boolean" || typeof value === "number") return "";
    if (typeof value === "string") return value.trim();
    if (Array.isArray(value)) {
        const parts = [];
        for (const item of value) {
            if (typeof item === "string") {
                const s = item.trim();
                if (s) parts.push(s);
                continue;
            }
            if (!item || typeof item !== "object" || item.active === false) continue;
            if (Array.isArray(item.items)) {
                const kids = item.items
                    .filter((child) => child && child.active !== false)
                    .map((child) => String(child.text || "").trim())
                    .filter(Boolean);
                if (kids.length) parts.push(kids.join(", "));
                continue;
            }
            const s = String(item.text || "").trim();
            if (s) parts.push(s);
        }
        return parts.join(", ");
    }
    if (typeof value === "object" && value.__value__ != null) {
        return textFromLoraTriggerValue(value.__value__);
    }
    return "";
}

function readLinkedLoraTrigger(origin) {
    if (!origin) return "";
    const widgets = origin.widgets || [];
    for (const item of widgets) {
        if (!LORA_TRIGGER_WIDGET_NAMES.has(String(item?.name || "").toLowerCase())) continue;
        const text = textFromLoraTriggerValue(item?.value);
        if (text) return text;
    }
    const fromTag = textFromLoraTriggerValue(origin.tagWidget?.value);
    if (fromTag) return fromTag;
    for (const item of widgets) {
        const type = String(item?.type || "").toLowerCase();
        if (type === "toggle" || type === "boolean" || type === "number" || type === "combo") continue;
        if (typeof item?.value === "boolean" || typeof item?.value === "number") continue;
        const text = textFromLoraTriggerValue(item?.value);
        if (text) return text;
    }
    return "";
}

function readLoraTriggerWords(editor, inputName = "lora_trigger_words") {
    const node = editor?.node;
    const w = node?.widgets?.find((item) => item?.name === inputName);
    const local = textFromLoraTriggerValue(w?.value);
    if (local) return local;
    const inp = (node?.inputs || []).find((item) => item?.name === inputName);
    const linkId = inp?.link;
    if (linkId == null) return "";
    const link = node.graph?.links?.[linkId];
    const origin = node.graph?.getNodeById?.(link?.origin_id);
    return readLinkedLoraTrigger(origin);
}

function executedPromptTaskKey(editor, seg) {
    const globalKey = resolveTaskKey(
        editor?.getTaskKey?.()
        || editor?.timeline?.global?.taskType
        || "",
    );
    if (isMixedTask(globalKey)) return resolveMixedGroupKey(seg);
    return globalKey;
}

function r2vLoraTriggerIsLinked(editor) {
    const inp = (editor?.node?.inputs || []).find((item) => item?.name === "lora_trigger_words_r2v");
    return inp?.link != null || (Array.isArray(inp?.links) && inp.links.length > 0);
}

function effectiveLoraTriggerWords(editor, seg) {
    const taskKey = executedPromptTaskKey(editor, seg);
    if (["r2v", "v2v", "rv2v"].includes(taskKey) && r2vLoraTriggerIsLinked(editor)) {
        return readLoraTriggerWords(editor, "lora_trigger_words_r2v");
    }
    return readLoraTriggerWords(editor);
}

function loraStyleTagsBlock(trigger) {
    return `style_tags:\n${trigger}`;
}

function executedPromptText(editor, seg) {
    let body = String(seg?.prompt || "").trim();
    const lora = effectiveLoraTriggerWords(editor, seg);
    if (!lora) return body;
    if (executedPromptTaskKey(editor, seg) === "r2v") {
        const block = loraStyleTagsBlock(lora);
        if (!body) return block;
        if (body === block || body.endsWith(`\n${block}`)) return body;
        return `${body}\n\n${block}`;
    }
    if (!body) return lora;
    if (body === lora || body.startsWith(`${lora},`) || body.startsWith(`${lora}\n`)) return body;
    return lora.endsWith(",") ? `${lora} ${body}`.trim() : `${lora}, ${body}`;
}

function fillExecPrompt(el, editor, seg) {
    if (!el) return;
    const text = executedPromptText(editor, seg);
    el.textContent = text;
    el.dataset.empty = t("batch.execPromptEmpty");
}

function samplingPreviewLabel(seg) {
    return seg?.previewPass === "second"
        ? t("liveSample.secondPass")
        : t("liveSample.firstPass");
}

function wrapPreviewColumn(preview, editor, seg) {
    const col = document.createElement("div");
    col.className = "bd-preview-col";
    const promptOpen = !!seg?.execPromptOpen;
    if (promptOpen) col.classList.add("exec-open");
    const head = document.createElement("div");
    head.className = "bd-preview-col-head";
    const sampleBtn = document.createElement("button");
    sampleBtn.type = "button";
    sampleBtn.className = "bd-preview-tab" + (promptOpen ? "" : " active");
    sampleBtn.setAttribute("data-r", "preview-tab-sample");
    sampleBtn.setAttribute("data-i18n", seg?.previewPass === "second" ? "liveSample.secondPass" : "liveSample.firstPass");
    sampleBtn.textContent = samplingPreviewLabel(seg);
    const promptBtn = document.createElement("button");
    promptBtn.type = "button";
    promptBtn.className = "bd-preview-tab" + (promptOpen ? " active" : "");
    promptBtn.setAttribute("data-r", "preview-tab-prompt");
    promptBtn.setAttribute("data-i18n", "batch.promptPreview");
    promptBtn.textContent = t("batch.promptPreview");
    const setPane = (nextPrompt) => {
        const live = (editor.timeline?.segments || []).find((s) => s?.id && s.id === seg?.id) || seg;
        if (live) live.execPromptOpen = nextPrompt;
        col.classList.toggle("exec-open", nextPrompt);
        sampleBtn.classList.toggle("active", !nextPrompt);
        promptBtn.classList.toggle("active", nextPrompt);
    };
    sampleBtn.onclick = (e) => {
        e.stopPropagation();
        setPane(false);
    };
    promptBtn.onclick = (e) => {
        e.stopPropagation();
        setPane(true);
    };
    head.appendChild(sampleBtn);
    head.appendChild(promptBtn);
    col.appendChild(head);
    col.appendChild(preview);
    const execWrap = document.createElement("div");
    execWrap.className = "bd-exec-prompt-wrap";
    const exec = document.createElement("div");
    exec.className = "bd-exec-prompt";
    exec.setAttribute("data-r", "exec-prompt");
    fillExecPrompt(exec, editor, seg);
    execWrap.appendChild(exec);
    col.appendChild(execWrap);
    return col;
}

export function isBatchDetailSolo(editor) {
    return (editor?.timeline?.batchDetailMode || "solo") !== "all";
}

function syncBatchDetailModeButton(editor) {
    const btn = editor?.batchDetailModeBtn;
    if (!btn) return;
    const solo = isBatchDetailSolo(editor);
    btn.textContent = t(solo ? "toolbar.batchDetailSolo" : "toolbar.batchDetailAll");
    btn.title = t(solo ? "tooltip.batchDetailSolo" : "tooltip.batchDetailAll");
    btn.setAttribute("data-i18n", solo ? "toolbar.batchDetailSolo" : "toolbar.batchDetailAll");
    btn.setAttribute("data-i18n-title", solo ? "tooltip.batchDetailSolo" : "tooltip.batchDetailAll");
}

export function toggleBatchDetailMode(editor) {
    if (!editor?.timeline) return;
    flushBatchPromptInputs(editor);
    flushBatchDurationInputs(editor);
    editor.timeline.batchDetailMode = isBatchDetailSolo(editor) ? "all" : "solo";
    syncBatchDetailModeButton(editor);
    editor.renderImageBatchGroups?.();
    editor.commit?.(false, { syncTimeline: true });
    editor.updateDomWidgetHeight?.();
}

export function selectBatchGroup(editor, index) {
    const segs = editor?.timeline?.segments || [];
    if (!segs.length) return;
    const next = Math.max(0, Math.min(segs.length - 1, Number(index) || 0));
    const expanded = expandBatchGroup(editor, next);
    if (next === editor.selectedIndex) {
        if (expanded) editor.renderImageBatchGroups?.();
        editor._syncR2vCardSelection?.();
        return;
    }
    flushBatchPromptInputs(editor);
    flushBatchDurationInputs(editor);
    editor.selectedIndex = next;
    if (isBatchDetailSolo(editor)) {
        editor.renderImageBatchGroups?.();
    } else {
        editor._syncR2vCardSelection?.();
        editor.scheduleRender?.();
    }
    editor.updateVideoNameLabel?.();
}

export function expandBatchGroup(editor, index) {
    const seg = editor?.timeline?.segments?.[index];
    if (!seg?.uiCollapsed) return false;
    seg.uiCollapsed = false;
    editor.scheduleTimelineSync?.();
    return true;
}

export function toggleBatchGroupCollapsed(editor, index) {
    const seg = editor?.timeline?.segments?.[index];
    if (!seg) return false;
    seg.uiCollapsed = !seg.uiCollapsed;
    editor.scheduleTimelineSync?.();
    return true;
}

function batchCardEl(editor, segmentIndex) {
    return editor?.batchList?.querySelector?.(`.bd-batch-card[data-batch-index="${segmentIndex}"]`);
}

function renderBatchGroupPicker(editor, ctx) {
    const picker = editor.batchPicker;
    if (!picker) return;
    const segs = editor.timeline?.segments || [];
    const solo = isBatchDetailSolo(editor);
    const usePicker = solo && segs.length > 1 && !editor.usesBatchTimeline?.();
    picker.innerHTML = "";
    picker.classList.toggle("visible", usePicker);
    if (!usePicker) return;
    const runSelectOn = !!(editor.isRunSelectEnabled?.() && editor.supportsRunSelect?.());
    const { key, isVideo, runningIdx, externalLocked } = ctx;
    segs.forEach((seg, index) => {
        const chip = document.createElement("div");
        chip.setAttribute("role", "button");
        chip.tabIndex = 0;
        chip.className = "bd-batch-pick";
        chip.dataset.batchIndex = String(index);
        const runEnabled = !runSelectOn || !!editor.isSegmentRunEnabled?.(index);
        if (index === editor.selectedIndex) chip.classList.add("selected");
        if (index === runningIdx) chip.classList.add("running");
        if (runSelectOn && !runEnabled) chip.classList.add("run-skipped");
        const head = document.createElement("div");
        head.className = "bd-batch-pick-title";
        if (runSelectOn) {
            const runCb = document.createElement("input");
            runCb.type = "checkbox";
            runCb.className = "bd-batch-run-check";
            runCb.checked = runEnabled;
            runCb.title = t("tooltip.batchRunCheck");
            runCb.onclick = (e) => {
                e.stopPropagation();
                editor.toggleSegmentRun(index);
            };
            head.appendChild(runCb);
        }
        const title = document.createElement("span");
        const chipKey = ctx.mixed ? resolveMixedGroupKey(seg) : key;
        title.textContent = t(
            ctx.mixed ? "batch.groupTitle.mixed" : (chipKey === "r2v" ? "batch.groupTitle.asset" : "batch.groupTitle.prompt"),
            { n: index + 1 },
        );
        head.appendChild(title);
        chip.appendChild(head);
        const meta = document.createElement("span");
        meta.className = "bd-batch-pick-meta";
        if (isVideo) {
            const sec = resolveSegmentDurationSec(seg);
            meta.textContent = `${Number(sec).toFixed(1)}s`;
        } else {
            meta.textContent = `#${index + 1}`;
        }
        chip.appendChild(meta);
        const thumbSrc = seg.previewB64 || (Array.isArray(seg.previewFrames) ? seg.previewFrames[0] : "");
        if (thumbSrc) {
            const img = document.createElement("img");
            img.className = "bd-batch-pick-thumb";
            img.alt = "";
            img.src = frameSrc(thumbSrc);
            chip.appendChild(img);
        }
        chip.onclick = (e) => {
            if (e.target.closest?.("input")) return;
            selectBatchGroup(editor, index);
        };
        if (externalLocked) chip.title = t("external.durationLocked");
        picker.appendChild(chip);
    });
}

function _batchWidgetValue(node, name, fallback) {
    const w = node?.widgets?.find((item) => item?.name === name);
    if (!w) return fallback;
    let v = w.value;
    if (v && typeof v === "object") {
        if (typeof v.content === "string") v = v.content;
        else if (typeof v.value === "string") v = v.value;
    }
    return v == null || v === "" ? fallback : v;
}

function directorHasRefineLink(node) {
    const inp = (node?.inputs || []).find((item) => item?.name === "refine");
    if (!inp) return false;
    if (inp.link != null) return true;
    return Array.isArray(inp.links) && inp.links.length > 0;
}

function _boolWidget(node, name) {
    const value = _batchWidgetValue(node, name, false);
    if (value === true || value === 1) return true;
    const text = String(value ?? "").trim().toLowerCase();
    return text === "true" || text === "yes" || text === "on";
}

export function directorRefineActive(node) {
    if (directorHasRefineLink(node)) return true;
    return _boolWidget(node, "refine_enable");
}

export function defaultBatchPassMode(node) {
    return directorRefineActive(node) ? "second" : "first";
}

export function applyDirectorRefinePassDefaults(editor, enabled) {
    if (!editor?.timeline?.segments) return;
    const mode = enabled ? "second" : "first";
    let changed = false;
    for (const seg of editor.timeline.segments) {
        if (!seg) continue;
        if (seg.passMode !== mode) {
            seg.passMode = mode;
            changed = true;
        }
    }
    const list = editor.batchList;
    if (list) {
        for (const wrap of list.querySelectorAll(".bd-batch-pass")) {
            wrap.setAttribute("data-pass-mode", mode);
        }
    }
    syncBatchPassButtons(editor);
    if (changed) {
        editor.commit?.(false, { syncTimeline: true });
        editor.flushTimelineSync?.();
    }
}

function directorHasInputLink(node, name) {
    const inp = (node?.inputs || []).find((item) => String(item?.name) === String(name));
    if (!inp) return false;
    if (inp.link != null) return true;
    return Array.isArray(inp.links) && inp.links.length > 0;
}

function directorHasSigmasLink(node) {
    return directorHasInputLink(node, "sigmas");
}

function _selfliftDrawerEl(editor, name) {
    return editor?.selfLiftPanelEl?.querySelector(`[data-w="${name}"]`);
}

function _selfliftCacheFlag(editor, name, fallback = false) {
    const node = editor?.node;
    if (node?.widgets?.some((item) => item?.name === name)) {
        return _boolWidget(node, name);
    }
    const el = _selfliftDrawerEl(editor, name);
    if (el) return el.type === "checkbox" ? !!el.checked : _boolWidget({ widgets: [{ name, value: el.value }] }, name);
    return !!fallback;
}

function _selfliftCacheValue(editor, name, fallback) {
    const node = editor?.node;
    if (node?.widgets?.some((item) => item?.name === name)) {
        return _batchWidgetValue(node, name, fallback);
    }
    const el = _selfliftDrawerEl(editor, name);
    if (el && el.type === "checkbox") return el.checked ? "true" : "false";
    if (el && el.value != null && el.value !== "") return el.value;
    return fallback;
}

function passCachePayload(editor, index) {
    const node = editor?.node;
    if (!node) return null;
    try {
        editor._writeTimelineWidget?.();
    } catch {
        /* best effort */
    }
    const payload = {
        node_id: String(node.id),
        timeline_data: String(_batchWidgetValue(node, "timeline_data", "")),
        task_type: String(_batchWidgetValue(node, "task_type", "")),
        global_prompt: String(_batchWidgetValue(node, "global_prompt", "")),
        total_frames: Number(_batchWidgetValue(node, "total_frames", 124)),
        frame_rate: 24,
        width: Number(_batchWidgetValue(node, "width", 864)),
        height: Number(_batchWidgetValue(node, "height", 480)),
        ref_max_size: Number(_batchWidgetValue(node, "ref_max_size", 864)),
        seed: Number(_batchWidgetValue(node, "seed", 0)),
        cfg: Number(_batchWidgetValue(node, "cfg", 1)),
        steps: Number(_batchWidgetValue(node, "steps", 25)),
        sampler: String(_batchWidgetValue(node, "sampler", "")),
        scheduler: String(_batchWidgetValue(node, "scheduler", "")),
        shift_video: Number(_batchWidgetValue(node, "shift_video", 12)),
        shift_audio: Number(_batchWidgetValue(node, "shift_audio", 3)),
        sigmas_linked: directorHasSigmasLink(node),
        lora_trigger_words: readLoraTriggerWords(editor),
        selflift_enable: _selfliftCacheFlag(editor, "selflift_enable"),
        selflift_split_mode: _selfliftCacheValue(editor, "selflift_split_mode", "highres_steps"),
        selflift_highres_steps: _selfliftCacheValue(editor, "selflift_highres_steps", "2"),
        selflift_transition_step: _selfliftCacheValue(editor, "selflift_transition_step", "6"),
        selflift_lowres_scale: _selfliftCacheValue(editor, "selflift_lowres_scale", "0.5"),
        selflift_latent_upscale_model: _selfliftCacheValue(editor, "selflift_latent_upscale_model", ""),
        selflift_native_low_carry: _selfliftCacheFlag(editor, "selflift_native_low_carry", true),
        selflift_sampler_mode: _selfliftCacheValue(editor, "selflift_sampler_mode", "euler"),
        selflift_rho: _selfliftCacheValue(editor, "selflift_rho", "0"),
        selflift_w_min: _selfliftCacheValue(editor, "selflift_w_min", "0.5"),
        selflift_w_max: _selfliftCacheValue(editor, "selflift_w_max", "1"),
        selflift_latent_upsample: _selfliftCacheValue(editor, "selflift_latent_upsample", "bilinear"),
        selflift_enable_latent_chunking: _selfliftCacheFlag(editor, "selflift_enable_latent_chunking"),
        selflift_enable_tiling: _selfliftCacheFlag(editor, "selflift_enable_tiling"),
        selflift_tile_count: _selfliftCacheValue(editor, "selflift_tile_count", "2"),
        selflift_tile_overlap: _selfliftCacheValue(editor, "selflift_tile_overlap", "128"),
        selflift_model_hires: directorHasInputLink(node, "selflift_model_hires"),
        semantic_bridge_enable: _selfliftCacheFlag(editor, "semantic_bridge_enable"),
        semantic_bridge_adapter: _selfliftCacheValue(editor, "semantic_bridge_adapter", ""),
        semantic_bridge_alpha: _selfliftCacheValue(editor, "semantic_bridge_alpha", "0.15"),
        semantic_bridge_magnitude_match: _selfliftCacheFlag(editor, "semantic_bridge_magnitude_match", true),
    };
    if (r2vLoraTriggerIsLinked(editor)) {
        payload.lora_trigger_words_r2v = readLoraTriggerWords(editor, "lora_trigger_words_r2v");
    }
    if (Number.isInteger(index) && index >= 0) payload.cache_index = index;
    return payload;
}

function passCacheRowForIndex(data, index) {
    const rows = Array.isArray(data?.segments) ? data.segments : [];
    return rows.find((row) => Number(row?.ui_index ?? row?.index) === Number(index))
        || rows.find((row) => Number(row?.segment) === Number(index) + 1)
        || null;
}

function passDiffLabel(key) {
    const mapped = t(`batch.pass.diff.${key}`);
    if (mapped && mapped !== `batch.pass.diff.${key}`) return mapped;
    return key;
}

function passStatusKind(row) {
    if (!row) return "pending";
    if (row.error) return "error";
    const status = String(row.status || "");
    if (["valid", "mismatch", "missing", "unchecked", "pending"].includes(status)) return status;
    if (row.matches) return "valid";
    if (row.exists) return "mismatch";
    return "missing";
}

let _passPop = null;
let _passPopCloser = null;

function closePassCachePopover() {
    _passPop?.remove();
    _passPop = null;
    if (_passPopCloser) {
        document.removeEventListener("pointerdown", _passPopCloser, true);
        _passPopCloser = null;
    }
}

function showPassCachePopover(anchor, row) {
    closePassCachePopover();
    const kind = passStatusKind(row);
    const lines = [t(`batch.pass.status.${kind}`)];
    if (row?.cached_seed != null && row.cached_seed !== "") {
        lines.push(t("batch.pass.cachedSeed", { seed: row.cached_seed }));
    }
    const diffs = Array.isArray(row?.diff_keys)
        ? row.diff_keys.filter((key) => key !== "<missing-cache>").map(passDiffLabel)
        : [];
    if (kind === "mismatch" && diffs.length) {
        lines.push(t("batch.pass.diffs", { diffs: diffs.join(", ") }));
    }
    if (row?.error) lines.push(String(row.error));
    const pop = document.createElement("div");
    pop.className = "bd-batch-pass-pop";
    pop.style.color = {
        valid: "#65d68a",
        mismatch: "#f0bd58",
        error: "#ef7777",
        missing: "#aaa",
    }[kind] || "#ddd";
    pop.textContent = lines.join("\n");
    document.body.appendChild(pop);
    const rect = anchor.getBoundingClientRect();
    const pad = 8;
    let left = rect.left;
    let top = rect.bottom + 6;
    const width = pop.offsetWidth || 240;
    const height = pop.offsetHeight || 48;
    if (left + width + pad > window.innerWidth) left = Math.max(pad, window.innerWidth - width - pad);
    if (top + height + pad > window.innerHeight) top = Math.max(pad, rect.top - height - 6);
    pop.style.left = `${Math.round(left)}px`;
    pop.style.top = `${Math.round(top)}px`;
    _passPop = pop;
    _passPopCloser = (event) => {
        if (pop.contains(event.target) || anchor.contains(event.target)) return;
        closePassCachePopover();
    };
    document.addEventListener("pointerdown", _passPopCloser, true);
}

function paintPassCacheStatus(editor, data) {
    const list = editor?.batchList;
    if (!list) return;
    const previous = Array.isArray(editor._mmxPassCache?.segments)
        ? editor._mmxPassCache.segments
        : [];
    const incoming = Array.isArray(data?.segments) ? data.segments : [];
    const rows = new Map(previous.map((row) => [Number(row?.ui_index ?? row?.index), row]));
    for (const row of incoming) rows.set(Number(row?.ui_index ?? row?.index), row);
    editor._mmxPassCache = { ...(editor._mmxPassCache || {}), ...(data || {}), segments: [...rows.values()] };
    for (const el of list.querySelectorAll("[data-batch-pass-status]")) {
        const index = Number(el.getAttribute("data-batch-pass-index"));
        const row = passCacheRowForIndex(editor._mmxPassCache, index);
        const selected = editor.isSegmentRunEnabled?.(index) !== false;
        const kind = selected ? passStatusKind(row) : "unchecked";
        el.className = `bd-batch-pass-status ${kind}`;
        el.title = row?.error ? String(row.error) : t(`batch.pass.status.${kind}`);
        el._mmxPassRow = row;
    }
}

function syncBatchPassButtons(editor) {
    const hasRefine = directorRefineActive(editor?.node);
    const list = editor?.batchList;
    if (!list) return hasRefine;
    for (const wrap of list.querySelectorAll(".bd-batch-pass")) {
        const firstBtn = wrap.querySelector("[data-batch-pass-first]");
        const secondBtn = wrap.querySelector("[data-batch-pass-second]");
        if (!firstBtn || !secondBtn) continue;
        secondBtn.disabled = !hasRefine;
        secondBtn.title = hasRefine ? t("batch.pass.tooltip.second") : t("batch.pass.secondDisabled");
        const stored = wrap.getAttribute("data-pass-mode") === "first" ? "first" : "second";
        const visual = hasRefine ? stored : "first";
        firstBtn.classList.toggle("active", visual === "first");
        secondBtn.classList.toggle("active", visual === "second");
    }
    return hasRefine;
}

async function refreshBatchPassCacheStatus(editor, index) {
    if (!editor?.node || !editor.batchList?.querySelector("[data-batch-pass-status]")) return;
    // A status refresh is only informational.  Never let an older filesystem
    // scan remain in flight while a newer one is requested after UI edits.
    editor._mmxPassCacheAbort?.abort();
    const controller = new AbortController();
    editor._mmxPassCacheAbort = controller;
    const seq = (editor._mmxPassCacheSeq || 0) + 1;
    editor._mmxPassCacheSeq = seq;
    const payload = passCachePayload(editor, index);
    if (!payload) return;
    const batch = index === "selected";
    for (const status of editor.batchList.querySelectorAll("[data-batch-pass-status]")) {
        const itemIndex = Number(status.getAttribute("data-batch-pass-index"));
        if (!batch && itemIndex !== index) continue;
        const selected = editor.isSegmentRunEnabled?.(itemIndex) !== false;
        status.className = `bd-batch-pass-status ${selected ? "checking" : "unchecked"}`;
        status.title = t(`batch.pass.status.${selected ? "checking" : "unchecked"}`);
    }
    try {
        const response = await api.fetchApi("/minimax/director/first_pass_cache_status", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: controller.signal,
            body: JSON.stringify(payload),
        });
        const data = await response.json().catch(() => ({}));
        if (seq !== editor._mmxPassCacheSeq) return;
        if (!response.ok || data?.error) {
            throw new Error(data?.error || `HTTP ${response.status}`);
        }
        paintPassCacheStatus(editor, data);
    } catch (error) {
        if (error?.name === "AbortError") return;
        if (seq !== editor._mmxPassCacheSeq) return;
        const message = error?.message || String(error);
        const selectedVisibleIndices = [...editor.batchList.querySelectorAll("[data-batch-pass-status]")]
            .map((status) => Number(status.getAttribute("data-batch-pass-index")))
            .filter((itemIndex) => editor.isSegmentRunEnabled?.(itemIndex) !== false);
        paintPassCacheStatus(editor, {
            segments: batch
                ? selectedVisibleIndices.map((ui_index) => ({ ui_index, status: "error", error: message }))
                : [{ ui_index: index, status: "error", error: message }],
        });
    }
}

export function scheduleDirectorPassCacheRefresh(nodeOrEditor, delay = 120, index = null) {
    const editor = nodeOrEditor?._minimaxEditor || nodeOrEditor;
    if (!editor?.batchList) return;
    closePassCachePopover();
    syncBatchPassButtons(editor);
    const batch = index === "selected";
    if (!batch && (!Number.isInteger(index) || index < 0)) {
        editor._mmxPassCache = null;
        for (const el of editor.batchList.querySelectorAll("[data-batch-pass-status]")) {
            const itemIndex = Number(el.getAttribute("data-batch-pass-index"));
            const selected = editor.isSegmentRunEnabled?.(itemIndex) !== false;
            const kind = selected ? "pending" : "unchecked";
            el.className = `bd-batch-pass-status ${kind}`;
            el.title = t(`batch.pass.status.${kind}`);
            el._mmxPassRow = null;
        }
        return;
    }
    for (const status of editor.batchList.querySelectorAll("[data-batch-pass-status]")) {
        const itemIndex = Number(status.getAttribute("data-batch-pass-index"));
        if (!batch && itemIndex !== index) continue;
        const selected = editor.isSegmentRunEnabled?.(itemIndex) !== false;
        status.className = `bd-batch-pass-status ${selected ? "checking" : "unchecked"}`;
        status.title = t(`batch.pass.status.${selected ? "checking" : "unchecked"}`);
    }
    clearTimeout(editor._mmxPassCacheTimer);
    editor._mmxPassCacheTimer = setTimeout(() => refreshBatchPassCacheStatus(editor, index), delay);
}

function hasDuplicateGroupMedia(list, mediaPath, slot) {
    const target = String(mediaPath || "").replace(/\\/g, "/").toLowerCase();
    if (!target) return false;
    return (list || []).some((item) => {
        if (Number(item?.index ?? item?.slot) === Number(slot)) return false;
        const path = item?.imageFile || item?.videoFile || item?.audioFile || "";
        return String(path).replace(/\\/g, "/").toLowerCase() === target;
    });
}

function commitSegmentPassMode(editor, index, mode) {
    const segs = editor?.timeline?.segments;
    const live = segs?.[index];
    if (!live) return;
    live.passMode = mode === "first" ? "first" : "second";
    const wrap = editor.batchList?.querySelector(`[data-batch-pass-index="${index}"]`)?.closest?.(".bd-batch-pass");
    if (wrap) wrap.setAttribute("data-pass-mode", live.passMode);
    syncBatchPassButtons(editor);
    editor.commit?.(false, { syncTimeline: true });
    editor.flushTimelineSync?.();
}

function commitSegmentForceResample(editor, index, checked) {
    const live = editor?.timeline?.segments?.[index];
    if (!live) return;
    live.forceResample = !!checked;
    editor.commit?.(false, { syncTimeline: true });
    editor.flushTimelineSync?.();
}

async function clearGroupFirstPassCache(editor, index) {
    const node = editor?.node;
    if (!node) return;
    if (!await editor.showBdDialog?.({ title: t("batch.pass.clear"), message: t("batch.pass.clearConfirm"), confirmText: t("dialog.confirm"), cancelText: t("dialog.cancel") })) return;
    try {
        const response = await api.fetchApi("/minimax/director/clear_segment_cache", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                node_id: String(node.id),
                kind: "first_pass",
                index,
            }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || data?.error) {
            throw new Error(data?.error || `HTTP ${response.status}`);
        }
        closePassCachePopover();
        scheduleDirectorPassCacheRefresh(editor, 80, index);
    } catch (error) {
        await editor.showBdMessage?.(t("batch.pass.clear"), error?.message || String(error));
    }
}

export async function clearAllDirectorCache(editor) {
    const node = editor?.node;
    if (!node || !await editor.showBdDialog?.({ title: t("batch.cache.clearAll"), message: t("batch.cache.clearAllConfirm"), confirmText: t("dialog.confirm"), cancelText: t("dialog.cancel") })) return;
    try {
        const response = await api.fetchApi("/minimax/director/clear_segment_cache", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ node_id: String(node.id), kind: "all" }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || data?.error) throw new Error(data?.error || `HTTP ${response.status}`);
        scheduleDirectorPassCacheRefresh(editor, 80, "selected");
        editor.renderImageBatchGroups?.();
    } catch (error) {
        await editor.showBdMessage?.(t("batch.cache.clearAll"), error?.message || String(error));
    }
}

function appendBatchPassControls(meta, editor, seg, index) {
    const wrap = document.createElement("div");
    wrap.className = "bd-batch-pass";
    wrap.setAttribute("data-pass-mode", resolveSegmentPassMode(seg));
    wrap.onclick = (e) => e.stopPropagation();
    const forceLabel = document.createElement("label");
    forceLabel.className = "bd-batch-force-resample";
    forceLabel.title = t("batch.pass.tooltip.forceResample");
    const forceInput = document.createElement("input");
    forceInput.type = "checkbox";
    forceInput.checked = !!seg.forceResample;
    forceInput.setAttribute("data-batch-force-resample", "");
    forceInput.onchange = (e) => {
        e.stopPropagation();
        commitSegmentForceResample(editor, index, forceInput.checked);
    };
    const forceText = document.createElement("span");
    forceText.textContent = t("batch.pass.forceResample");
    forceLabel.append(forceInput, forceText);
    const firstBtn = document.createElement("button");
    firstBtn.type = "button";
    firstBtn.className = "bd-batch-pass-btn";
    firstBtn.setAttribute("data-batch-pass-first", "");
    firstBtn.textContent = t("batch.pass.first");
    firstBtn.title = t("batch.pass.tooltip.first");
    firstBtn.onclick = (e) => {
        e.stopPropagation();
        commitSegmentPassMode(editor, index, "first");
    };
    const secondBtn = document.createElement("button");
    secondBtn.type = "button";
    secondBtn.className = "bd-batch-pass-btn";
    secondBtn.setAttribute("data-batch-pass-second", "");
    secondBtn.textContent = t("batch.pass.second");
    secondBtn.title = t("batch.pass.tooltip.second");
    secondBtn.onclick = (e) => {
        e.stopPropagation();
        if (secondBtn.disabled) return;
        commitSegmentPassMode(editor, index, "second");
    };
    const status = document.createElement("button");
    status.type = "button";
    status.className = "bd-batch-pass-status pending";
    status.setAttribute("data-batch-pass-status", "");
    status.setAttribute("data-batch-pass-index", String(index));
    status.title = t("batch.pass.status.pending");
    status.onmouseenter = () => {
        const row = status._mmxPassRow || passCacheRowForIndex(editor._mmxPassCache, index);
        showPassCachePopover(status, row);
    };
    status.onmouseleave = () => {
        closePassCachePopover();
    };
    status.onclick = (e) => {
        e.stopPropagation();
        // Hover is informational; an explicit click always refreshes the
        // filesystem-backed status and does not open a stale popover.
        scheduleDirectorPassCacheRefresh(editor, 0, index);
    };
    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "bd-batch-pass-clear";
    clearBtn.textContent = t("batch.pass.clear");
    clearBtn.title = t("batch.pass.tooltip.clear");
    clearBtn.onclick = (e) => {
        e.stopPropagation();
        void clearGroupFirstPassCache(editor, index);
    };
    wrap.append(forceLabel, firstBtn, secondBtn, status, clearBtn);
    meta.appendChild(wrap);
}

export function renderImageBatchGroups(editor) {
    const list = editor.batchList;
    if (!list) return;
    // Recover drafts before wiping the DOM (duration flush also flushes prompts).
    flushBatchPromptInputs(editor);
    flushBatchDurationInputs(editor);
    stopAllPlayers(list);
    const key = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    const mixed = isMixedTask(key);
    const variant = imageBatchVariant(key);
    const isVideo = isVideoBatchTask(key) || mixed;
    const runningIdx = editor._runHighlightSeg;
    const fps = parseFloat(editor.frameRateWidget?.value || editor.timeline?.frameRate || 24);

    if (editor.batchHint) {
        const hintKey = mixed ? "batch.hint.mixed" : `batch.hint.${key}`;
        editor.batchHint.textContent = t(hintKey) !== hintKey
            ? t(hintKey)
            : t(isVideo ? "batch.hint.defaultVideo" : "batch.hint.defaultImage");
    }
    const externalLocked = !!(editor.hasExternalI2vGroups?.() || editor.hasExternalR2vGroups?.());
    const addBtn = editor.batchPanel?.querySelector('[data-a="batch-add"]');
    if (addBtn) {
        // Video batch modes add from the shared top toolbar, matching mixed mode.
        // External groups: never add UI cards (graph is source of truth).
        addBtn.classList.toggle("hidden", isVideo || externalLocked);
        addBtn.disabled = externalLocked;
    }

    teardownPromptImageMentions(list);
    list.innerHTML = "";
    const ctx = { key, mixed, variant, isVideo, runningIdx, fps, externalLocked };
    const segs = editor.timeline.segments || [];
    if (editor.selectedIndex == null || editor.selectedIndex < 0 || editor.selectedIndex >= segs.length) {
        editor.selectedIndex = 0;
    }
    syncBatchDetailModeButton(editor);
    renderBatchGroupPicker(editor, ctx);
    const solo = isBatchDetailSolo(editor);
    const indices = solo ? [editor.selectedIndex] : segs.map((_, i) => i);
    for (const index of indices) {
        const seg = segs[index];
        if (!seg) continue;
        appendBatchCard(list, editor, seg, index, ctx);
    }
    updateR2vToolbarBtns(editor);
    refreshPromptTokenEditors(list);
    editor.updateDomWidgetHeight?.();
    if (editor._batchAutosizeFrame) cancelAnimationFrame(editor._batchAutosizeFrame);
    editor._batchAutosizeFrame = requestAnimationFrame(() => {
        editor._batchAutosizeFrame = 0;
        editor.resizeNodeForContentMinChange?.();
    });
    closePassCachePopover();
    syncBatchPassButtons(editor);
    // Re-rendering the batch cards recreates the status buttons.  Reapply the
    // last completed scan instead of leaving newly-created buttons gray until
    // the next filesystem request.
    if (editor._mmxPassCache) {
        paintPassCacheStatus(editor, editor._mmxPassCache);
    }
    // Cache status is refreshed when the cache popover is opened or after an
    // explicit cache operation, not after every repaint of the batch panel.
    restoreSlotLoadOverlays(editor);
    if (!editor._mmxPassCacheInitialCheckStarted) {
        editor._mmxPassCacheInitialCheckStarted = true;
        scheduleDirectorPassCacheRefresh(editor, 250, "selected");
    }
}

function appendBatchCard(list, editor, seg, index, ctx) {
        const { key, mixed, isVideo, runningIdx, fps, externalLocked } = ctx;
        const cardKey = mixed ? resolveMixedGroupKey(seg) : key;
        const isRv2v = cardKey === "rv2v";
        const isVideoEdit = cardKey === "v2v" || isRv2v;
        const variant = isRv2v ? "refs" : imageBatchVariant(cardKey);
        const isR2v = cardKey === "r2v" || isRv2v;
        const isFl2v = cardKey === "fl2v";
        const card = document.createElement("div");
        const layoutClass = isR2v
            ? "bd-batch-r2v"
            : (isVideoEdit ? "bd-batch-v2v"
                : (isFl2v ? "bd-batch-fl2v"
                : (variant === "source" ? "bd-batch-source"
                    : (variant === "refs" ? "bd-batch-refs" : "bd-batch-plain"))));
        card.className = `bd-batch-card ${layoutClass} bd-batch-task-${cardKey}${mixed ? " bd-batch-mixed" : ""}`;
        card.dataset.batchIndex = String(index);
        if (seg.id) card.dataset.batchSegId = String(seg.id);
        const collapsed = !!seg.uiCollapsed;
        card.classList.toggle("bd-batch-collapsed", collapsed);
        const savedCardHeight = Number(seg.uiCardHeight);
        if (!collapsed && Number.isFinite(savedCardHeight) && savedCardHeight > 0) {
            card.style.height = `${Math.round(savedCardHeight)}px`;
            card.style.flex = "0 0 auto";
        }
        card.addEventListener("pointerdown", (event) => {
            if (event.button !== 0) return;
            const rect = card.getBoundingClientRect();
            const gripSize = 24;
            const onResizeGrip = event.clientX >= rect.right - gripSize
                && event.clientY >= rect.bottom - gripSize;
            if (!onResizeGrip) return;
            const startHeight = card.offsetHeight;
            const previousHeight = card.style.height;
            const previousFlex = card.style.flex;
            card.style.height = `${startHeight}px`;
            card.style.flex = "0 0 auto";
            card.dataset.userResizing = "1";
            let resizeFrame = 0;
            const syncNodeHeight = () => {
                if (resizeFrame) return;
                resizeFrame = requestAnimationFrame(() => {
                    resizeFrame = 0;
                    editor.resizeNodeForContentMinChange?.();
                });
            };
            const finishResize = () => {
                window.removeEventListener("pointerup", finishResize, true);
                window.removeEventListener("pointercancel", finishResize, true);
                window.removeEventListener("pointermove", syncNodeHeight, true);
                if (resizeFrame) cancelAnimationFrame(resizeFrame);
                const nextHeight = card.offsetHeight;
                delete card.dataset.userResizing;
                if (Math.abs(nextHeight - startHeight) < 2) {
                    card.style.height = previousHeight;
                    card.style.flex = previousFlex;
                    return;
                }
                const live = (editor.timeline.segments || []).find((item) => item?.id && item.id === seg.id)
                    || editor.timeline.segments?.[index];
                if (!live || !card.isConnected) return;
                live.uiCardHeight = nextHeight;
                editor.scheduleTimelineSync?.();
                editor.flushTimelineSync?.();
                editor.resizeNodeForContentMinChange?.();
            };
            window.addEventListener("pointermove", syncNodeHeight, true);
            window.addEventListener("pointerup", finishResize, true);
            window.addEventListener("pointercancel", finishResize, true);
        });
        const runSelectOn = !!(editor.isRunSelectEnabled?.() && editor.supportsRunSelect?.());
        const runEnabled = !runSelectOn || !!editor.isSegmentRunEnabled?.(index);
        // r2v / mixed: always show focus selected. t2v/i2v: only run-select participation chrome.
        if ((isR2v || mixed) && index === editor.selectedIndex) {
            card.classList.add("selected");
        }
        if (index === runningIdx) card.classList.add("running");
        if (runSelectOn && runEnabled) card.classList.add("run-on");
        if (runSelectOn && !runEnabled) card.classList.add("run-skipped");
        card.onclick = (e) => {
            if (e.target.closest?.("button, input, textarea, select, .bd-batch-ref, .bd-batch-audio, .bd-batch-video, .bd-batch-src, .bd-batch-fl2v-slots, .bd-r2v-section, .bd-r2v-play, .x, video, audio")) {
                return;
            }
            selectBatchGroup(editor, index);
        };
        const hasPreview = isVideo
            ? (seg.previewFrames?.length > 0 || seg.previewB64)
            : !!seg.previewB64;
        if (hasPreview && index !== runningIdx) card.classList.add("done");

        const head = document.createElement("div");
        head.className = "bd-batch-head";
        const collapseBtn = document.createElement("button");
        collapseBtn.type = "button";
        collapseBtn.className = "bd-batch-collapse";
        collapseBtn.textContent = collapsed ? "▸" : "▾";
        collapseBtn.title = t(collapsed ? "tooltip.batchGroupExpand" : "tooltip.batchGroupCollapse");
        collapseBtn.setAttribute("aria-expanded", String(!collapsed));
        collapseBtn.onclick = (event) => {
            event.stopPropagation();
            const live = (editor.timeline.segments || []).find((item) => item?.id && item.id === seg.id)
                || editor.timeline.segments?.[index];
            if (!live) return;
            live.uiCollapsed = !live.uiCollapsed;
            editor.scheduleTimelineSync?.();
            editor.flushTimelineSync?.();
            editor.renderImageBatchGroups?.();
            editor.resizeNodeForContentMinChange?.();
        };
        head.appendChild(collapseBtn);
        // Timeline + cards stay in sync for run-select (incl. r2v).
        if (runSelectOn) {
            const runCb = document.createElement("input");
            runCb.type = "checkbox";
            runCb.className = "bd-batch-run-check";
            runCb.checked = runEnabled;
            runCb.title = t("tooltip.batchRunCheck");
            runCb.onclick = (e) => {
                e.stopPropagation();
                editor.toggleSegmentRun(index);
            };
            head.appendChild(runCb);
        }
        const title = document.createElement("b");
        const defaultTitle = t(
            mixed ? "batch.groupTitle.mixed" : (isR2v ? "batch.groupTitle.asset" : "batch.groupTitle.prompt"),
            { n: index + 1 },
        );
        title.textContent = String(seg.uiGroupName || "").trim() || defaultTitle;
        title.title = t("tooltip.batchGroupRename");
        title.ondblclick = (event) => {
            event.preventDefault();
            event.stopPropagation();
            const input = document.createElement("input");
            input.type = "text";
            input.className = "bd-batch-title-input";
            input.value = String(seg.uiGroupName || "").trim() || defaultTitle;
            title.replaceWith(input);
            input.focus();
            input.select();
            let cancelled = false;
            const finish = () => {
                const live = (editor.timeline.segments || []).find((item) => item?.id && item.id === seg.id)
                    || editor.timeline.segments?.[index];
                if (live && !cancelled) {
                    const name = input.value.trim();
                    live.uiGroupName = name === defaultTitle ? "" : name;
                    editor.scheduleTimelineSync?.();
                    editor.flushTimelineSync?.();
                }
                title.textContent = String(live?.uiGroupName || "").trim() || defaultTitle;
                input.replaceWith(title);
            };
            input.onclick = (inputEvent) => inputEvent.stopPropagation();
            input.ondblclick = (inputEvent) => inputEvent.stopPropagation();
            input.onkeydown = (keyEvent) => {
                keyEvent.stopPropagation();
                if (keyEvent.key === "Enter") input.blur();
                if (keyEvent.key === "Escape") {
                    cancelled = true;
                    input.blur();
                }
            };
            input.onblur = finish;
        };
        head.appendChild(title);
        if (mixed && !externalLocked) {
            const typeSel = document.createElement("select");
            typeSel.className = "bd-select bd-batch-type";
            typeSel.title = t("tooltip.mixedGroupType");
            for (const optKey of MIXED_GROUP_TASKS) {
                const o = document.createElement("option");
                o.value = optKey;
                o.textContent = `${optKey.toUpperCase()} · ${t(`task.${optKey}`)}`;
                if (optKey === cardKey) o.selected = true;
                typeSel.appendChild(o);
            }
            typeSel.onchange = (e) => {
                e.stopPropagation();
                applyMixedGroupType(editor, index, typeSel.value);
            };
            typeSel.onclick = (e) => e.stopPropagation();
            head.appendChild(typeSel);
        }
        // Per-segment continuity (master「段间引导」must be on; skip segment 1).
        const masterCont = isContinuityMasterEnabled(editor.timeline?.output);
        let continuityLabel = null;
        if (masterCont && index > 0 && isVideo) {
            const contLabel = document.createElement("label");
            contLabel.className = "bd-batch-continuity";
            contLabel.title = t("tooltip.segmentContinuityFromPrev");
            const contCb = document.createElement("input");
            contCb.type = "checkbox";
            contCb.className = "bd-batch-continuity-check";
            contCb.checked = isSegmentContinuityFromPrev(seg, index);
            contCb.onchange = (e) => {
                e.stopPropagation();
                seg.continuityFromPrev = !!contCb.checked;
                if (!contCb.checked) {
                    seg.continuityForcePrevCache = false;
                    forceCb.checked = false;
                }
                forceCb.disabled = !contCb.checked;
                // Flush timeline_data immediately so Queue Prompt cannot race the debounce.
                editor.commit?.(false, { syncTimeline: true });
                editor.flushTimelineSync?.();
            };
            contCb.onclick = (e) => e.stopPropagation();
            const contText = document.createElement("span");
            contText.setAttribute("data-i18n", "batch.continuityFromPrev");
            contText.textContent = t("batch.continuityFromPrev");
            contLabel.appendChild(contCb);
            contLabel.appendChild(contText);
            const forceLabel = document.createElement("label");
            forceLabel.className = "bd-batch-continuity";
            forceLabel.title = t("tooltip.segmentContinuityForcePrevCache");
            const forceCb = document.createElement("input");
            forceCb.type = "checkbox";
            forceCb.className = "bd-batch-continuity-force-check";
            forceCb.checked = contCb.checked && isSegmentContinuityForcePrevCache(seg, index);
            forceCb.disabled = !contCb.checked;
            forceCb.onchange = (e) => {
                e.stopPropagation();
                seg.continuityForcePrevCache = !!forceCb.checked;
                editor.commit?.(false, { syncTimeline: true });
                editor.flushTimelineSync?.();
            };
            forceCb.onclick = (e) => e.stopPropagation();
            const forceText = document.createElement("span");
            forceText.setAttribute("data-i18n", "batch.continuityForcePrevCache");
            forceText.textContent = t("batch.continuityForcePrevCache");
            forceLabel.appendChild(forceCb);
            forceLabel.appendChild(forceText);
            continuityLabel = [contLabel, forceLabel];
        }
        if (continuityLabel) head.append(...continuityLabel);
        const meta = document.createElement("div");
        meta.className = "bd-batch-head-meta";
        if (!externalLocked) {
            const seedRow = document.createElement("label");
            seedRow.className = "bd-batch-seed";
            seedRow.title = t("tooltip.batchSeed");
            const seedLabel = document.createElement("span");
            seedLabel.textContent = t("batch.seed");
            const seedMode = document.createElement("select");
            seedMode.title = t("tooltip.batchSeedMode");
            for (const mode of ["inherit", "random", "fixed"]) {
                const option = document.createElement("option");
                option.value = mode;
                option.textContent = t(`batch.seedMode.${mode}`);
                option.selected = mode === normalizeSegmentSeedMode(seg.seedMode);
                seedMode.appendChild(option);
            }
            const seedInput = document.createElement("input");
            seedInput.type = "text";
            seedInput.inputMode = "numeric";
            seedInput.pattern = "[0-9]*";
            seedInput.value = normalizeSegmentSeed(seg.seed);
            seedInput.title = t("tooltip.batchFixedSeed");
            seedInput.hidden = seedMode.value !== "fixed";
            const commitSeed = () => {
                seg.seedMode = normalizeSegmentSeedMode(seedMode.value);
                seg.seed = normalizeSegmentSeed(seedInput.value);
                seedInput.value = seg.seed;
                seedInput.hidden = seg.seedMode !== "fixed";
                editor.commit?.(false, { syncTimeline: true });
                editor.flushTimelineSync?.();
                scheduleDirectorPassCacheRefresh(editor, 100, "selected");
            };
            seedMode.onchange = (event) => {
                event.stopPropagation();
                commitSeed();
            };
            seedInput.onchange = commitSeed;
            seedInput.onblur = commitSeed;
            seedMode.onclick = (event) => event.stopPropagation();
            seedInput.onclick = (event) => event.stopPropagation();
            seedRow.append(seedLabel, seedMode, seedInput);
            meta.appendChild(seedRow);
        }
        if (isVideo) {
            const secRow = document.createElement("label");
            secRow.className = "bd-batch-fc";
            const { frames, durationSec: syncedSec, sourceFrames } = resolveVideoSegmentDuration(
                key,
                seg,
                resolveSegmentDurationSec(seg),
                { enforceSourceFrames: false },
            );
            const playSec = framesToDurationSec(frames, 24);
            seg.durationSec = syncedSec;
            seg.frameCount = frames;
            seg.length = frames;
            seg._videoFrameCount = frames;
            const hasHeldImage = seg.sourceVideo?.mediaKind === "image"
                && !!seg.sourceVideo?.image?.imageFile;
            const hasSourceMedia = sourceFrames > 0 || hasHeldImage;
            const displayedSec = isVideoEdit && !hasSourceMedia ? 0 : seg.durationSec;
            const minSec = minDurationSec();
            const durationTitle = t(
                hasHeldImage
                    ? "batch.heldImageDurationTooltip"
                    : (isVideoEdit ? "batch.videoEditDurationTooltip" : "batch.durationTooltip"),
                { frames, play: playSec },
            );
            secRow.append(document.createTextNode(`${t("batch.seconds")} `));
            const secInput = document.createElement("input");
            secInput.type = "number";
            secInput.dataset.batchSecIndex = String(index);
            secInput.dataset.batchSegId = String(seg.id || "");
            secInput.min = String(minSec);
            secInput.max = String(maxDurationSec());
            secInput.step = "0.1";
            secInput.value = String(displayedSec);
            secInput.title = durationTitle;
            secRow.appendChild(secInput);
            // Do not rewrite value/title while focused: frame snapping would
            // bounce 20.7↔20.5 and interrupt typing. Normalize on blur.
            let secFocused = false;
            secInput.addEventListener("focus", () => { secFocused = true; });
            const applySec = () => {
                if (isVideoEdit) {
                    const requested = durationToClampedMiniMaxFrames(Number(secInput.value), 24);
                    resizeSourceRangeToFrameCount(
                        editor.timeline.segments?.[index]?.sourceVideo,
                        requested.frames,
                    );
                }
                const updated = applyBatchSegmentDuration(editor, index, secInput.value);
                if (!updated) return;
                if (!secFocused) {
                    const play = framesToDurationSec(updated.frameCount, 24);
                    secInput.value = String(updated.durationSec);
                    secInput.title = t("batch.durationTooltip", {
                        frames: updated.frameCount,
                        play,
                    });
                }
                editor.scheduleTimelineSync();
                editor.scheduleRender?.();
                if (!secFocused) editor.renderImageBatchGroups?.();
                editor.updateVideoNameLabel?.();
                editor.updateOutputPreview?.();
                // Keep total_frames widget in sync with sum of group frames.
                if (editor.totalFramesWidget) {
                    editor.totalFramesWidget.value = sumFrameCounts(editor.timeline.segments);
                }
            };
            if (externalLocked) {
                secInput.readOnly = true;
                secInput.disabled = true;
                secInput.title = t("external.durationLocked");
            } else {
                secInput.onchange = applySec;
                secInput.oninput = () => {
                    clearTimeout(secInput._t);
                    secInput._t = setTimeout(applySec, 200);
                };
                secInput.onblur = () => {
                    clearTimeout(secInput._t);
                    secInput._t = null;
                    secFocused = false;
                    applySec();
                };
            }
            meta.appendChild(secRow);
            appendBatchPassControls(meta, editor, seg, index);
        }
        if (!externalLocked) {
            const del = document.createElement("button");
            del.type = "button";
            del.className = "bd-batch-del";
            del.textContent = t("batch.delete");
            del.disabled = editor.timeline.segments.length <= 1;
            del.onclick = (e) => {
                e.stopPropagation();
                const liveIdx = (editor.timeline.segments || []).findIndex((s) => s?.id && s.id === seg.id);
                deleteImageBatchGroup(editor, liveIdx >= 0 ? liveIdx : index);
            };
            meta.appendChild(del);
        }
        head.appendChild(meta);
        card.appendChild(head);

        if (isFl2v) {
            appendMixedFl2vSlots(card, editor, seg, index);
        } else if (isVideoEdit && !isRv2v) {
            const media = document.createElement("div");
            media.className = "bd-batch-media";
            mountSegSourceVideoTimeline(media, editor, seg, index);
            card.appendChild(media);
        } else if (variant === "source") {
            const media = document.createElement("div");
            media.className = "bd-batch-media";
            const src = document.createElement("div");
            src.className = "bd-batch-src";
            const file = seg.genImage?.imageFile || "";
            const label = t("panel.uploadSourceImage");
            renderSourceSlot(src, file);
            src.title = file
                ? t("source.imageTitleFilled", { label, file })
                : t("tooltip.uploadSourceImage");
            bindSlotActivate(src, {
                editor,
                hasMedia: !!file,
                onPick: () => uploadSegSource(editor, index),
                onPreview: () => openSlotPreview({
                    kind: "image",
                    src: viewUrl(file),
                    label,
                    onReplace: () => uploadSegSource(editor, index),
                }),
            });
            bindOsFileDrop(src, (files) => {
                const file = files.find(isBatchImageFile);
                if (file) void assignSegSourceFromFile(editor, index, file);
            });
            bindImageClipboardPaste(src, (file) => assignSegSourceFromFile(editor, index, file));
            if (file) {
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.title = t("common.delete");
                x.onclick = (e) => {
                    e.stopPropagation();
                    clearSegSourceImage(editor, index);
                };
                src.appendChild(x);
            }
            media.appendChild(src);
            card.appendChild(media);
        }
        let r2vMain = null;
        if (variant === "refs" && isR2v) {
            r2vMain = appendR2vMediaSections(card, seg, index, editor, {
                sourceVideo: isRv2v,
                referenceVideos: !isRv2v,
            });
        } else if (variant === "refs") {
            const media = document.createElement("div");
            media.className = "bd-batch-media";
            const refs = document.createElement("div");
            refs.className = "bd-batch-refs";
            for (let i = 0; i < MAX_REFERENCE_IMAGES; i++) {
                const ref = (seg.refs || []).find((r) => Number(r.index ?? r.slot) === i);
                const slot = document.createElement("div");
                slot.className = "bd-batch-ref";
                slot.dataset.refKind = "image";
                slot.dataset.refIndex = String(i);
                slot.dataset.refScope = "group";
                renderRefSlot(slot, ref, i, index, editor);
                bindBatchKindSlot(
                    slot,
                    editor,
                    index,
                    "image",
                    i,
                    refHasImage(ref),
                    (file) => void assignSegRefFromFile(editor, index, i, file),
                    () => uploadSegRef(editor, index, i),
                    slotPreviewSrc(ref, "image"),
                    refImageLabel(i),
                );
                refs.appendChild(slot);
            }
            media.appendChild(refs);
            card.appendChild(media);
        }

        const prompts = document.createElement("div");
        prompts.className = "bd-batch-prompts";
        const ph = t(isR2v ? "placeholder.batchR2v" : "placeholder.batchDefault");
        prompts.innerHTML = `
            <span class="bd-label">${t("batch.prompt")}</span>
            <textarea data-f="prompt" data-batch-prompt-index="${index}" data-batch-seg-id="${seg.id || ""}" placeholder=""></textarea>`;
        prompts.querySelector("textarea").placeholder = ph;
        prompts.querySelector("textarea").value = seg.prompt || "";
        const promptEl = prompts.querySelector('[data-f="prompt"]');
        const segIndex = index;
        const segId = seg.id;
        promptEl.oninput = (e) => {
            // Write by index/id — never capture a stale `seg` after normalize.
            const live = (editor.timeline.segments || []).find((s) => s?.id && s.id === segId)
                || editor.timeline.segments?.[segIndex];
            if (!live) return;
            live.prompt = e.target.value;
            live.negativePrompt = live.negativePrompt ?? "";
            editor.scheduleTimelineSync();
            // External groups execute from Group-node widgets — keep them aligned.
            editor.writeExternalGroupPrompt?.(segIndex, live.prompt);
            const card = promptEl.closest(".bd-batch-card");
            fillExecPrompt(card?.querySelector("[data-r=exec-prompt]"), editor, live);
        };
        if (isR2v || isVideoEdit) {
            wirePromptImageMentions(editor, promptEl, () => {
                const live = (editor.timeline.segments || []).find((s) => s?.id && s.id === segId)
                    || editor.timeline.segments?.[segIndex]
                    || seg;
                return {
                    refs: live.refs || [],
                    audios: live.refAudios || [],
                    videos: promptVideosFor(editor, live, live.refVideos || []),
                };
            });
        }

        const preview = document.createElement("div");
        preview.className = "bd-batch-preview";
        renderPreview(preview, seg, index === runningIdx, isVideo, seg.previewFps || fps);
        const previewCol = wrapPreviewColumn(preview, editor, seg);

        if (isR2v && r2vMain) {
            r2vMain.appendChild(prompts);
            r2vMain.parentElement?.appendChild(previewCol);
        } else {
            card.appendChild(prompts);
            card.appendChild(previewCol);
        }

        list.appendChild(card);
}

export function setImageBatchPreview(editor, segmentIndex, imageB64, extra = {}) {
    const seg = editor.timeline.segments[segmentIndex];
    if (!seg) return;
    const incomingPass = extra.pass || "first";
    if (seg.previewPass && seg.previewPass !== incomingPass) {
        // Never let the first-pass animation survive into second-pass sampling.
        seg.previewFrames = [];
    }
    seg.previewB64 = imageB64 || "";
    if (extra.step != null) seg.previewStep = extra.step;
    if (extra.total_steps != null) seg.previewTotalSteps = extra.total_steps;
    seg.previewPass = incomingPass;
    if (Array.isArray(extra.frames) && extra.frames.length) {
        seg.previewFrames = extra.frames;
        seg.previewFps = extra.fps || seg.previewFps || 16;
        seg.previewLive = !!extra.live;
    } else if (imageB64) {
        if (extra.live) {
            // Keep final multi-frame playback until a real final payload arrives.
            if (!Array.isArray(seg.previewFrames) || seg.previewFrames.length <= 1) {
                seg.previewFrames = [imageB64];
            }
            seg.previewLive = true;
        } else {
            seg.previewFrames = [imageB64];
            seg.previewLive = false;
        }
    }

    // Live sampling updates: patch the card preview in-place (avoid full re-render thrash).
    if (extra.live && imageB64) {
        const card = batchCardEl(editor, segmentIndex);
        const preview = card?.querySelector?.(".bd-batch-preview");
        if (preview) {
            const label = preview.parentElement?.querySelector?.(".bd-label");
            if (label) {
                label.textContent = samplingPreviewLabel(seg);
            }
            const tab = preview.parentElement?.querySelector?.('[data-r="preview-tab-sample"]');
            if (tab) {
                tab.textContent = samplingPreviewLabel(seg);
                tab.setAttribute("data-i18n", seg.previewPass === "second" ? "liveSample.secondPass" : "liveSample.firstPass");
            }
            const step = seg.previewStep;
            const total = seg.previewTotalSteps;
            const badgeText = (step && total)
                ? t("batch.generatingStep", { step, total })
                : t("batch.generating");
            let img = preview.querySelector("img.bd-live-preview");
            let badge = preview.querySelector(".bd-batch-live-badge");
            if (!img) {
                mountLivePreview(preview, seg, badgeText);
            } else {
                playLiveImgFrames(
                    img,
                    (extra.frames?.length ? extra.frames : [imageB64]),
                    extra.fps || seg.previewFps || 16,
                );
                if (badge) badge.textContent = badgeText;
            }
            const pickThumb = editor.batchPicker?.querySelector?.(`.bd-batch-pick[data-batch-index="${segmentIndex}"] img.bd-batch-pick-thumb`);
            if (pickThumb) pickThumb.src = frameSrc(imageB64);
            return;
        }
        const pick = editor.batchPicker?.querySelector?.(`.bd-batch-pick[data-batch-index="${segmentIndex}"]`);
        if (pick) {
            pick.classList.add("running");
            let img = pick.querySelector("img.bd-batch-pick-thumb");
            if (!img) {
                img = document.createElement("img");
                img.className = "bd-batch-pick-thumb";
                img.alt = "";
                pick.appendChild(img);
            }
            img.src = frameSrc(imageB64);
        }
        return;
    }
    editor.renderImageBatchGroups();
}

export function bindImageBatchEvents(editor) {
    editor.batchAddBtn?.addEventListener("click", (e) => {
        e.stopPropagation();
        addImageBatchGroup(editor);
    });
}

/** Kept for compatibility; the batch list now grows with all visible groups. */
export const BATCH_LIST_MAX_H = Number.POSITIVE_INFINITY;
const BATCH_LIST_MIN_H = 160;
const BATCH_LIST_GAP = 8;
const BATCH_TOOLBAR_H = 48;
const BATCH_PANEL_CHROME = 28;
const BATCH_COLLAPSED_ROW_H = 34;

function elementOuterHeight(el) {
    if (!el) return 0;
    const h = Number(el.offsetHeight) || 0;
    try {
        const style = getComputedStyle(el);
        return h + (Number.parseFloat(style.marginTop) || 0) + (Number.parseFloat(style.marginBottom) || 0);
    } catch {
        return h;
    }
}

export function getImageBatchUiHeight(editor) {
    const solo = isBatchDetailSolo(editor);
    const segments = editor?.timeline?.segments || [];
    const visibleSegments = solo
        ? [segments[Math.max(0, Math.min(segments.length - 1, editor?.selectedIndex || 0))]].filter(Boolean)
        : segments;
    const key = resolveTaskKey(editor?.getTaskKey?.() || editor?.taskTypeWidget?.value);
    const defaultRowHeight = (segment) => {
        const segmentKey = key === "mixed" ? resolveMixedGroupKey(segment) : key;
        return segmentKey === "r2v" || segmentKey === "rv2v"
            ? 490
            : (segmentKey === "v2v" ? 360 : 195);
    };
    const showPicker = solo && segments.length > 1 && !editor?.usesBatchTimeline?.();
    const pickerH = showPicker ? 56 : 0;
    const renderedCards = [...(editor?.batchList?.querySelectorAll?.(":scope > .bd-batch-card") || [])];
    const measuredRowsH = renderedCards.length === visibleSegments.length && renderedCards.length > 0
        ? Math.ceil(editor.batchList.scrollHeight)
        : 0;
    const estimatedRowsH = visibleSegments.length
        ? visibleSegments.reduce((sum, segment) => {
            if (segment.uiCollapsed) return sum + BATCH_COLLAPSED_ROW_H;
            const savedHeight = Number(segment.uiCardHeight);
            return sum + (Number.isFinite(savedHeight) && savedHeight > 0
                ? savedHeight
                : defaultRowHeight(segment));
        }, 0)
        : defaultRowHeight(null);
    const listRowsH = measuredRowsH || (
        estimatedRowsH + Math.max(0, visibleSegments.length - 1) * BATCH_LIST_GAP
    );
    const panel = editor?.batchPanel;
    if (panel) {
        const children = [...panel.children].filter((child) => (
            !child.classList?.contains("hidden") && getComputedStyle(child).display !== "none"
        ));
        if (children.length) {
            const childrenHeight = children.reduce((sum, child) => (
                sum + (child === editor.batchList ? listRowsH : elementOuterHeight(child))
            ), 0);
            return Math.ceil(childrenHeight + Math.max(0, children.length - 1) * BATCH_LIST_GAP);
        }
    }
    const listH = Math.min(listRowsH + pickerH, BATCH_LIST_MAX_H);
    return BATCH_TOOLBAR_H + BATCH_PANEL_CHROME + listH;
}

function _clearBatchListFillStyles(list, host, wrap, panel) {
    if (list) {
        list.style.height = "";
        list.style.maxHeight = "";
        list.style.minHeight = "";
        list.style.flex = "";
        list.classList?.remove("bd-batch-solo");
        for (const card of list.querySelectorAll?.(".bd-batch-card") || []) {
            card.style.flex = "";
            card.style.minHeight = "";
            card.style.height = "";
        }
    }
    if (panel) {
        panel.style.flex = "";
        panel.style.minHeight = "";
        panel.style.height = "";
        panel.style.maxHeight = "";
        panel.style.overflow = "";
    }
    if (wrap) {
        wrap.style.height = "";
        wrap.style.minHeight = "";
        wrap.style.maxHeight = "";
        wrap.style.overflow = "";
    }
    if (host) {
        host.style.height = "";
        host.style.maxHeight = "";
        host.style.overflow = "";
    }
    const main = host?.querySelector?.(".bd-main");
    if (main) {
        main.style.height = "";
        main.style.minHeight = "";
        main.style.maxHeight = "";
        main.style.overflow = "";
    }
}

/** Soft cap above content min — blocks Vue ResizeObserver / stretch feedback runaway. */
export const DIRECTOR_UI_MAX_EXTRA_H = 1200;

function _mmxHeightDebug(...args) {
    try {
        if (typeof localStorage !== "undefined" && localStorage.getItem("mmxHeightDebug") === "1") {
            console.debug("[mmx-height]", ...args);
        }
    } catch {
        /* ignore */
    }
}

/**
 * Trusted pixel height LiteGraph/ComfyUI allocates to the DOM widget.
 * Never derive from node.size (other-widgets underestimate → write-back loop) or from
 * host.clientHeight after we stamped style.height (self-referential ratchet).
 *
 * Do NOT clamp to DIRECTOR_UI_MAX_EXTRA_H here — that cap is only for heal/runaway.
 * Clamping fill caused "drag taller → content stops → huge blank below".
 */
function measureWidgetSlotHeight(editor) {
    const widget = editor?.domWidget;
    const minH = contentDomWidgetMinHeight(editor);
    const computed = Number(widget?.computedHeight);
    if (Number.isFinite(computed) && computed > 0) {
        return computed;
    }
    // Only read element box when we have not forced an inline height (avoids self-inflate).
    const el = widget?.element;
    if (el && !el.style.height) {
        const elH = Number(el.clientHeight || el.offsetHeight);
        if (Number.isFinite(elH) && elH > 0) return Math.max(elH, minH);
    }
    const host = editor?.container;
    const parent = host?.parentElement;
    if (parent && host && !host.style.height) {
        const pH = Number(parent.clientHeight);
        if (Number.isFinite(pH) && pH > minH) return pH;
    }
    // No trusted slot → content min only. Callers must not invent height from node.size.
    return minH;
}

/** True when measure came from LiteGraph computedHeight (safe to allocate px inside). */
function hasTrustedComputedSlot(editor) {
    const computed = Number(editor?.domWidget?.computedHeight);
    return Number.isFinite(computed) && computed > 0;
}

export function contentDomWidgetMinHeight(editor) {
    const minH = typeof editor?.getDirectorUiMinHeight === "function"
        ? editor.getDirectorUiMinHeight()
        : 0;
    return Math.max(0, minH || 0);
}

/**
 * Make the Director DOM widget *growable* in LiteGraph._arrangeWidgets.
 *
 * IMPORTANT: never assign widget.computeSize. If computeSize exists, LiteGraph
 * treats the widget as fixed-height and never distributes free space when the
 * user drags the node taller — 素材组 stays short with a void below.
 * Use computeLayoutSize + getMinHeight/getMaxHeight only.
 */
export function bindDomWidgetContentComputeSize(editor) {
    const widget = editor?.domWidget;
    const minH = contentDomWidgetMinHeight(editor);
    if (!widget) return minH;
    // Remove any fixed-size hook (ours or leftover) so free space can flow in.
    try {
        delete widget.computeSize;
    } catch {
        widget.computeSize = undefined;
    }
    // No maxHeight → LiteGraph gives this widget all leftover node height on drag.
    widget.computeLayoutSize = () => ({
        minHeight: minH,
        maxHeight: undefined,
        minWidth: 0,
    });
    if (widget.options) {
        widget.options.getMinHeight = () => contentDomWidgetMinHeight(editor);
        delete widget.options.getMaxHeight;
        delete widget.options.getHeight;
    }
    return minH;
}

/**
 * Grow the 素材组 list into leftover node height when the user drags the Director taller.
 * Uses the widget slot height for *layout only* — never writes it into getMinHeight
 * (that was the infinite-growth bug).
 *
 * @param {{ settle?: boolean }} [opts] settle=false skips rAF (progress path); default one rAF.
 */
export function syncBatchPanelFillHeight(editor, opts = {}) {
    const list = editor?.batchList;
    const wrap = editor?.root;
    const host = editor?.container;
    const panel = editor?.batchPanel;
    const main = editor?.mainBody;
    if (!list || !wrap || !host) return;

    const batchOn = !!editor.isImageBatch?.()
        && !panel?.classList?.contains("hidden");
    wrap.classList.toggle("bd-batch-fill", batchOn);
    const minH = contentDomWidgetMinHeight(editor);
    bindDomWidgetContentComputeSize(editor);

    host.style.minHeight = `${minH || 0}px`;
    host.style.setProperty("--comfy-widget-min-height", `${minH || 0}px`);

    if (!batchOn) {
        _clearBatchListFillStyles(list, host, wrap, panel);
        return;
    }

    // A restored snapshot rebuilds cards before LiteGraph arranges the widget.
    // Clear the previous project's 0px measurements instead of preserving them.
    if (opts.restore) {
        main?.style.removeProperty("height");
        main?.style.removeProperty("max-height");
        panel?.style.removeProperty("height");
        panel?.style.removeProperty("max-height");
        list.style.removeProperty("height");
        list.style.removeProperty("max-height");
    }

    const applyFill = () => {
        if (!editor.batchList || editor.batchPanel?.classList?.contains("hidden")) return;

        // LiteGraph DOM widgets keep a vertical margin inside computedHeight; if we
        // size content to the full slot, the run-status bar paints past the node edge.
        const widget = editor.domWidget;
        const margin = Number(widget?.margin ?? widget?.options?.margin ?? 10);
        const inset = Math.max(12, margin * 2 + 4);
        const trusted = hasTrustedComputedSlot(editor);
        const rawSlot = measureWidgetSlotHeight(editor);
        // Full LiteGraph slot (minus widget margin). Never EXTRA-cap — user drag must fill.
        const slotH = Math.max(0, rawSlot - inset);

        // Fill the allocated widget box. Prefer % so we never paint shorter than parent
        // (pixel maxHeight < computed was the "blank below 素材组" bug after EXTRA clamp).
        host.style.height = "";
        wrap.style.height = "";
        wrap.style.minHeight = "0";
        host.style.maxHeight = "100%";
        wrap.style.maxHeight = "100%";
        host.style.overflow = "hidden";
        wrap.style.overflow = "hidden";

        const status = wrap.querySelector(".bd-run-status");
        const live = wrap.querySelector(":scope > .bd-live-sample");
        const statusH = status ? (status.offsetHeight + 6) : 0;
        const liveHidden = !live || live.classList.contains("hidden")
            || getComputedStyle(live).display === "none";
        const liveH = liveHidden ? 0 : (live.offsetHeight + 6);
        let topChrome = 0;
        for (const child of wrap.children) {
            if (child === main || child === status || child === live) continue;
            if (child.tagName === "STYLE") continue;
            if (child.classList?.contains("hidden")) continue;
            if (getComputedStyle(child).display === "none") continue;
            topChrome += elementOuterHeight(child) + 6;
        }

        // During restore LiteGraph can briefly expose the old widget height.
        // Do not turn that transient measurement into a permanent 0px clamp.
        // The next settle pass will run after the DOM widget is arranged.
        const hasUsableBudget = trusted && slotH > 0
            ? slotH >= Math.max(minH, 120)
            : false;
        const budget = hasUsableBudget
            ? slotH
            : Math.max(minH, Number(wrap.clientHeight || host.clientHeight) || minH);
        const mainH = Math.max(0, budget - statusH - liveH - topChrome);

        if (main) {
            main.style.flex = "1 1 0";
            main.style.minHeight = "0";
            main.style.overflow = "hidden";
            if (hasUsableBudget) {
                main.style.height = `${mainH}px`;
                main.style.maxHeight = `${mainH}px`;
            } else {
                main.style.height = "";
                main.style.maxHeight = "";
            }
        }

        let used = 0;
        let visible = 0;
        if (main) {
            for (const child of main.children) {
                if (child === panel) continue;
                if (child.tagName === "STYLE") continue;
                if (child.classList?.contains("hidden")) continue;
                if (getComputedStyle(child).display === "none") continue;
                used += elementOuterHeight(child);
                visible += 1;
            }
        }
        const gaps = 6 * Math.max(0, visible);
        const batchH = Math.max(0, Math.floor(mainH - used - gaps));
        panel.style.flex = "1 1 0";
        panel.style.minHeight = "0";
        panel.style.overflow = "hidden";
        if (hasUsableBudget) {
            panel.style.height = `${batchH}px`;
            panel.style.maxHeight = `${batchH}px`;
        } else {
            panel.style.height = "";
            panel.style.maxHeight = "";
        }

        const batchToolbar = panel.querySelector?.(".bd-batch-toolbar");
        const picker = editor.batchPicker;
        const pickerH = picker?.classList?.contains("visible") ? (picker.offsetHeight + 8) : 0;
        const listH = Math.max(
            0,
            batchH - (batchToolbar?.offsetHeight || 0) - pickerH - 10,
        );
        const solo = (editor.timeline?.segments?.length || 0) <= 1 || isBatchDetailSolo(editor);
        list.style.flex = "0 0 auto";
        list.style.minHeight = "0";
        list.style.overflowY = "visible";
        list.style.height = "";
        list.style.maxHeight = "";

        list.classList.toggle("bd-batch-solo", solo);
        for (const card of list.querySelectorAll(".bd-batch-card")) {
            const index = Number.parseInt(card.dataset.batchIndex, 10);
            const savedHeight = Number(editor.timeline?.segments?.[index]?.uiCardHeight);
            if (card.dataset.userResizing === "1") {
                continue;
            } else if (Number.isFinite(savedHeight) && savedHeight > 0) {
                card.style.flex = "0 0 auto";
                card.style.minHeight = "";
                card.style.height = `${Math.round(savedHeight)}px`;
            } else {
                card.style.flex = "";
                card.style.minHeight = "";
                card.style.height = "";
            }
        }

        _mmxHeightDebug({
            task: editor.getTaskKey?.(),
            trusted,
            minH,
            rawSlot,
            slotH,
            nodeH: editor.node?.size?.[1],
            computed: editor.domWidget?.computedHeight,
        });
    };

    applyFill();
    // One settle frame after layout (resize / mode switch). Never triple-rAF on progress.
    if (opts.settle !== false) {
        requestAnimationFrame(applyFill);
    }
}

export function setToolbarDisabledForBatch(editor, disabled) {
    const btns = [
        editor.btnVideo,
        editor.btnVideoAppend,
        editor.root?.querySelector('[data-a="del"]'),
        editor.root?.querySelector('[data-a="mode-global"]'),
        editor.root?.querySelector('[data-a="mode-segment"]'),
    ];
    for (const btn of btns) {
        if (!btn) continue;
        // Batch / t2v / i2v: fully hide video-editing controls (not just disable).
        btn.classList.toggle("hidden", disabled);
        btn.disabled = disabled;
        btn.classList.toggle("bd-disabled", disabled);
    }
    editor.root?.querySelector(".bd-mode")?.classList.toggle("hidden", disabled);
}

/** r2v: fl2v-like toolbar — timeline visible; add group sits left of task select. */
export function setR2vToolbar(editor, enabled) {
    const hide = [
        editor.btnVideo,
        editor.btnVideoAppend,
        editor.root?.querySelector('[data-a="mode-global"]'),
        editor.root?.querySelector('[data-a="mode-segment"]'),
    ];
    for (const btn of hide) {
        if (!btn) continue;
        btn.classList.toggle("hidden", enabled);
        btn.disabled = enabled;
        btn.classList.toggle("bd-disabled", enabled);
    }
    editor.root?.querySelector(".bd-mode")?.classList.toggle("hidden", enabled);

    const externalLocked = !!(editor.hasExternalI2vGroups?.() || editor.hasExternalR2vGroups?.());
    const del = editor.root?.querySelector('[data-a="del"]');
    if (del) {
        if (externalLocked) {
            del.classList.add("hidden");
            del.disabled = true;
        } else {
            const deleteDisabled = enabled && (editor.timeline?.segments?.length || 0) <= 1;
            del.disabled = deleteDisabled;
            del.classList.toggle("bd-disabled", deleteDisabled);
            del.classList.remove("hidden");
            del.textContent = enabled ? t("toolbar.deleteSelectedGroup") : t("toolbar.deleteSegment");
            del.setAttribute("data-i18n", enabled ? "toolbar.deleteSelectedGroup" : "toolbar.deleteSegment");
            del.setAttribute("data-i18n-title", enabled ? "tooltip.deleteSelectedR2vGroup" : "tooltip.deleteSegment");
            del.title = enabled
                ? t("tooltip.deleteSelectedR2vGroup")
                : t("tooltip.deleteSegment");
        }
    }
    const addBtn = editor.root?.querySelector('[data-a="r2v-add-group"]');
    if (addBtn) {
        addBtn.classList.toggle("hidden", !enabled || externalLocked);
        addBtn.disabled = !enabled || externalLocked;
        if (enabled) {
            const addKey = "toolbar.addShot";
            addBtn.textContent = t(addKey);
            addBtn.setAttribute("data-i18n", addKey);
            addBtn.setAttribute("data-i18n-title", "tooltip.addShot");
            addBtn.title = t("tooltip.addShot");
        }
    }
    const batchAdd = editor.batchPanel?.querySelector('[data-a="batch-add"]');
    if (batchAdd) batchAdd.classList.toggle("hidden", enabled || externalLocked);
    updateR2vToolbarBtns(editor);
}

export function updateR2vToolbarBtns(editor) {
    const addBtn = editor?.root?.querySelector?.('[data-a="r2v-add-group"]');
    const externalLocked = !!(editor?.hasExternalI2vGroups?.() || editor?.hasExternalR2vGroups?.());
    const isVideoBatch = !!editor?.isImageBatch?.()
        && isVideoBatchTask(editor?.getTaskKey?.());
    const show = isVideoBatch && !externalLocked;
    if (addBtn) {
        addBtn.classList.toggle("hidden", !show);
        addBtn.disabled = !show;
        if (show) {
            const addKey = "toolbar.addShot";
            addBtn.textContent = t(addKey);
            addBtn.setAttribute("data-i18n", addKey);
            addBtn.setAttribute("data-i18n-title", "tooltip.addShot");
            addBtn.title = t("tooltip.addShot");
        }
    }
    if (!isVideoBatch) return;

    const del = editor?.root?.querySelector?.('[data-a="del"]');
    if (!del) return;
    const canDelete = !externalLocked && (editor.timeline?.segments?.length || 0) > 1;
    del.classList.toggle("hidden", externalLocked);
    del.disabled = !canDelete;
    del.classList.toggle("bd-disabled", !canDelete);
    del.textContent = t("toolbar.deleteSelectedGroup");
    del.setAttribute("data-i18n", "toolbar.deleteSelectedGroup");
    del.setAttribute("data-i18n-title", "tooltip.deleteSelectedPromptGroup");
    del.title = t("tooltip.deleteSelectedPromptGroup");
}
