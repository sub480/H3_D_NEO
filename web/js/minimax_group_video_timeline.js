import { api } from "../../scripts/api.js";
import { floorMiniMaxFrameCount } from "./minimax_gen_timeline.js";
import { t } from "./minimax_i18n.js";

const MIN_SEGMENT_FRAMES = 5;

const STYLES = `
.bd-group-video-timeline{display:flex;flex-direction:column;gap:7px;min-width:0;background:#0b0d0e;border:1px solid #293033;border-radius:8px;padding:8px;box-sizing:border-box}
.bd-group-video-player{position:relative;width:100%;height:clamp(120px,22vw,260px);display:flex;align-items:center;justify-content:center;background:#050606;border-radius:6px;overflow:hidden;cursor:pointer}
.bd-group-video-player video{width:100%;height:100%;object-fit:contain;background:#050606}
.bd-group-video-seek{width:100%;height:14px;margin:0;accent-color:#69d99a;cursor:pointer}
.bd-group-video-seek:disabled{opacity:.4;cursor:not-allowed}
    .bd-group-video-controls{display:flex;align-items:center;gap:5px}
    .bd-group-video-controls button{width:25px;height:23px;padding:0;border:1px solid #364044;border-radius:4px;background:#15191b;color:#d6dcdf;cursor:pointer}
    .bd-group-video-sync-material{margin-left:auto}
.bd-group-video-time{font-size:10px;color:#899397;white-space:nowrap;font-variant-numeric:tabular-nums}
.bd-group-video-track-viewport{width:100%;overflow-x:auto;overflow-y:hidden;border-radius:5px}
.bd-group-video-track{position:relative;height:42px;min-width:100%;border:1px solid #354045;border-radius:5px;background:#0f1315;overflow:hidden;cursor:grab;box-sizing:border-box;touch-action:none;user-select:none}
.bd-group-video-track:hover{border-color:#59686e}
.bd-group-video-track.dragging{cursor:grabbing;border-color:#69d99a}
.bd-group-video-track-fill{position:absolute;top:5px;bottom:5px;background:#293b40;border-right:1px solid rgba(255,255,255,.18)}
.bd-group-video-track-fill:nth-child(even){background:#314147}
.bd-group-video-tick{position:absolute;top:5px;bottom:5px;width:1px;background:rgba(255,255,255,.15)}
.bd-group-video-selection{position:absolute;top:0;bottom:0;background:rgba(105,217,154,.16);border-top:1px solid rgba(105,217,154,.75);border-bottom:1px solid rgba(105,217,154,.75);box-sizing:border-box;cursor:grab;touch-action:none;z-index:2}
.bd-group-video-selection.dragging{cursor:grabbing;background:rgba(105,217,154,.24)}
.bd-group-video-range-handle{position:absolute;top:0;bottom:0;width:10px;background:#69d99a;border:2px solid #13251c;border-radius:3px;box-shadow:0 1px 4px rgba(0,0,0,.6);transform:translateX(-50%);cursor:ew-resize;touch-action:none;z-index:3}
.bd-group-video-playhead{position:absolute;top:0;bottom:0;width:2px;background:#69d99a;box-shadow:0 0 0 1px rgba(0,0,0,.45);pointer-events:none}
.bd-group-video-playhead::after{content:"";position:absolute;left:50%;top:50%;width:12px;height:12px;border-radius:50%;background:#69d99a;border:2px solid #13251c;box-shadow:0 1px 4px rgba(0,0,0,.55);transform:translate(-50%,-50%)}
.bd-group-video-message{font-size:10px;color:#9aa4a8;min-height:14px;line-height:1.4}
.bd-group-video-message.error{color:#e58b8b}
`;

let stylesInjected = false;
const timelineViewStates = new WeakMap();

function timelineViewState(editor, seg) {
    let states = timelineViewStates.get(editor);
    if (!states) {
        states = new Map();
        timelineViewStates.set(editor, states);
    }
    const key = seg?.id ?? seg;
    if (!states.has(key)) states.set(key, { zoom: 1, scrollLeft: 0 });
    return states.get(key);
}

function injectStyles() {
    if (stylesInjected) return;
    stylesInjected = true;
    const style = document.createElement("style");
    style.textContent = STYLES;
    document.head.appendChild(style);
}

function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
}

function alignedRangeEnd(start, end) {
    const alignedFrames = floorMiniMaxFrameCount(end - start);
    return alignedFrames > 0 ? start + alignedFrames : end;
}

function warnRangeAlignment(start, originalEnd, alignedEnd) {
    if (alignedEnd === originalEnd) return;
    console.warn(
        `[MiniMax H3 Director] Source range aligned down: start=${start}, frames=${originalEnd - start} -> ${alignedEnd - start} `
        + `(cropped ${originalEnd - alignedEnd} tail frame(s)).`,
    );
}

function sourceModel(seg) {
    const source = seg?.sourceVideo || {};
    const video = source.video || source;
    const clips = Array.isArray(source.videoClips) && source.videoClips.length
        ? source.videoClips
        : ((video.videoFile || video.fileName) ? [video] : []);
    const frameMap = Array.isArray(video.frameMap) ? video.frameMap : [];
    const totalFrames = Number(
        frameMap.length
        || source.totalFrames
        || video.sourceFrameCount
        || clips.reduce((sum, clip) => sum + (Number(clip.sourceFrameCount) || 0), 0)
        || 0
    );
    return { source, video, clips, frameMap, totalFrames };
}

function normalizeMapEntry(entry) {
    if (entry && typeof entry === "object") {
        return {
            clip: Number(entry.clip ?? entry.videoClip ?? 0) || 0,
            frame: Number(entry.frame) || 0,
        };
    }
    return { clip: 0, frame: Number(entry) || 0 };
}

function logicalEntry(model, logicalFrame) {
    if (model.frameMap.length) return normalizeMapEntry(model.frameMap[logicalFrame]);
    let cursor = logicalFrame;
    for (let clipIndex = 0; clipIndex < model.clips.length; clipIndex++) {
        const count = Number(model.clips[clipIndex]?.sourceFrameCount) || 0;
        if (cursor < count || clipIndex === model.clips.length - 1) {
            return { clip: clipIndex, frame: Math.max(0, cursor) };
        }
        cursor -= count;
    }
    return { clip: 0, frame: Math.max(0, logicalFrame) };
}

function viewUrl(clip) {
    const raw = String(clip?.videoFile || clip?.fileName || "").replace(/\\/g, "/");
    if (!raw) return "";
    const slash = raw.lastIndexOf("/");
    const filename = slash >= 0 ? raw.slice(slash + 1) : raw;
    const subfolder = clip?.subfolder || (slash >= 0 ? raw.slice(0, slash) : "");
    const params = new URLSearchParams({ filename, type: clip?.type || "input" });
    if (subfolder) params.set("subfolder", subfolder);
    return api.apiURL(`/view?${params.toString()}`);
}

function clipLogicalRanges(model) {
    if (!model.clips.length) return [];
    if (!model.frameMap.length) {
        let cursor = 0;
        return model.clips.map((clip, clipIndex) => {
            const start = cursor;
            cursor += Number(clip.sourceFrameCount) || 0;
            return { clip, clipIndex, start, end: Math.min(cursor, model.totalFrames) };
        }).filter((range) => range.end > range.start);
    }
    const ranges = model.clips.map((clip, clipIndex) => ({
        clip,
        clipIndex,
        start: model.totalFrames,
        end: 0,
    }));
    model.frameMap.forEach((raw, logicalFrame) => {
        const entry = normalizeMapEntry(raw);
        const range = ranges[entry.clip];
        if (!range) return;
        range.start = Math.min(range.start, logicalFrame);
        range.end = Math.max(range.end, logicalFrame + 1);
    });
    return ranges.filter((range) => range.end > range.start);
}

function detectionRanges(model) {
    if (!model.frameMap.length) {
        return clipLogicalRanges(model).map((range) => ({
            ...range,
            sourceFrameStart: 0,
            sourceFrameEnd: range.end - range.start,
        }));
    }
    const ranges = [];
    let run = null;
    model.frameMap.forEach((raw, logicalFrame) => {
        const entry = normalizeMapEntry(raw);
        if (!run || run.clipIndex !== entry.clip || entry.frame !== run.sourceFrameEnd) {
            if (run) ranges.push(run);
            run = {
                clip: model.clips[entry.clip],
                clipIndex: entry.clip,
                start: logicalFrame,
                end: logicalFrame + 1,
                sourceFrameStart: entry.frame,
                sourceFrameEnd: entry.frame + 1,
            };
        } else {
            run.end = logicalFrame + 1;
            run.sourceFrameEnd = entry.frame + 1;
        }
    });
    if (run) ranges.push(run);
    return ranges.filter((range) => range.clip && range.end > range.start);
}

function button(label, title = label) {
    const element = document.createElement("button");
    element.type = "button";
    element.textContent = label;
    element.title = title;
    return element;
}

export function mountGroupVideoTimeline(container, options) {
    injectStyles();
    const { editor, seg, onUpload, onDropFile, onRangeChange, onRangePreview, onSyncMaterial, onSyncSeconds } = options;
    const model = sourceModel(seg);
    const logicalRanges = clipLogicalRanges(model);
    const videoLabel = options.sourceLabel || t("slot.video", { n: 1 });
    const firstFile = model.clips[0]?.videoFile || model.clips[0]?.fileName || "";
    const sourceTitle = firstFile
        ? t("ref.videoTitleFilled", { label: videoLabel, file: firstFile })
        : t("ref.videoTitleEmpty", { label: videoLabel });
    const root = document.createElement("div");
    root.className = "bd-group-video-timeline";
    root.dataset.refKind = "video";
    root.dataset.refIndex = "0";
    root.dataset.refScope = "group";
    root.title = sourceTitle;
    container.appendChild(root);

    const playerWrap = document.createElement("div");
    playerWrap.className = "bd-group-video-player";
    playerWrap.title = sourceTitle;
    const player = document.createElement("video");
    player.muted = false;
    player.playsInline = true;
    player.preload = "metadata";
    playerWrap.appendChild(player);
    playerWrap.tabIndex = 0;
    playerWrap.setAttribute("role", "button");
    playerWrap.onclick = () => onUpload?.();
    playerWrap.onkeydown = (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        onUpload?.();
    };
    playerWrap.ondragover = (event) => {
        if (![...(event.dataTransfer?.types || [])].includes("Files")) return;
        event.preventDefault();
        event.stopPropagation();
        event.dataTransfer.dropEffect = "copy";
    };
    playerWrap.ondrop = (event) => {
        const file = [...(event.dataTransfer?.files || [])][0];
        if (!file) return;
        event.preventDefault();
        event.stopPropagation();
        onDropFile?.(file);
    };
    root.appendChild(playerWrap);

    const seek = document.createElement("input");
    seek.className = "bd-group-video-seek";
    seek.type = "range";
    seek.min = "1";
    seek.max = String(Math.max(1, model.totalFrames));
    seek.step = "1";
    seek.value = "1";
    seek.disabled = !model.totalFrames;
    seek.title = t("player.frameJump");
    seek.setAttribute("aria-label", t("player.frameJump"));
    root.appendChild(seek);

    const controls = document.createElement("div");
    controls.className = "bd-group-video-controls";
    const play = button("▶", t("player.playPause"));
    const mute = button("🔊", t("player.mute"));
    const exportRange = button("⇩", t("player.exportRange"));
    const time = document.createElement("span");
    time.className = "bd-group-video-time";
    controls.append(play, mute, exportRange);
    if (onSyncSeconds) {
        const syncSeconds = button("⏱", t("player.syncSeconds"));
        syncSeconds.className = "bd-group-video-sync-seconds";
        syncSeconds.onclick = (event) => {
            event.preventDefault();
            event.stopPropagation();
            onSyncSeconds();
        };
        controls.appendChild(syncSeconds);
    }
    controls.appendChild(time);
    if (onSyncMaterial) {
        const syncMaterial = button("⟳", t("player.syncMaterial"));
        syncMaterial.className = "bd-group-video-sync-material";
        syncMaterial.onclick = (event) => {
            event.preventDefault();
            event.stopPropagation();
            onSyncMaterial();
        };
        controls.appendChild(syncMaterial);
    }
    root.appendChild(controls);

    const trackViewport = document.createElement("div");
    trackViewport.className = "bd-group-video-track-viewport";
    const track = document.createElement("div");
    track.className = "bd-group-video-track";
    for (const range of logicalRanges) {
        const fill = document.createElement("div");
        fill.className = "bd-group-video-track-fill";
        fill.style.left = `${model.totalFrames ? (range.start / model.totalFrames) * 100 : 0}%`;
        fill.style.width = `${model.totalFrames ? ((range.end - range.start) / model.totalFrames) * 100 : 0}%`;
        const clipFile = range.clip.videoFile || range.clip.fileName || "";
        fill.title = t("ref.videoTitleFilled", { label: videoLabel, file: clipFile || videoLabel });
        track.appendChild(fill);
    }
    for (let tickIndex = 1; tickIndex < 5; tickIndex++) {
        const tick = document.createElement("span");
        tick.className = "bd-group-video-tick";
        tick.style.left = `${tickIndex * 20}%`;
        track.appendChild(tick);
    }
    const selection = document.createElement("span");
    selection.className = "bd-group-video-selection";
    selection.title = t("player.rangeMove");
    const rangeStartHandle = document.createElement("span");
    rangeStartHandle.className = "bd-group-video-range-handle";
    rangeStartHandle.title = t("player.rangeStart");
    const rangeEndHandle = document.createElement("span");
    rangeEndHandle.className = "bd-group-video-range-handle";
    rangeEndHandle.title = t("player.rangeEnd");
    track.append(selection, rangeStartHandle, rangeEndHandle);
    const playhead = document.createElement("span");
    playhead.className = "bd-group-video-playhead";
    track.appendChild(playhead);
    trackViewport.appendChild(track);
    root.appendChild(trackViewport);

    const message = document.createElement("div");
    message.className = "bd-group-video-message";
    root.appendChild(message);

    let rangeStart = clamp(Math.round(Number(model.source.rangeStart) || 0), 0, Math.max(0, model.totalFrames - 1));
    let rangeEnd = clamp(Math.round(Number(model.source.rangeEnd) || model.totalFrames), rangeStart + 1, model.totalFrames);
    if (rangeEnd - rangeStart < Math.min(MIN_SEGMENT_FRAMES, model.totalFrames)) {
        rangeStart = 0;
        rangeEnd = model.totalFrames;
    }
    const initialRangeEnd = rangeEnd;
    rangeEnd = alignedRangeEnd(rangeStart, rangeEnd);
    let currentFrame = rangeStart;
    let activeClip = -1;
    let activeRange = -1;
    let raf = 0;
    let trackDragging = false;
    let resumeAfterTrackDrag = false;
    let rangeDragging = "";
    let rangeDragAnchor = 0;
    let rangeDragStart = 0;
    let rangeDragEnd = 0;
    let rangeDragCurrent = 0;
    const viewState = timelineViewState(editor, seg);
    let trackZoom = clamp(Number(viewState.zoom) || 1, 1, 10);
    track.style.width = `${trackZoom * 100}%`;
    requestAnimationFrame(() => {
        trackViewport.scrollLeft = Math.max(0, Number(viewState.scrollLeft) || 0);
    });
    trackViewport.addEventListener("scroll", () => {
        viewState.scrollLeft = trackViewport.scrollLeft;
    });

    trackViewport.addEventListener("wheel", (event) => {
        if (!event.altKey) return;
        const zoomDelta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX;
        if (zoomDelta === 0) return;
        event.preventDefault();
        event.stopPropagation();
        const viewportRect = trackViewport.getBoundingClientRect();
        const pointerX = event.clientX - viewportRect.left;
        const oldWidth = track.getBoundingClientRect().width || viewportRect.width;
        const anchorRatio = oldWidth > 0
            ? (trackViewport.scrollLeft + pointerX) / oldWidth
            : 0.5;
        trackZoom = clamp(trackZoom + (zoomDelta < 0 ? 0.25 : -0.25), 1, 10);
        viewState.zoom = trackZoom;
        track.style.width = `${trackZoom * 100}%`;
        requestAnimationFrame(() => {
            trackViewport.scrollLeft = Math.max(0, anchorRatio * track.offsetWidth - pointerX);
            viewState.scrollLeft = trackViewport.scrollLeft;
        });
    }, { passive: false });

    const selectedRanges = () => detectionRanges(model).map((range) => ({
        ...range,
        start: Math.max(range.start, rangeStart),
        end: Math.min(range.end, rangeEnd),
    })).filter((range) => range.end > range.start);

    const syncSelection = () => {
        const total = Math.max(1, model.totalFrames);
        seek.min = String(rangeStart + 1);
        seek.max = String(Math.max(rangeStart + 1, rangeEnd));
        selection.style.left = `${(rangeStart / total) * 100}%`;
        selection.style.width = `${((rangeEnd - rangeStart) / total) * 100}%`;
        rangeStartHandle.style.left = `${(rangeStart / total) * 100}%`;
        rangeEndHandle.style.left = `${(rangeEnd / total) * 100}%`;
    };

    const setMessage = (text, error = false) => {
        message.textContent = String(text || "");
        message.classList.toggle("error", error);
    };

    const syncTimeText = () => {
        seek.value = String(currentFrame + 1);
        time.textContent = t("player.rangeStatus", {
            current: currentFrame + 1,
            total: model.totalFrames,
            start: rangeStart + 1,
            end: rangeEnd,
            count: rangeEnd - rangeStart,
        });
    };

    const syncPlayer = (force = false) => {
        if (!model.totalFrames || !model.clips.length) return;
        currentFrame = clamp(Math.round(currentFrame), rangeStart, rangeEnd - 1);
        const entry = logicalEntry(model, currentFrame);
        const clip = model.clips[entry.clip] || model.clips[0];
        activeRange = selectedRanges().findIndex((range) => (
            currentFrame >= range.start && currentFrame < range.end
        ));
        if (force || activeClip !== entry.clip) {
            activeClip = entry.clip;
            player.src = viewUrl(clip);
        }
        const fps = Number(editor?.timeline?.frameRate) || 24;
        const target = entry.frame / fps;
        if (Number.isFinite(target) && Math.abs((player.currentTime || 0) - target) > 0.04) {
            try { player.currentTime = target; } catch { /* metadata not ready */ }
        }
        playhead.style.left = `${model.totalFrames > 1 ? (currentFrame / (model.totalFrames - 1)) * 100 : 0}%`;
        syncTimeText();
    };

    const stopAnimation = () => {
        if (raf) cancelAnimationFrame(raf);
        raf = 0;
        play.textContent = "▶";
    };

    const animate = () => {
        if (player.paused || player.ended) {
            stopAnimation();
            return;
        }
        const fps = Number(editor?.timeline?.frameRate) || 24;
        const ranges = selectedRanges();
        const range = ranges[activeRange];
        if (!range) {
            player.pause();
            stopAnimation();
            return;
        }
        const sourceFrame = Math.round(player.currentTime * fps);
        const lastEntry = logicalEntry(model, range.end - 1);
        if (sourceFrame >= lastEntry.frame) {
            currentFrame = range.end - 1;
            const nextRange = ranges[activeRange + 1];
            if (nextRange) {
                currentFrame = nextRange.start;
                syncPlayer(true);
                void player.play();
                raf = requestAnimationFrame(animate);
                return;
            }
            player.pause();
            syncPlayer();
            stopAnimation();
            return;
        }
        let mappedFrame = range.start;
        for (let logicalFrame = range.start; logicalFrame < range.end; logicalFrame++) {
            const entry = logicalEntry(model, logicalFrame);
            if (entry.frame > sourceFrame) break;
            mappedFrame = logicalFrame;
        }
        currentFrame = mappedFrame;
        playhead.style.left = `${model.totalFrames > 1 ? (currentFrame / (model.totalFrames - 1)) * 100 : 0}%`;
        syncTimeText();
        raf = requestAnimationFrame(animate);
    };

    play.onclick = () => {
        if (player.paused) {
            if (player.ended || currentFrame >= rangeEnd - 1) {
                currentFrame = rangeStart;
                syncPlayer(true);
            }
            void player.play();
            play.textContent = "Ⅱ";
            raf = requestAnimationFrame(animate);
        } else {
            player.pause();
            stopAnimation();
        }
    };
    mute.onclick = () => {
        player.muted = !player.muted;
        mute.textContent = player.muted ? "🔇" : "🔊";
        mute.title = t(player.muted ? "player.unmute" : "player.mute");
    };
    exportRange.onclick = async () => {
        exportRange.disabled = true;
        setMessage(t("player.exportingRange"));
        try {
            const clips = selectedRanges().map((range) => ({
                videoFile: range.clip.videoFile || range.clip.fileName,
                subfolder: range.clip.subfolder || "",
                type: range.clip.type || "input",
                sourceFrameStart: logicalEntry(model, range.start).frame,
                sourceFrameEnd: logicalEntry(model, range.end - 1).frame + 1,
            }));
            const response = await api.fetchApi("/minimax/director/export_video_range", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    clips,
                    frameRate: Number(editor?.timeline?.frameRate) || 24,
                }),
            });
            if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement("a");
            anchor.href = url;
            anchor.download = "minimax_selected_range.mp4";
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            URL.revokeObjectURL(url);
            setMessage("");
        } catch (error) {
            setMessage(t("player.exportRangeFailed", { err: error?.message || error }), true);
        } finally {
            exportRange.disabled = !model.totalFrames;
        }
    };
    seek.oninput = () => {
        currentFrame = clamp(Number(seek.value) - 1, rangeStart, rangeEnd - 1);
        syncPlayer();
    };
    const seekTrack = (event) => {
        const rect = track.getBoundingClientRect();
        currentFrame = clamp(
            Math.round(clamp((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1) * Math.max(0, model.totalFrames - 1)),
            rangeStart,
            rangeEnd - 1,
        );
        syncPlayer();
    };
    track.onpointerdown = (event) => {
        if (!model.totalFrames || event.button !== 0) return;
        event.preventDefault();
        event.stopPropagation();
        trackDragging = true;
        resumeAfterTrackDrag = !player.paused;
        player.pause();
        track.classList.add("dragging");
        try { track.setPointerCapture(event.pointerId); } catch { /* ignore */ }
        seekTrack(event);
    };
    track.onpointermove = (event) => {
        if (!trackDragging) return;
        event.preventDefault();
        seekTrack(event);
    };
    const finishTrackDrag = (event) => {
        if (!trackDragging) return;
        seekTrack(event);
        trackDragging = false;
        track.classList.remove("dragging");
        try { track.releasePointerCapture(event.pointerId); } catch { /* ignore */ }
        if (resumeAfterTrackDrag) {
            void player.play();
            play.textContent = "Ⅱ";
            raf = requestAnimationFrame(animate);
        }
        resumeAfterTrackDrag = false;
    };
    track.onpointerup = finishTrackDrag;
    track.onpointercancel = finishTrackDrag;

    const rangeFrameAt = (event) => {
        const rect = track.getBoundingClientRect();
        return Math.round(clamp((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1) * model.totalFrames);
    };
    const moveRangeHandle = (event) => {
        if (!rangeDragging) return;
        const minFrames = Math.min(MIN_SEGMENT_FRAMES, model.totalFrames);
        if (rangeDragging === "start") {
            rangeStart = clamp(rangeFrameAt(event), 0, rangeEnd - minFrames);
            currentFrame = rangeStart;
        } else if (rangeDragging === "end") {
            rangeEnd = clamp(rangeFrameAt(event), rangeStart + minFrames, model.totalFrames);
            currentFrame = rangeEnd - 1;
        } else {
            const length = rangeDragEnd - rangeDragStart;
            const nextStart = clamp(
                rangeDragStart + rangeFrameAt(event) - rangeDragAnchor,
                0,
                model.totalFrames - length,
            );
            rangeStart = nextStart;
            rangeEnd = nextStart + length;
            currentFrame = clamp(
                rangeDragCurrent + nextStart - rangeDragStart,
                rangeStart,
                rangeEnd - 1,
            );
        }
        syncSelection();
        syncPlayer();
        const previewEnd = alignedRangeEnd(rangeStart, rangeEnd);
        onRangePreview?.(rangeStart, previewEnd);
    };
    const startRangeDrag = (kind, event) => {
        if (!model.totalFrames || event.button !== 0) return;
        event.preventDefault();
        event.stopPropagation();
        rangeDragging = kind;
        rangeDragAnchor = rangeFrameAt(event);
        rangeDragStart = rangeStart;
        rangeDragEnd = rangeEnd;
        rangeDragCurrent = currentFrame;
        resumeAfterTrackDrag = !player.paused;
        player.pause();
        if (kind === "move") selection.classList.add("dragging");
        try { event.currentTarget.setPointerCapture(event.pointerId); } catch { /* ignore */ }
        moveRangeHandle(event);
    };
    const finishRangeDrag = (event) => {
        if (!rangeDragging) return;
        moveRangeHandle(event);
        const originalRangeEnd = rangeEnd;
        rangeEnd = alignedRangeEnd(rangeStart, rangeEnd);
        warnRangeAlignment(rangeStart, originalRangeEnd, rangeEnd);
        currentFrame = clamp(currentFrame, rangeStart, rangeEnd - 1);
        syncSelection();
        syncPlayer();
        rangeDragging = "";
        selection.classList.remove("dragging");
        try { event.currentTarget.releasePointerCapture(event.pointerId); } catch { /* ignore */ }
        onRangeChange?.(rangeStart, rangeEnd);
        if (resumeAfterTrackDrag) {
            currentFrame = rangeStart;
            syncPlayer(true);
            void player.play();
            play.textContent = "Ⅱ";
            raf = requestAnimationFrame(animate);
        }
        resumeAfterTrackDrag = false;
    };
    for (const [handle, kind] of [[rangeStartHandle, "start"], [rangeEndHandle, "end"]]) {
        handle.onpointerdown = (event) => startRangeDrag(kind, event);
        handle.onpointermove = moveRangeHandle;
        handle.onpointerup = finishRangeDrag;
        handle.onpointercancel = finishRangeDrag;
    }
    selection.onpointerdown = (event) => startRangeDrag("move", event);
    selection.onpointermove = moveRangeHandle;
    selection.onpointerup = finishRangeDrag;
    selection.onpointercancel = finishRangeDrag;
    player.onended = () => {
        const ranges = selectedRanges();
        const nextRange = ranges[activeRange + 1];
        if (nextRange) {
            currentFrame = nextRange.start;
        } else {
            stopAnimation();
            return;
        }
        syncPlayer(true);
        void player.play();
        raf = requestAnimationFrame(animate);
    };
    player.onpause = stopAnimation;
    player.onloadedmetadata = () => syncPlayer();

    const hasVideo = model.totalFrames > 0 && model.clips.length > 0;
    for (const control of [play, mute, exportRange]) {
        control.disabled = !hasVideo;
    }
    if (hasVideo) {
        syncSelection();
        syncPlayer(true);
        if (rangeEnd !== initialRangeEnd) {
            warnRangeAlignment(rangeStart, initialRangeEnd, rangeEnd);
            queueMicrotask(() => onRangeChange?.(rangeStart, rangeEnd));
        }
    }
    else setMessage(t("toolbar.noVideo"));

    return { root, destroy: stopAnimation };
}
