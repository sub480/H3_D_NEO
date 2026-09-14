/** Server-side Director snapshots, stored as Director-pack zips. */
import { api } from "../../scripts/api.js";
import { t, applyI18nDom } from "./minimax_i18n.js";
import { resolveMixedGroupKey, resolveTaskKey } from "./minimax_gen_timeline.js";

const MAX_NAME = 80;
const SNAPSHOT_WIDGET_NAMES = ["steps", "sampler", "scheduler", "cfg", "shift_video", "shift_audio", "seed"];

async function request(path, body, method = "POST") {
    const options = { method, headers: { "Content-Type": "application/json" } };
    if (body !== undefined) options.body = JSON.stringify(body);
    const response = await api.fetchApi(path, options);
    if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
    return response.json();
}

function timestampOf(snapshot) { return snapshot.updatedAt || snapshot.createdAt || 0; }
function sort(items) { return [...items].sort((a, b) => timestampOf(b) - timestampOf(a) || a.name.localeCompare(b.name)); }
function nameOf(value) { return String(value ?? "").trim().slice(0, MAX_NAME); }
function stableValue(value) {
    if (Array.isArray(value)) return value.map(stableValue);
    if (!value || typeof value !== "object") return value;
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stableValue(value[key])]));
}
function stateSignature(state) { return JSON.stringify(stableValue(state)); }
function collectSnapshotWidgets(editor) {
    const widgets = {};
    for (const name of SNAPSHOT_WIDGET_NAMES) {
        const widget = editor.widget?.(name);
        if (widget && widget.value != null && widget.value !== "") widgets[name] = widget.value;
    }
    const task = editor.taskTypeWidget?.value || editor.timeline?.global?.taskType || "";
    if (task) widgets.task_type = task;
    return widgets;
}

function pruneHiddenMedia(target, taskKey) {
    if (!target || typeof target !== "object") return;
    const keep = new Set(taskKey === "i2v"
        ? ["genImage", "imageFile"]
        : taskKey === "fl2v"
            ? ["startImage", "endImage"]
            : taskKey === "r2v" || taskKey === "rv2v"
                ? ["refs", "refAudios", "refVideos"]
                : []);
    for (const key of ["genImage", "imageFile", "startImage", "endImage", "refs", "refAudios", "refVideos", "referenceVideo"]) {
        if (!keep.has(key)) delete target[key];
    }
}

export function buildSnapshotTimeline(editor) {
    const timeline = JSON.parse(JSON.stringify(editor.buildTimelinePayload()));
    delete timeline.batchWorkspace;
    delete timeline.batchWorkspaces;
    delete timeline.videoWorkspace;
    delete timeline.videoWorkspaces;
    delete timeline.fl2vWorkspace;
    delete timeline.video;
    delete timeline.videoClips;
    delete timeline.shots;
    delete timeline.keyframes;
    const groups = Array.isArray(timeline.segments) ? timeline.segments : [];
    for (const group of groups) {
        pruneHiddenMedia(group, resolveMixedGroupKey(group));
    }
    pruneHiddenMedia(timeline.global, "mixed");
    return timeline;
}

export function bindSnapshotSelector(editor) {
    const select = editor.globalTask;
    if (!select) return null;
    const updateButton = editor.snapshotUpdateBtn;
    const resetButton = editor.snapshotResetBtn;

    let items = [];
    let activeSnapshotId = null;
    let currentWorkspace = null;
    let busy = false;
    let baselineSignature = null;
    let dirty = false;
    let dirtyTimer = null;

    const mixedValue = () => {
        const options = editor.taskTypeWidget?.options?.values || [];
        return options.find((value) => resolveTaskKey(value) === "mixed")
            || editor.taskTypeWidget?.value
            || "mixed";
    };
    const renderControls = () => {
        const mixed = !activeSnapshotId;
        if (updateButton) {
            updateButton.disabled = !mixed && !dirty;
            updateButton.classList.toggle("active", mixed || dirty);
        }
        if (resetButton) {
            resetButton.disabled = busy || (!mixed && !dirty);
            resetButton.classList.toggle("active", mixed || dirty);
        }
    };
    const render = () => {
        const value = mixedValue();
        select.textContent = "";
        const current = document.createElement("option");
        current.value = "current";
        current.dataset.snapshotId = "";
        current.textContent = t("task.option", { key: "mixed", label: t("task.mixed") });
        select.appendChild(current);
        for (const snapshot of items) {
            const option = document.createElement("option");
            option.value = `snapshot:${snapshot.id}`;
            option.dataset.snapshotId = snapshot.id;
            option.textContent = `${t("toolbar.snapshots")} · ${snapshot.name}`;
            select.appendChild(option);
        }
        const index = items.findIndex((snapshot) => snapshot.id === activeSnapshotId);
        select.selectedIndex = index >= 0 ? index + 1 : 0;
        if (editor.taskTypeWidget) editor.taskTypeWidget.value = value;
        renderControls();
    };
    const setItems = (nextItems) => {
        items = sort(Array.isArray(nextItems) ? nextItems : []);
        if (activeSnapshotId && !items.some((item) => item.id === activeSnapshotId)) {
            activeSnapshotId = null;
        }
        render();
    };
    const refresh = async () => {
        const data = await request("/minimax/director/snapshots", undefined, "GET");
        setItems(data.items);
    };
    const applyWorkspace = (timeline, widgets) => {
        editor._snapshotApplying = true;
        try {
            editor.applyImportedTimeline(timeline, widgets || {});
        } finally {
            editor._snapshotApplying = false;
        }
    };
    const captureState = (flush = false) => {
        if (flush) editor.flushTimelineSync?.();
        return {
            timeline: buildSnapshotTimeline(editor),
            widgets: collectSnapshotWidgets(editor),
        };
    };
    const setBaseline = () => {
        baselineSignature = stateSignature(captureState(true));
        dirty = false;
        renderControls();
    };
    const checkDirty = () => {
        clearTimeout(dirtyTimer);
        dirtyTimer = null;
        if (busy || !activeSnapshotId || baselineSignature == null) return;
        dirty = stateSignature(captureState()) !== baselineSignature;
        renderControls();
    };
    const notifyChanged = () => {
        if (busy || !activeSnapshotId || baselineSignature == null) return;
        clearTimeout(dirtyTimer);
        dirtyTimer = setTimeout(checkDirty, 0);
    };
    const switchSelection = async () => {
        if (busy) return;
        const snapshotId = select.selectedOptions?.[0]?.dataset?.snapshotId || null;
        if (snapshotId === activeSnapshotId) return;
        busy = true;
        select.disabled = true;
        renderControls();
        try {
            if (!snapshotId) {
                if (currentWorkspace) {
                    applyWorkspace(currentWorkspace.timeline, currentWorkspace.widgets);
                }
                activeSnapshotId = null;
                currentWorkspace = null;
                baselineSignature = null;
                dirty = false;
                render();
                return;
            }
            if (!activeSnapshotId) {
                editor.flushTimelineSync?.();
                currentWorkspace = {
                    timeline: buildSnapshotTimeline(editor),
                    widgets: collectSnapshotWidgets(editor),
                };
            }
            const data = await request("/minimax/director/snapshots/restore", { id: snapshotId });
            if (!data?.timeline) throw new Error(t("snapshot.restoreError"));
            applyWorkspace(data.timeline, data.widgets || {});
            activeSnapshotId = snapshotId;
            setBaseline();
            render();
        } catch (error) {
            console.error("[MiniMax H3 Director] snapshot selector:", error);
            render();
            await editor.showBdMessage?.(t("snapshot.errorTitle"), String(error?.message || error));
        } finally {
            busy = false;
            select.disabled = false;
            renderControls();
        }
    };
    const updateCurrent = async () => {
        if (busy) return;
        busy = true;
        select.disabled = true;
        renderControls();
        try {
            const state = captureState(true);
            if (!activeSnapshotId) {
                currentWorkspace = state;
                const result = await request("/minimax/director/snapshots/save", {
                    name: timestampName(),
                    ...state,
                });
                activeSnapshotId = result.id;
                setItems([result, ...items.filter((item) => item.id !== result.id)]);
            } else {
                const result = await request("/minimax/director/snapshots/update", {
                    id: activeSnapshotId,
                    ...state,
                });
                setItems(items.map((item) => item.id === result.id ? result : item));
            }
            baselineSignature = stateSignature(state);
            dirty = false;
            render();
        } catch (error) {
            console.error("[MiniMax H3 Director] snapshot update:", error);
            await editor.showBdMessage?.(t("snapshot.errorTitle"), String(error?.message || error));
        } finally {
            busy = false;
            select.disabled = false;
            renderControls();
        }
    };
    const resetSnapshot = async () => {
        if (busy) return;
        if (!activeSnapshotId) {
            editor._resetBatchWorkspaceLive?.("mixed");
            editor.timeline.global.taskType = "mixed";
            editor._batchWsMem = editor._batchWsMem || {};
            editor._batchWsMem.mixed = editor._captureBatchWorkspace?.();
            editor._persistCurrentBatchWorkspace?.();
            editor.renderImageBatchGroups?.();
            editor.syncMixedCommonLayout?.();
            editor.updateSelectionUI?.();
            editor.updateVideoNameLabel?.();
            editor.commit?.(false, { syncTimeline: true });
            renderControls();
            return;
        }
        busy = true;
        select.disabled = true;
        renderControls();
        try {
            const data = await request("/minimax/director/snapshots/restore", { id: activeSnapshotId });
            if (!data?.timeline) throw new Error(t("snapshot.restoreError"));
            applyWorkspace(data.timeline, data.widgets || {});
            setBaseline();
            render();
        } catch (error) {
            console.error("[MiniMax H3 Director] snapshot reset:", error);
            await editor.showBdMessage?.(t("snapshot.errorTitle"), String(error?.message || error));
        } finally {
            busy = false;
            select.disabled = false;
            renderControls();
        }
    };
    const resetCurrent = () => {
        activeSnapshotId = null;
        currentWorkspace = null;
        baselineSignature = null;
        dirty = false;
        render();
    };
    select.onchange = () => { void switchSelection(); };
    if (updateButton) updateButton.onclick = () => { void updateCurrent(); };
    if (resetButton) resetButton.onclick = () => { void resetSnapshot(); };
    render();
    void refresh().catch((error) => {
        console.error("[MiniMax H3 Director] snapshot list:", error);
    });

    return {
        render,
        refresh,
        resetCurrent,
        setItems,
        notifyChanged,
        destroy() {
            clearTimeout(dirtyTimer);
            if (select.onchange) select.onchange = null;
            if (updateButton) updateButton.onclick = null;
            if (resetButton) resetButton.onclick = null;
        },
    };
}

function timestampName(prefix = "") {
    const now = new Date();
    const stamp = [now.getFullYear(), String(now.getMonth() + 1).padStart(2, "0"), String(now.getDate()).padStart(2, "0")].join("")
        + `-${String(now.getHours()).padStart(2, "0")}${String(now.getMinutes()).padStart(2, "0")}${String(now.getSeconds()).padStart(2, "0")}${String(now.getMilliseconds()).padStart(3, "0")}`;
    const prefixText = String(prefix).slice(0, 30);
    return nameOf(prefixText ? `${prefixText}-${stamp}` : stamp);
}
function summary(snapshot) {
    const size = Number(snapshot.size || 0);
    const sizeText = size >= 1024 * 1024 ? `${(size / 1024 / 1024).toFixed(1)} MB` : `${Math.max(0, Math.round(size / 1024))} KB`;
    return `${snapshot.name}\n${t("snapshot.created")}: ${new Date(timestampOf(snapshot)).toLocaleString()}\n${t("snapshot.file")}: ${snapshot.filename}\n${t("snapshot.size")}: ${sizeText}`;
}

export function bindSnapshotActions(editor) {
    const modal = document.createElement("div");
    modal.className = "bd-snapshot-modal hidden";
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.inert = true;
     modal.innerHTML = `<div class="bd-snapshot-box"><header><strong data-i18n="snapshot.title">快照</strong><button class="bd-icon-btn" data-snap="close" aria-label="×">×</button></header><div class="bd-snapshot-body"><aside><button class="bd-btn bd-btn-primary" data-snap="save" data-i18n="snapshot.save">保存当前配置</button><div class="bd-snapshot-count" data-snap="count"></div><div class="bd-snapshot-list" data-snap="list"></div></aside><section><div class="bd-snapshot-detail" data-snap="detail"></div><div class="bd-snapshot-actions"><button class="bd-btn bd-btn-primary" data-snap="restore" data-i18n="snapshot.restore">还原</button><button class="bd-btn" data-snap="export" data-i18n="snapshot.export">导出</button><button class="bd-btn" data-snap="import" data-i18n="snapshot.import">导入</button><button class="bd-btn" data-snap="rename" data-i18n="snapshot.rename">重命名</button><button class="bd-btn" data-snap="duplicate" data-i18n="snapshot.duplicate">复制</button><button class="bd-btn bd-btn-danger" data-snap="remove" data-i18n="snapshot.remove">删除</button></div></section></div></div>`;
    editor._snapshotModal = modal;
    let items = [];
    let selected = null;
    let busy = false;
    const listEl = modal.querySelector('[data-snap="list"]');
    const detailEl = modal.querySelector('[data-snap="detail"]');
    const buttons = [...modal.querySelectorAll("button[data-snap]")];
    const selectedSnapshot = () => items.find((item) => item.id === selected);
    const alertInView = (message, title = t("snapshot.errorTitle")) => new Promise((resolve) => {
        const bar = document.createElement("div");
        bar.className = "bd-snapshot-confirm bd-snapshot-alert";
        const text = document.createElement("span");
        text.textContent = `${title}: ${message}`;
        const close = document.createElement("button");
        close.className = "bd-btn"; close.textContent = t("dialog.confirm");
        const finish = () => { bar.remove(); resolve(); };
        close.onclick = finish;
        bar.append(text, close);
        modal.querySelector(".bd-snapshot-actions").before(bar);
        close.focus();
    });
    const alertError = (error) => alertInView(String(error?.message || error));
    const setBusy = (value) => { busy = value; buttons.forEach((button) => { if (button.dataset.snap !== "close") button.disabled = value; }); };
    const downloadSnapshot = async (snapshot) => {
        const response = await api.fetchApi(`/minimax/director/snapshots/export?id=${encodeURIComponent(snapshot.id)}`);
        if (!response.ok) throw new Error((await response.text()) || t("snapshot.exportError"));
        const blob = await response.blob();
        if (!blob.size) throw new Error(t("snapshot.exportError"));
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url; link.download = `${snapshot.name}.mmxsnapshot.zip`;
        document.body.appendChild(link); link.click(); link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 2000);
    };
    const importSnapshot = () => new Promise((resolve, reject) => {
        const input = document.createElement("input");
        input.type = "file"; input.accept = ".zip,application/zip"; input.style.display = "none";
        let settled = false;
        const cleanup = () => {
            window.removeEventListener("focus", handleWindowFocus, true);
            input.remove();
        };
        const finish = (callback, value) => {
            if (settled) return;
            settled = true;
            cleanup();
            callback(value);
        };
        const handleWindowFocus = () => setTimeout(() => {
            if (!input.files?.length) finish(resolve, null);
        }, 100);
        input.onchange = async () => {
            const file = input.files?.[0];
            if (!file) return finish(resolve, null);
            cleanup();
            try {
                const body = new FormData(); body.append("snapshot", file, file.name);
                const response = await api.fetchApi("/minimax/director/snapshots/import", { method: "POST", body });
                if (!response.ok) throw new Error((await response.text()) || t("snapshot.importError"));
                finish(resolve, await response.json());
            } catch (error) { finish(reject, error); }
        };
        input.oncancel = () => finish(resolve, null);
        window.addEventListener("focus", handleWindowFocus, true);
        document.body.appendChild(input);
        input.click();
    });

    const load = async () => {
        const data = await request("/minimax/director/snapshots", undefined, "GET");
        items = sort(Array.isArray(data.items) ? data.items : []);
        if (!items.some((item) => item.id === selected)) selected = items[0]?.id || null;
        render();
        editor._snapshotSelector?.setItems(items);
    };
    const renderSelection = () => {
        for (const row of listEl.querySelectorAll("[data-snapshot-id]")) {
            row.classList.toggle("active", row.dataset.snapshotId === selected);
        }
        const snapshot = selectedSnapshot();
        detailEl.textContent = snapshot ? summary(snapshot) : t("snapshot.noSelection");
    };
    const render = () => {
        modal.querySelector('[data-snap="count"]').textContent = t("snapshot.count", { n: items.length });
        listEl.textContent = "";
        if (!items.length) listEl.textContent = t("snapshot.empty");
        for (const snapshot of items) {
            const button = document.createElement("div");
            button.dataset.snapshotId = snapshot.id;
            button.className = `bd-snapshot-item${snapshot.id === selected ? " active" : ""}`;
            button.setAttribute("role", "button"); button.tabIndex = 0;
            const name = document.createElement("div");
            name.className = "bd-snapshot-name"; name.textContent = snapshot.name;
            name.ondblclick = (event) => {
                event.preventDefault(); event.stopPropagation();
                selected = snapshot.id; renderSelection(); beginInlineRename(snapshot);
            };
            const time = document.createElement("div");
            time.textContent = new Date(timestampOf(snapshot)).toLocaleString();
            button.append(name, time);
            button.onclick = () => { selected = snapshot.id; renderSelection(); };
            button.onkeydown = (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selected = snapshot.id; renderSelection(); } };
            listEl.appendChild(button);
        }
        renderSelection();
    };
    const operation = async (fn) => {
        if (busy) return;
        setBusy(true);
        try { await fn(); await load(); } catch (error) { console.error("[MiniMax H3 Director] snapshot:", error); await alertError(error); }
        finally { setBusy(false); }
    };
    const beginInlineRename = (snapshot) => {
        if (busy) return;
        const row = [...listEl.querySelectorAll("[data-snapshot-id]")].find((el) => el.dataset.snapshotId === snapshot.id);
        if (!row || row.querySelector("input")) return;
        row.textContent = "";
        const input = document.createElement("input");
        // Do not select the existing value.  ComfyUI/LiteGraph can make an
        // inactive selection look gray after a graph restore.  An empty field
        // with the old name as placeholder avoids that state entirely.
        input.type = "text"; input.value = ""; input.placeholder = snapshot.name; input.maxLength = MAX_NAME;
        input.className = "bd-snapshot-name-input";
        let finished = false;
        const finish = (save) => {
            if (finished) return;
            finished = true;
            if (!save || !nameOf(input.value)) { render(); return; }
            operation(async () => {
                const result = await request("/minimax/director/snapshots/rename", { id: snapshot.id, name: nameOf(input.value) });
                selected = result.id;
            });
        };
        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter") { event.preventDefault(); finish(true); }
            if (event.key === "Escape") { event.preventDefault(); finish(false); }
        }, true);
        input.onclick = (event) => event.stopPropagation();
        input.onblur = () => setTimeout(() => {
            if (document.activeElement !== input) finish(true);
        }, 0);
        row.appendChild(input); input.focus();
    };

    modal.querySelector('[data-snap="save"]').onclick = () => operation(async () => {
        const name = timestampName();
        editor.flushTimelineSync?.();
        const result = await request("/minimax/director/snapshots/save", { name, timeline: buildSnapshotTimeline(editor), widgets: collectSnapshotWidgets(editor) });
        selected = result.id;
    });
    modal.querySelector('[data-snap="export"]').onclick = () => operation(async () => {
        const snapshot = selectedSnapshot(); if (!snapshot) return;
        await downloadSnapshot(snapshot);
    });
    modal.querySelector('[data-snap="import"]').onclick = () => operation(async () => {
        const result = await importSnapshot();
        if (result?.id) selected = result.id;
    });
    modal.querySelector('[data-snap="rename"]').onclick = () => {
        const snapshot = selectedSnapshot(); if (!snapshot) return;
        beginInlineRename(snapshot);
    };
    modal.querySelector('[data-snap="duplicate"]').onclick = () => operation(async () => {
        const snapshot = selectedSnapshot(); if (!snapshot) return;
        const name = timestampName(`${snapshot.name}-${t("snapshot.copySuffix")}`);
        const result = await request("/minimax/director/snapshots/duplicate", { id: snapshot.id, name }); selected = result.id;
    });
    modal.querySelector('[data-snap="remove"]').onclick = () => operation(async () => {
        const snapshot = selectedSnapshot(); if (!snapshot) return;
        await request("/minimax/director/snapshots/delete", { id: snapshot.id }); selected = null;
    });
    modal.querySelector('[data-snap="restore"]').onclick = () => operation(async () => {
        const snapshot = selectedSnapshot(); if (!snapshot) return;
        const data = await request("/minimax/director/snapshots/restore", { id: snapshot.id });
        if (!data?.timeline) throw new Error(t("snapshot.restoreError"));
        close();
        editor.applyImportedTimeline(data.timeline, data.widgets || {});
    });
    const close = () => {
        if (modal.contains(document.activeElement)) document.activeElement.blur?.();
        modal.classList.add("hidden");
        modal.inert = true;
        // Remove the overlay completely instead of leaving a hidden fixed DOM
        // subtree behind.  ComfyUI Desktop's shortcut routing is sensitive to
        // stale DOM-widget focus/ownership after modal close.
        modal.remove();
    };
    modal.querySelector('[data-snap="close"]').onclick = close;
    modal.addEventListener("click", (event) => { if (event.target === modal) close(); });
    const keydown = (event) => { if (event.key === "Escape") { event.preventDefault(); close(); } };
    modal.addEventListener("keydown", keydown);
    editor._snapshotCleanup = () => { modal.remove(); };
    const open = async () => {
        if (busy) return;
        if (!modal.isConnected) document.body.appendChild(modal);
        modal.classList.remove("hidden");
        modal.inert = false;
        setBusy(true);
        try { await load(); } catch (error) { await alertError(error); }
        finally { setBusy(false); }
    };
    applyI18nDom(modal); render();
    return open;
}
