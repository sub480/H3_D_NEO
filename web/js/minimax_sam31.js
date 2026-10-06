/** SAM3.1 normal object analysis inside the Director drawer. */
import { api } from "../../scripts/api.js";
import { closePassPanels } from "./minimax_refine.js";
import { resolveTaskKey } from "./minimax_gen_timeline.js";

export const SAM31_WIDGET_NAMES = ["sam31_config"];

export function normalizeSam31Config(raw) {
    if (typeof raw === "string") raw = raw.trim() ? JSON.parse(raw) : {};
    raw ??= {};
    if (typeof raw !== "object" || Array.isArray(raw) || (raw.version ?? 1) !== 1) {
        throw new Error("不支持的 SAM3.1 配置格式或版本。");
    }
    const number = (value, fallback, low, high, integer = false) => {
        const empty = value == null || (typeof value === "string" && !value.trim());
        const n = empty ? fallback : Number(value);
        const bounded = Math.min(high, Math.max(low, Number.isFinite(n) ? n : fallback));
        return integer ? Math.trunc(bounded) : bounded;
    };
    return {
        version: 1,
        enabled: raw.enabled === true || ["1", "true", "yes", "on"].includes(String(raw.enabled).toLowerCase()),
        checkpoint: String(raw.checkpoint || "").trim().slice(0, 512),
        threshold: number(raw.threshold, 0.5, 0, 1),
        max_objects: number(raw.max_objects, 4, 1, 16, true),
        detect_interval: number(raw.detect_interval, 5, 1, 256, true),
        long_edge: number(raw.long_edge, 768, 256, 1024, true),
        target_group: String(raw.target_group || "").slice(0, 128),
    };
}

export function normalizeSam31Group(raw) {
    raw ??= {};
    if (typeof raw !== "object" || Array.isArray(raw) || (raw.version ?? 1) !== 1) {
        throw new Error("不支持的 SAM3.1 组配置版本。");
    }
    const composite = raw.composite && typeof raw.composite === "object" ? raw.composite : {};
    const bounded = (value, fallback, high, integer = false) => {
        const number = value == null || value === "" ? fallback : Number(value);
        const result = Math.min(high, Math.max(0, Number.isFinite(number) ? number : fallback));
        return integer ? Math.trunc(result) : result;
    };
    return {
        version: 1,
        prompt: String(raw.prompt || "").slice(0, 256),
        selected: [...new Set((Array.isArray(raw.selected) ? raw.selected : [])
            .filter(value => Number.isInteger(value) && value >= 0 && value < 16))]
            .sort((a, b) => a - b),
        result: raw.result && typeof raw.result === "object" && !Array.isArray(raw.result)
            ? raw.result : null,
        composite: {
            enabled: composite.enabled === true,
            grow: bounded(composite.grow, 0, 64, true),
            feather: bounded(composite.feather, 4, 64, true),
            blend: bounded(composite.blend, 1, 1),
        },
        ...(raw.jobId ? { jobId: String(raw.jobId) } : {}),
        ...(raw.jobSignature ? { jobSignature: String(raw.jobSignature) } : {}),
        ...(raw.resultSignature ? { resultSignature: String(raw.resultSignature) } : {}),
    };
}

function stableValue(value) {
    if (Array.isArray(value)) return value.map(stableValue);
    if (!value || typeof value !== "object") return value;
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, stableValue(value[key])]));
}

function mediaSignature(media) {
    if (!media || typeof media !== "object"
        || !(media.imageFile || media.videoFile || media.fileName)) return null;
    // Match persisted references, not metadata stripped by serialization.
    // Actual source content and dimensions are checked by the backend.
    return {
        imageFile: String(media.imageFile || ""), videoFile: String(media.videoFile || ""),
        fileName: String(media.fileName || ""), type: String(media.type || "input"),
        subfolder: String(media.subfolder || ""),
    };
}

export function sam31InputSignature(timeline, item, config) {
    const source = item.sourceVideo || {};
    const video = source.video || {};
    const start = Math.max(0, Math.round(Number(source.rangeStart) || 0));
    const end = source.rangeEnd == null ? start + 257 : Math.max(0, Math.round(Number(source.rangeEnd) || 0));
    const parameters = Object.fromEntries([
        "checkpoint", "threshold", "max_objects", "detect_interval", "long_edge",
    ].map(key => [key, config[key]]));
    return JSON.stringify(stableValue({
        id: String(item.id), taskType: item.taskType || "t2v",
        frameRate: Number(timeline.frameRate) || 24,
        parameters, prompt: String(item.sam31?.prompt || "").trim(),
        genImage: mediaSignature(item.genImage),
        startImage: mediaSignature(item.startImage), endImage: mediaSignature(item.endImage),
        refs: (item.refs || []).map(mediaSignature),
        sourceVideo: {
            mediaKind: source.mediaKind || "video", totalFrames: Number(source.totalFrames) || 0,
            rangeStart: source.rangeStart == null ? null : start,
            rangeEnd: source.rangeEnd == null ? null : end,
            image: mediaSignature(source.image),
            video: { ...mediaSignature(video), sourceFrameCount: Number(video.sourceFrameCount) || 0 },
            videoClips: (source.videoClips || []).map(clip => ({
                ...mediaSignature(clip), sourceFrameCount: Number(clip.sourceFrameCount) || 0,
            })),
            deletedSourceRanges: video.deletedSourceRanges || [],
            frameMap: (video.frameMap || []).slice(start, Math.min(end, start + 257)),
        },
    }));
}

function sourceInfo(item) {
    if (!item) return { kind: "", text: "请选择一个组。" };
    const task = resolveTaskKey(item.taskType || "t2v");
    const source = item.sourceVideo || {};
    let image;
    if (task === "i2v") image = item.genImage;
    else if (task === "fl2v") {
        image = item.startImage?.imageFile ? item.startImage : item.endImage;
    } else if (task === "r2v") image = item.refs?.[0];
    else if (["v2v", "rv2v"].includes(task) && source.mediaKind === "image") {
        image = source.image;
    } else if (["v2v", "rv2v"].includes(task)) {
        const video = source.video || source.videoClips?.[0];
        const name = video?.videoFile || video?.fileName;
        if (name) {
            return {
                kind: "video",
                text: `来源视频：${name}；逻辑帧范围 ${source.rangeStart ?? 0} 至 ${source.rangeEnd ?? source.totalFrames ?? "结尾"}（右端不含）。`,
            };
        }
    }
    const name = image?.imageFile || image?.fileName;
    return name
        ? { kind: "image", text: `来源图片：${name}` }
        : { kind: "", text: "当前组没有可分析的上传素材。T2V 无素材组不支持分割；不会自动分析生成结果。" };
}

export function mountDirectorSam31Panel(editor) {
    const node = editor?.node;
    const bar = editor?.outputBarEl;
    if (!node || !bar || editor.sam31PanelEl) return;
    const widget = node.widgets?.find(item => item.name === "sam31_config");
    if (!widget) return;

    const wrap = document.createElement("span");
    wrap.className = "bd-out-refine-wrap";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "bd-btn";
    button.textContent = "SAM3.1";
    button.title = "对象分割、视频跟踪及 V2V 源 latent 局部编辑";
    button.setAttribute("aria-expanded", "false");
    wrap.appendChild(button);
    const tools = bar.querySelector(".bd-live-preview-tools");
    if (tools) bar.insertBefore(wrap, tools);
    else bar.appendChild(wrap);

    const panel = document.createElement("div");
    panel.className = "bd-refine-panel hidden";
    panel.dataset.r = "sam31-panel";
    panel.style.maxHeight = "70vh";
    panel.style.overflowY = "auto";
    panel.innerHTML = `
        <div class="bd-refine-group">SAM3.1 对象分割</div>
        <label class="bd-refine-field row">
            <input type="checkbox" data-config="enabled"><span>启用分析工具</span>
        </label>
        <label class="bd-refine-field">
            <span>本地 checkpoint</span><select data-config="checkpoint"></select>
        </label>
        <label class="bd-refine-field">
            <span>来源组</span><select data-role="group"></select>
        </label>
        <label class="bd-refine-field">
            <span>短对象描述</span>
            <input type="text" data-role="prompt" maxlength="256" placeholder="例如：person / dog / car"
                style="width:100%;box-sizing:border-box;background:#111;color:#ddd;border:1px solid #333;border-radius:4px;height:26px;padding:0 6px;font-size:11px">
        </label>
        <label class="bd-refine-field">
            <span>检测阈值</span><input type="number" data-config="threshold" min="0" max="1" step="0.01">
        </label>
        <label class="bd-refine-field">
            <span>最多对象数</span><input type="number" data-config="max_objects" min="1" max="16" step="1">
        </label>
        <label class="bd-refine-field">
            <span>文本跟踪检测间隔（帧）</span><input type="number" data-config="detect_interval" min="1" max="256" step="1">
        </label>
        <label class="bd-refine-field">
            <span>分析长边（像素）</span><input type="number" data-config="long_edge" min="256" max="1024" step="32">
        </label>
        <div data-role="environment" role="status" style="grid-column:1/-1;white-space:pre-wrap;overflow-wrap:anywhere"></div>
        <div data-role="source" style="grid-column:1/-1;overflow-wrap:anywhere"></div>
        <div style="grid-column:1/-1;color:#888;font-size:11px;white-space:normal">
            仅分析上传素材。单次最多 256 帧，解码图像预算 384 MiB；超限请缩小长边或截取来源。
            跟踪从所选范围首帧正向开始。不提供点框编辑、双向跟踪或取消按钮。
            选择对象跟踪不再发现新对象，检测间隔在该模式下不生效。
        </div>
        <div style="grid-column:1/-1;display:flex;gap:6px;flex-wrap:wrap">
            <button type="button" class="bd-btn" data-action="detect">分割首帧</button>
            <button type="button" class="bd-btn" data-action="track">文本跟踪</button>
            <button type="button" class="bd-btn" data-action="track_selected">跟踪已选对象</button>
            <button type="button" class="bd-btn" data-role="refresh">刷新能力 / 任务</button>
            <button type="button" class="bd-btn" data-role="preview-button">校验并预览缓存</button>
        </div>
        <div data-role="status" role="status" style="grid-column:1/-1;white-space:pre-wrap;overflow-wrap:anywhere"></div>
        <div data-role="result" style="grid-column:1/-1;white-space:normal"></div>
        <div data-role="objects" style="grid-column:1/-1;display:flex;gap:10px;flex-wrap:wrap"></div>
        <div class="bd-refine-group">当前组 V2V 源 latent 局部编辑</div>
        <label class="bd-refine-field row">
            <input type="checkbox" data-compose="enabled"><span>源 latent 硬蒙版局部编辑，并贴回已选区域</span>
        </label>
        <label class="bd-refine-field"><span>生成区域扩张（像素，向上取整为 32px 格）</span>
            <input type="number" data-compose="grow" min="0" max="64" step="1"></label>
        <label class="bd-refine-field"><span>最终贴回羽化半径（输出像素）</span>
            <input type="number" data-compose="feather" min="0" max="64" step="1"></label>
        <label class="bd-refine-field"><span>最终贴回混合强度</span>
            <input type="number" data-compose="blend" min="0" max="1" step="0.05"></label>
        <div data-role="compose-status" style="grid-column:1/-1;white-space:normal"></div>
        <div style="grid-column:1/-1;color:#888;font-size:11px;white-space:normal">
            编辑要求写在该 V2V / RV2V 组提示词里，例如将现有服装改为白色。
            完成整个源范围的视频跟踪并勾选衣物对象后启用，正常运行导演台。
            源视频编码为实际待编辑 latent，不重复作为 Video 参考；采样使用逐帧硬蒙版和 H3 原生逐 token 时间步。
            需要关闭段间引导、二采、SelfLift 和 FaceRefine，帧范围须为 17k+5。
            适合普通衣物换色换装；提示词描述衣物编辑，保留原动作。跟踪错误或模型生成仍可能造成接缝，不保证视觉完全成功。
            扩张改变采样区域，会重跑首采；羽化和混合仅影响最终贴回。空帧在贴回时保留源画面。
        </div>
        <label class="bd-refine-field">
            <span>预览帧 <span data-role="frame-label">0</span></span>
            <input type="range" data-role="frame" min="0" max="0" value="0" step="1">
        </label>
        <label class="bd-refine-field">
            <span>蒙版透明度</span><input type="range" data-role="opacity" min="0" max="1" value="0.5" step="0.05">
        </label>
        <div data-role="mapping" style="grid-column:1/-1;white-space:normal"></div>
        <img data-role="preview" alt="SAM3.1 对象蒙版叠加预览"
            style="grid-column:1/-1;width:100%;max-height:440px;object-fit:contain;display:none">
    `;
    bar.after(panel);
    editor.sam31PanelEl = panel;
    editor.sam31BarEl = wrap;

    const el = role => panel.querySelector(`[data-role="${role}"]`);
    const actions = [...panel.querySelectorAll("[data-action]")];
    const configInputs = [...panel.querySelectorAll("[data-config]")];
    const requests = new Set();
    let disposed = false;
    let apiReady = false;
    let capability = { available: false, models: [] };
    let submitting = false;
    let polling = false;
    let pollToken = 0;
    let pollTimer = null;
    let wakePoll = null;
    let previewTimer = null;
    let previewToken = 0;
    let capabilityToken = 0;
    let configSignature = "";
    let groupListSignature = "";
    let objectSignature = "";
    let viewSignature = "";
    let activeGroupId = "";
    let previewBusy = false;

    const open = () => !!editor._mmxSAM31PanelOpen;
    const groups = () => editor.timeline?.segments || [];
    const byId = id => groups().find(item => String(item.id) === String(id));
    const config = () => {
        const raw = widget.value;
        return normalizeSam31Config(raw && typeof raw === "object"
            ? raw.content ?? raw.value ?? raw : raw);
    };
    const currentGroup = () => {
        const target = config().target_group;
        return target ? byId(target) : groups()[editor.selectedIndex ?? 0];
    };
    const state = item => normalizeSam31Group(item?.sam31);
    const persist = (id, change) => {
        const item = byId(id);
        if (!item) return null;
        item.sam31 = normalizeSam31Group(change(state(item)));
        editor.scheduleTimelineSync?.();
        node.setDirtyCanvas?.(true, true);
        return item;
    };
    const writeConfig = value => {
        widget.value = JSON.stringify(normalizeSam31Config(value));
        widget.callback?.(widget.value);
        editor.scheduleTimelineSync?.();
        node.setDirtyCanvas?.(true, true);
    };
    const signature = item => sam31InputSignature(editor.timeline, item, config());
    const bodyFor = id => {
        const timeline = editor.buildTimelinePayload();
        if (!timeline.segments.some(item => String(item.id) === String(id))) {
            throw new Error("原来源组已移除，不能继续本次操作。");
        }
        return { timeline, config: config(), group_id: String(id) };
    };
    const message = text => { if (!disposed) el("status").textContent = text; };
    const hideWidget = () => {
        widget.hidden = true;
        widget.options ??= {};
        widget.options.hidden = true;
        widget.computeSize = () => [0, -4];
        if (widget.element) widget.element.style.display = "none";
    };
    const abortKind = kind => {
        for (const entry of requests) {
            if (!kind || entry.kind === kind) entry.controller.abort();
        }
    };
    const fetchData = async (path, body, kind = "normal") => {
        if (disposed) throw new Error("SAM3.1 编辑器已关闭。");
        const entry = { kind, controller: new AbortController() };
        requests.add(entry);
        try {
            const response = await api.fetchApi(`/minimax/director/sam31/${path}`, {
                signal: entry.controller.signal,
                ...(body === undefined ? {} : {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body),
                }),
            });
            const text = await response.text();
            let data;
            try { data = JSON.parse(text); }
            catch {
                const error = new Error(`SAM3.1 接口返回非 JSON（HTTP ${response.status}）。请检查插件路由是否已加载。`);
                error.httpStatus = response.status;
                throw error;
            }
            if (!response.ok) {
                const detail = typeof data.error === "string" ? data.error : JSON.stringify(data.error || data);
                const error = new Error(`SAM3.1 请求失败：${detail}`);
                error.httpStatus = response.status;
                throw error;
            }
            return data;
        } finally { requests.delete(entry); }
    };
    const clearPreview = () => {
        previewToken++;
        clearTimeout(previewTimer);
        previewTimer = null;
        abortKind("preview");
        previewBusy = false;
        el("preview").removeAttribute("src");
        el("preview").style.display = "none";
        el("mapping").textContent = "";
    };
    const stopPolling = () => {
        pollToken++;
        polling = false;
        clearTimeout(pollTimer);
        pollTimer = null;
        const wake = wakePoll;
        wakePoll = null;
        wake?.();
        abortKind("poll");
    };
    const pause = () => {
        stopPolling();
        clearPreview();
        capabilityToken++;
        abortKind("capability");
        button.setAttribute("aria-expanded", "false");
    };
    editor.pauseSAM31Panel = pause;

    const sync = () => {
        if (disposed) return;
        hideWidget();
        panel.classList.toggle("hidden", !open());
        button.setAttribute("aria-expanded", String(open()));
        let cfg, item, data;
        try {
            cfg = config();
            item = currentGroup();
            data = state(item);
        } catch (error) {
            for (const action of actions) action.disabled = true;
            el("preview-button").disabled = true;
            message(error.message);
            return;
        }
        button.classList.toggle("active", cfg.enabled);
        const id = item ? String(item.id) : "";
        if (id !== activeGroupId) {
            stopPolling();
            clearPreview();
            activeGroupId = id;
            el("frame").value = "0";
            el("status").textContent = "";
        }
        const listSignature = JSON.stringify([cfg.target_group, groups().map((g, i) => [g.id, g.uiGroupName, g.taskType, i])]);
        if (listSignature !== groupListSignature) {
            groupListSignature = listSignature;
            const follow = document.createElement("option");
            follow.value = "";
            follow.textContent = "跟随导演台当前选中组";
            const options = groups().map((g, i) => {
                const option = document.createElement("option");
                option.value = String(g.id);
                option.textContent = `${i + 1}：${g.uiGroupName || g.taskType || "组"} [${g.id}]`;
                return option;
            });
            if (cfg.target_group && !byId(cfg.target_group)) {
                const missing = document.createElement("option");
                missing.value = cfg.target_group;
                missing.textContent = `原来源组已移除 [${cfg.target_group}]`;
                options.push(missing);
            }
            el("group").replaceChildren(follow, ...options);
            el("group").value = cfg.target_group;
        }
        const nextConfigSignature = JSON.stringify([cfg, capability.models]);
        if (nextConfigSignature !== configSignature) {
            configSignature = nextConfigSignature;
            for (const input of configInputs) {
                const name = input.dataset.config;
                if (name === "checkpoint") {
                    const values = [...new Set([cfg.checkpoint, ...(capability.models || [])].filter(Boolean))];
                    input.replaceChildren(...["", ...values].map(value => {
                        const option = document.createElement("option");
                        option.value = value;
                        option.textContent = value
                            ? `${value}${capability.models?.includes(value) ? "" : "（未发现此模型）"}`
                            : "请选择已安装的 SAM3.1 模型";
                        return option;
                    }));
                }
                if (input.type === "checkbox") input.checked = cfg[name];
                else input.value = cfg[name];
            }
        }
        // Input events persist immediately; focused fields still follow group
        // changes and workflow/snapshot restores.
        if (el("prompt").value !== data.prompt) el("prompt").value = data.prompt;
        el("prompt").disabled = !item;
        for (const input of panel.querySelectorAll("[data-compose]")) {
            const value = data.composite[input.dataset.compose];
            if (input.type === "checkbox") input.checked = value;
            else if (document.activeElement !== input) input.value = value;
            input.disabled = !item;
        }
        const info = sourceInfo(item);
        el("source").textContent = item ? `组 ID：${item.id}；${info.text}` : info.text;
        const inputSignature = item ? signature(item) : "";
        const nextView = JSON.stringify([id, inputSignature, data.result?.key, data.selected]);
        if (nextView !== viewSignature) {
            clearPreview();
            viewSignature = nextView;
        }
        const result = data.result;
        const objects = Array.isArray(result?.objects) ? result.objects.slice(0, 16) : [];
        const stale = !!result && data.resultSignature !== inputSignature;
        el("compose-status").textContent = !data.composite.enabled
            ? "当前组源 latent 局部编辑关闭：普通 V2V 行为不变。"
            : info.kind !== "video" || !["v2v", "rv2v"].includes(resolveTaskKey(item?.taskType || "t2v"))
                ? "当前组不是源视频 V2V / RV2V；运行时会停止，不会退回整片编辑。"
                : !["track", "track_selected"].includes(result?.action) || !data.selected.length || stale
                    ? "请完成当前输入的视频跟踪并勾选对象；无效或过期蒙版会阻止生成。"
                    : "已绑定此组跟踪缓存；正常运行导演台后，源 latent 将按已选对象硬蒙版局部编辑，再贴回源画面。";
        el("result").textContent = result
            ? `${result.action === "detect" ? "首帧分割" : "视频跟踪"}缓存：${objects.length} 个对象，${result.frames} 帧。${stale ? "配置或素材引用已变化，旧结果须重新校验。" : "预览时会重新校验素材内容与缓存身份。"}`
            : "尚无分割结果。对象编号仅在对应缓存结果内有效。";
        const nextObjects = JSON.stringify([id, result?.key, objects, data.selected]);
        if (nextObjects !== objectSignature) {
            const changedResult = el("objects").dataset.key !== `${id}:${result?.key || ""}`;
            objectSignature = nextObjects;
            el("objects").dataset.key = `${id}:${result?.key || ""}`;
            el("objects").replaceChildren();
            for (const object of objects) {
                if (!Number.isInteger(object.index) || object.index < 0 || object.index >= 16) continue;
                const label = document.createElement("label");
                label.className = "bd-refine-field row";
                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.checked = data.selected.includes(object.index);
                const score = Number.isFinite(object.score) ? `，${Math.round(object.score * 100)}%` : "";
                const origin = Number.isInteger(object.detection_index) ? `，原检测 ${object.detection_index}` : "";
                label.append(checkbox, document.createTextNode(`对象 ${object.index}${score}${origin}`));
                const resultKey = result.key;
                checkbox.addEventListener("change", () => {
                    persist(id, current => {
                        if (current.result?.key !== resultKey) return current;
                        const selected = new Set(current.selected);
                        if (checkbox.checked) selected.add(object.index);
                        else selected.delete(object.index);
                        return { ...current, selected: [...selected] };
                    });
                    sync();
                    schedulePreview();
                });
                el("objects").appendChild(label);
            }
            el("frame").max = String(Math.max(0, Math.min(256, Number(result?.frames) || 1) - 1));
            if (changedResult) el("frame").value = "0";
        }
        el("frame-label").textContent = `${el("frame").value} / ${el("frame").max}`;
        const ready = apiReady && capability.available
            && capability.models?.includes(cfg.checkpoint) && cfg.enabled && !!info.kind;
        for (const action of actions) {
            const kind = action.dataset.action;
            action.disabled = !ready || submitting || polling || !!data.jobId
                || (kind !== "detect" && info.kind !== "video")
                || (kind === "track_selected"
                    && (result?.action !== "detect" || stale || !data.selected.length));
        }
        el("preview-button").disabled = !apiReady || !result?.key || previewBusy || submitting;
        el("frame").disabled = !result?.key;
        el("opacity").disabled = !result?.key;
        if (data.jobId && !polling && !submitting && !el("status").textContent) {
            message("此组有未结算任务。点击“刷新能力 / 任务”查询；关闭抽屉不会取消已入队任务。");
        }
    };
    editor.syncSAM31Panel = sync;

    const requestPreview = async () => {
        if (disposed || !open()) return;
        let item, data, captured, token;
        try {
            item = currentGroup();
            data = state(item);
            if (!item || !data.result?.key) return;
            const body = bodyFor(item.id);
            const payloadItem = body.timeline.segments.find(g => String(g.id) === String(item.id));
            captured = sam31InputSignature(body.timeline, payloadItem, body.config);
            clearPreview();
            token = previewToken;
            previewBusy = true;
            sync();
            message("正在校验缓存并读取所选帧；不运行 GPU 推理。");
            const preview = await fetchData("preview", {
                ...body, key: data.result.key, selected: data.selected,
                frame: Number(el("frame").value), opacity: Number(el("opacity").value),
            }, "preview");
            const current = byId(item.id);
            if (disposed || !open() || token !== previewToken || !current
                || String(currentGroup()?.id) !== String(item.id)
                || signature(current) !== captured || state(current).result?.key !== data.result.key) return;
            el("preview").src = preview.image;
            el("preview").style.display = "";
            const map = preview.mapping;
            el("mapping").textContent = `来源片段 ${map.clip}，来源逻辑帧 ${map.source_frame}（${map.frame_rate} fps）；来源尺寸 ${map.source_size.join(" × ")}，分析尺寸 ${map.analysis_size.join(" × ")}。`;
            message("缓存校验通过。蒙版按来源比例映射显示，不裁切素材。");
            editor.updateDomWidgetHeight?.();
        } catch (error) {
            if (!disposed && (token == null || token === previewToken)
                && error.name !== "AbortError") message(error.message);
        } finally {
            if (!disposed && token === previewToken) {
                previewBusy = false;
                sync();
            }
        }
    };
    const schedulePreview = () => {
        // Invalidate the previous response immediately, not after debounce.
        clearPreview();
        if (disposed || !open()) return;
        previewTimer = setTimeout(() => {
            previewTimer = null;
            requestPreview();
        }, 200);
    };
    const settleJob = (id, jobId, job) => {
        const item = byId(id);
        if (!item || state(item).jobId !== jobId) return;
        const previous = state(item);
        const matches = !!previous.jobSignature && previous.jobSignature === signature(item);
        persist(id, current => {
            const next = { ...current };
            delete next.jobId;
            delete next.jobSignature;
            if (job.status === "complete" && matches) {
                next.result = job.result;
                next.selected = (job.result.objects || []).map(object => object.index);
                next.resultSignature = previous.jobSignature;
            }
            return next;
        });
        sync();
        if (job.status === "error") {
            message(`任务失败：${job.error || "后端未提供原因。"}\n可修正配置后重新分析。`);
        } else if (!matches) {
            message("任务已完成，但该组输入已变化或缺少任务输入快照。结果仍在独立缓存中，请按当前输入重新分析。");
        } else {
            message(`${job.result.cached ? "已复用有效缓存" : "分析完成"}：${job.result.objects.length} 个对象，${job.result.frames} 帧。`);
            schedulePreview();
        }
    };
    const monitor = async (id, jobId) => {
        stopPolling();
        const token = pollToken;
        polling = true;
        sync();
        try {
            while (!disposed && open() && token === pollToken
                && String(currentGroup()?.id) === String(id)) {
                const job = await fetchData(`job?job_id=${encodeURIComponent(jobId)}`, undefined, "poll");
                if (disposed || !open() || token !== pollToken) return;
                if (job.status === "complete" || job.status === "error") {
                    settleJob(id, jobId, job);
                    return;
                }
                const labels = {
                    submitting: "正在提交到 ComfyUI",
                    queued: "已进入 ComfyUI 队列，等待执行",
                    running: "ComfyUI 正在执行 SAM 分割 / 跟踪",
                };
                if (!labels[job.status]) throw new Error(`未知 SAM 任务状态：${job.status}`);
                message(`${labels[job.status]}。\n不显示未经后端确认的百分比进度；关闭抽屉只停止界面轮询。`);
                await new Promise(resolve => {
                    wakePoll = resolve;
                    pollTimer = setTimeout(() => {
                        pollTimer = null;
                        wakePoll = null;
                        resolve();
                    }, 1500);
                });
            }
        } catch (error) {
            if (!disposed && token === pollToken && error.name !== "AbortError") {
                if (error.httpStatus === 404) {
                    persist(id, current => {
                        if (current.jobId !== jobId) return current;
                        const next = { ...current };
                        delete next.jobId;
                        delete next.jobSignature;
                        return next;
                    });
                }
                message(`${error.message}\n刷新可重试查询；未知或过期任务可重新分析。`);
            }
        } finally {
            if (token === pollToken) {
                polling = false;
                if (!disposed) sync();
            }
        }
    };
    const refresh = async () => {
        if (disposed || !open() || submitting) return;
        const token = ++capabilityToken;
        abortKind("capability");
        el("environment").textContent = "正在检查原生能力与本地模型列表，不加载模型权重。";
        try {
            const data = await fetchData("capabilities", undefined, "capability");
            if (disposed || !open() || token !== capabilityToken) return;
            apiReady = true;
            capability = { ...data, models: Array.isArray(data.models) ? data.models : [] };
            const missing = !data.available || !capability.models.length;
            el("environment").textContent = missing
                ? `SAM3.1 暂不可运行：${data.reason || "未发现可用模型。"}\n不会自动安装依赖、下载权重或更新 ComfyUI。`
                : `原生 SAM3.1 可用，已发现 ${capability.models.length} 个本地 checkpoint。`;
            sync();
            const item = currentGroup();
            if (item && state(item).jobId && !polling) {
                await monitor(String(item.id), state(item).jobId);
            }
        } catch (error) {
            if (!disposed && token === capabilityToken && error.name !== "AbortError") {
                apiReady = false;
                capability = { available: false, models: [] };
                el("environment").textContent = `SAM3.1 接口不可用：${error.message}`;
                sync();
            }
        }
    };
    for (const input of configInputs) {
        input.addEventListener("change", () => {
            try {
                const cfg = config();
                cfg[input.dataset.config] = input.type === "checkbox" ? input.checked : input.value;
                writeConfig(cfg);
                clearPreview();
                sync();
            } catch (error) { message(error.message); }
        });
    }
    for (const input of panel.querySelectorAll("[data-compose]")) {
        input.addEventListener("change", () => {
            try {
                const item = currentGroup();
                if (!item) return;
                const value = input.type === "checkbox" ? input.checked : input.value;
                persist(item.id, current => ({
                    ...current, composite: { ...current.composite, [input.dataset.compose]: value },
                }));
                sync();
            } catch (error) { message(error.message); }
        });
    }
    el("group").addEventListener("change", () => {
        try {
            writeConfig({ ...config(), target_group: el("group").value });
            sync();
            if (currentGroup() && state(currentGroup()).jobId) refresh();
        } catch (error) { message(error.message); }
    });
    el("prompt").addEventListener("input", () => {
        try {
            const item = currentGroup();
            if (!item) return;
            persist(item.id, current => ({ ...current, prompt: el("prompt").value }));
            sync();
        } catch (error) { message(error.message); }
    });
    el("frame").addEventListener("input", () => {
        el("frame-label").textContent = `${el("frame").value} / ${el("frame").max}`;
        schedulePreview();
    });
    el("opacity").addEventListener("input", schedulePreview);
    el("preview-button").addEventListener("click", requestPreview);
    el("refresh").addEventListener("click", refresh);
    panel.addEventListener("keydown", event => event.stopPropagation());

    for (const action of actions) {
        action.addEventListener("click", async () => {
            if (disposed || submitting || polling || action.disabled) return;
            let id;
            submitting = true;
            try {
                const item = currentGroup();
                if (!item) throw new Error("请选择来源组。");
                id = String(item.id);
                const body = bodyFor(id);
                const payloadItem = body.timeline.segments.find(g => String(g.id) === id);
                const captured = sam31InputSignature(body.timeline, payloadItem, body.config);
                sync();
                message("正在校验素材与参数并准备入队；尚未确认开始 GPU 推理。");
                const submitted = await fetchData("submit", {
                    ...body, action: action.dataset.action,
                }, "submit");
                if (disposed) return;
                persist(id, current => ({
                    ...current, jobId: submitted.job_id, jobSignature: captured,
                }));
                submitting = false;
                sync();
                if (!byId(id)) message(`任务已入队，但原来源组已移除。任务 ID：${submitted.job_id}`);
                else if (open() && String(currentGroup()?.id) === id) {
                    await monitor(id, submitted.job_id);
                }
            } catch (error) {
                if (!disposed && error.name !== "AbortError") {
                    if (error.httpStatus === 404 || error.httpStatus >= 500) {
                        apiReady = false;
                        el("environment").textContent = "SAM3.1 接口不可用，请刷新检查路由状态。";
                    }
                    message(error.message);
                }
            } finally {
                submitting = false;
                if (!disposed) sync();
            }
        });
    }
    button.addEventListener("click", () => {
        const next = !open();
        closePassPanels(editor, next ? "sam31" : "");
        editor._mmxSAM31PanelOpen = next;
        if (!next) pause();
        sync();
        editor.resizeNodeForContentMinChange?.();
        editor.updateDomWidgetHeight?.();
        if (next) refresh();
    });
    editor.disposeSAM31Panel = () => {
        if (disposed) return;
        disposed = true;
        pause();
        abortKind();
        panel.remove();
        wrap.remove();
        editor.syncSAM31Panel = null;
        editor.pauseSAM31Panel = null;
        editor.sam31PanelEl = null;
        editor.sam31BarEl = null;
        editor._mmxSAM31PanelOpen = false;
    };
    el("environment").textContent = "打开抽屉后检查原生 SAM3.1 与本地模型。";
    sync();
}
