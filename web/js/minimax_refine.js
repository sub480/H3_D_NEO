/** MiniMax H3 Director built-in refine controls. */

import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import {
    CUSTOM_ASPECT_RATIO,
    RESOLUTION_ASPECTS,
    resolutionFromSelector,
    snapResolutionDim,
} from "./minimax_gen_timeline.js";
import { t } from "./minimax_i18n.js";
import {
    applyDirectorRefinePassDefaults,
    directorRefineActive,
} from "./minimax_image_batch.js";

const DIRECTOR_CLASSES = new Set(["H3_D_NEO"]);
const FOLLOW_DIRECTOR_ASPECT = "跟随导演台";

function widgetByName(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function widgetValue(w) {
    if (!w) return undefined;
    const v = w.value;
    if (v && typeof v === "object") {
        if (typeof v.content === "string") return v.content;
        if (typeof v.value === "string") return v.value;
    }
    return v;
}

function setWidgetVisible(node, name, visible) {
    const w = widgetByName(node, name);
    if (!w) return;
    w.hidden = !visible;
    if (!w.options) w.options = {};
    w.options.hidden = !visible;
    if (visible) {
        if (w._mmxOrigComputeSize) {
            w.computeSize = w._mmxOrigComputeSize;
            delete w._mmxOrigComputeSize;
        } else if (w.computeSize) {
            delete w.computeSize;
        }
        if (w.element) w.element.style.display = "";
    } else {
        if (!w._mmxOrigComputeSize && typeof w.computeSize === "function") {
            w._mmxOrigComputeSize = w.computeSize.bind(w);
        }
        w.computeSize = () => [0, -4];
        w.draw = function () {};
        if (w.element) w.element.style.display = "none";
    }
}

function isCustomAspect(value) {
    const v = String(value ?? "").trim();
    return v === CUSTOM_ASPECT_RATIO || v === "Custom" || v.startsWith("自定义");
}

const ASPECT_CHOICES = new Set([
    FOLLOW_DIRECTOR_ASPECT,
    "Follow Director",
    CUSTOM_ASPECT_RATIO,
    "Custom",
    "1:1 (方形)",
    "2:3 (竖版照片)",
    "3:2 (横版照片)",
    "3:4 (竖版标准)",
    "4:3 (标准)",
    "9:16 (竖屏)",
    "16:9 (宽屏)",
    "21:9 (超宽)",
]);

const UPSCALE_METHOD_VALUES = new Set(["lanczos", "nvidia_rtx_vsr", "h3_latent"]);
const SEED_MODE_VALUES = new Set(["inherit", "offset"]);
const SAMPLER_HINTS = new Set([
    "euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral",
    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
    "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v",
    "deis", "res_multistep", "res_multistep_ancestral", "gradient_estimation",
    "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece",
    "uni_pc", "uni_pc_bh2", "ddim",
]);

function looksLikeUpscaleMethod(value) {
    return UPSCALE_METHOD_VALUES.has(String(value ?? "").trim().toLowerCase());
}

function looksLikeSampler(value) {
    return SAMPLER_HINTS.has(String(value ?? "").trim().toLowerCase());
}

function clampPasses(value) {
    const n = Math.round(Number(value));
    if (!Number.isFinite(n) || n < 1) return 1;
    return Math.min(9999, n);
}

function migrateRefineWidgetOrder(node) {
    const samplerW = widgetByName(node, "sampler");
    const passesW = widgetByName(node, "passes");
    const methodW = widgetByName(node, "upscale_method");
    if (samplerW && !looksLikeSampler(widgetValue(samplerW))) {
        samplerW.value = "euler";
    }
    if (passesW) {
        passesW.value = clampPasses(widgetValue(passesW));
    }
    const tilesW = widgetByName(node, "n_tiles");
    if (tilesW) {
        const n = Math.round(Number(widgetValue(tilesW)));
        tilesW.value = Number.isFinite(n) ? Math.min(8, Math.max(1, n)) : 2;
    }
    if (methodW && !looksLikeUpscaleMethod(widgetValue(methodW))) {
        methodW.value = "h3_latent";
    }
}

function migrateRefineWidgets(node) {
    migrateRefineWidgetOrder(node);
    const seedW = widgetByName(node, "seed_mode");
    const aspectW = widgetByName(node, "aspect_ratio");
    const mpW = widgetByName(node, "megapixels");
    const widthW = widgetByName(node, "width");
    const heightW = widgetByName(node, "height");
    if (seedW && !SEED_MODE_VALUES.has(String(widgetValue(seedW) ?? "").trim().toLowerCase())) {
        seedW.value = "inherit";
    }
    if (aspectW && !ASPECT_CHOICES.has(widgetValue(aspectW))) {
        aspectW.value = FOLLOW_DIRECTOR_ASPECT;
    }
    if (mpW) {
        const n = Number(widgetValue(mpW));
        if (!Number.isFinite(n) || n < 0.1 || n > 16) mpW.value = 1.0;
    }
    if (widthW) {
        const n = Number(widgetValue(widthW));
        if (!Number.isFinite(n) || n < 32 || n > 8192) widthW.value = 1280;
    }
    if (heightW) {
        const n = Number(widgetValue(heightW));
        if (!Number.isFinite(n) || n < 32 || n > 8192) heightW.value = 720;
    }
    setWidgetVisible(node, "schedule", false);
    setWidgetVisible(node, "denoise", false);
    setWidgetVisible(node, "steps", false);
    setWidgetVisible(node, "sigmas_text", false);
    setWidgetVisible(node, "sigmas", false);
    setWidgetVisible(node, "h3_latent_model", false);
    setWidgetVisible(node, "upscale_model", false);
    setWidgetVisible(node, "confirm_first_pass", false);
    setWidgetVisible(node, "first_pass_cache_status", false);
}

function isFollowAspect(value) {
    const v = String(value ?? "").trim();
    if (v === "0" || v === "0.0") return true;
    return !v || v === FOLLOW_DIRECTOR_ASPECT || v === "Follow Director";
}

function setAspectProgrammatic(node, value) {
    const aspectW = widgetByName(node, "aspect_ratio");
    if (!aspectW || widgetValue(aspectW) === value) return;
    node._mmxAspectProgrammatic = true;
    try {
        aspectW.value = value;
        aspectW.callback?.(value);
    } finally {
        node._mmxAspectProgrammatic = false;
    }
}

function syncFollowDirectorAspect(node) {
    if (!isRefineNode(node) || node._mmxAspectUserSet) return;
    const aspectW = widgetByName(node, "aspect_ratio");
    if (!aspectW || isCustomAspect(widgetValue(aspectW))) return;
    setAspectProgrammatic(node, FOLLOW_DIRECTOR_ASPECT);
}

function readMode(node) {
    const named = widgetByName(node, "mode");
    const raw = String(widgetValue(named) ?? "").toLowerCase();
    if (raw.includes("latent_upscale") || raw.includes("latent")) return "latent_upscale";
    if (raw.includes("upscale")) return "upscale";
    if (raw.includes("refine")) return "refine";
    for (const w of node.widgets || []) {
        const s = String(widgetValue(w) ?? "").toLowerCase();
        if (s === "latent_upscale") return "latent_upscale";
        if (s === "upscale") return "upscale";
        if (s === "refine") return "refine";
    }
    return null;
}

function syncRefineComputedSize(node) {
    const aspectW = widgetByName(node, "aspect_ratio");
    const mpW = widgetByName(node, "megapixels");
    const widthW = widgetByName(node, "width");
    const heightW = widgetByName(node, "height");
    if (!aspectW || isFollowAspect(widgetValue(aspectW)) || isCustomAspect(widgetValue(aspectW))) return;
    const resolved = resolutionFromSelector(widgetValue(aspectW), widgetValue(mpW) ?? 1.0);
    if (!resolved) return;
    if (widthW) widthW.value = resolved.width;
    if (heightW) heightW.value = resolved.height;
}

function readUpscaleMethod(node) {
    return String(widgetValue(widgetByName(node, "upscale_method")) ?? "").trim().toLowerCase();
}

function isTruthyFlag(value) {
    if (value === true || value === 1) return true;
    if (value === false || value === 0 || value == null) return false;
    const text = String(value).trim().toLowerCase();
    return text === "true" || text === "yes" || text === "on";
}

function boolWidgetValue(node, name) {
    return isTruthyFlag(widgetValue(widgetByName(node, name)));
}

function syncRefineWidgetVisibility(node) {
    const mode = readMode(node);
    const upscale = mode === "upscale";
    const latentOnly = mode === "latent_upscale";
    const needsCanvas = upscale || latentOnly;
    const aspect = widgetValue(widgetByName(node, "aspect_ratio"));
    const follow = isFollowAspect(aspect);
    const custom = isCustomAspect(aspect);
    setWidgetVisible(node, "aspect_ratio", needsCanvas);
    setWidgetVisible(node, "megapixels", needsCanvas && !custom);
    setWidgetVisible(node, "width", needsCanvas && custom);
    setWidgetVisible(node, "height", needsCanvas && custom);
    const method = readUpscaleMethod(node);
    const showH3Model = latentOnly || (upscale && method === "h3_latent");
    setWidgetVisible(node, "upscale_method", upscale);
    setWidgetVisible(node, "latent_upscale_model", showH3Model);
    setWidgetVisible(node, "h3_latent_model", false);
    setWidgetVisible(node, "upscale_model", false);
    setWidgetVisible(node, "schedule", false);
    setWidgetVisible(node, "denoise", false);
    setWidgetVisible(node, "steps", false);
    setWidgetVisible(node, "sigmas_text", false);
    setWidgetVisible(node, "sigmas", false);
    setWidgetVisible(node, "sampler", !latentOnly);
    setWidgetVisible(node, "passes", !latentOnly);
    setWidgetVisible(node, "seed_mode", !latentOnly);
    const nTiles = Math.max(1, Math.round(Number(widgetValue(widgetByName(node, "n_tiles"))) || 2));
    const tiled = !latentOnly && nTiles > 1;
    const seamOn = tiled && boolWidgetValue(node, "refine_seams");
    setWidgetVisible(node, "n_tiles", !latentOnly);
    setWidgetVisible(node, "tile_axis", tiled);
    setWidgetVisible(node, "tile_overlap", tiled);
    setWidgetVisible(node, "max_size_for_no_tile", tiled);
    setWidgetVisible(node, "refine_seams", tiled);
    setWidgetVisible(node, "refine_steps", seamOn);
    setWidgetVisible(node, "target_width", false);
    setWidgetVisible(node, "target_height", false);
    setWidgetVisible(node, "confirm_first_pass", false);
    setWidgetVisible(node, "first_pass_cache_status", false);
    if (needsCanvas && !follow && !custom) syncRefineComputedSize(node);
    try {
        const size = node.computeSize?.();
        if (Array.isArray(size) && size.length >= 2) {
            node.setSize?.([node.size?.[0] || size[0], size[1]]);
        }
    } catch {
        /* ignore */
    }
    node.setDirtyCanvas?.(true, true);
}

function hookWidget(node, name, fn) {
    if (!node._mmxRefineHooked) node._mmxRefineHooked = new Set();
    if (node._mmxRefineHooked.has(name)) return;
    const w = widgetByName(node, name);
    if (!w) return;
    node._mmxRefineHooked.add(name);
    const prev = w.callback;
    w.callback = function (...args) {
        const r = prev?.apply(this, args);
        fn();
        return r;
    };
}

function installRefineResolutionUI(node) {
    const onAspect = () => {
        const aspectW = widgetByName(node, "aspect_ratio");
        const widthW = widgetByName(node, "width");
        const heightW = widgetByName(node, "height");
        if (aspectW && isCustomAspect(widgetValue(aspectW)) && widthW && heightW) {
            widthW.value = snapResolutionDim(widgetValue(widthW) || 1280);
            heightW.value = snapResolutionDim(widgetValue(heightW) || 720);
        }
        syncRefineWidgetVisibility(node);
    };
    hookWidget(node, "mode", () => syncRefineWidgetVisibility(node));
    hookWidget(node, "upscale_method", () => syncRefineWidgetVisibility(node));
    hookWidget(node, "n_tiles", () => syncRefineWidgetVisibility(node));
    hookWidget(node, "refine_seams", () => syncRefineWidgetVisibility(node));
    hookWidget(node, "aspect_ratio", () => {
        if (!node._mmxAspectProgrammatic) node._mmxAspectUserSet = true;
        onAspect();
    });
    hookWidget(node, "megapixels", () => syncRefineComputedSize(node));
    hookWidget(node, "width", () => {
        const w = widgetByName(node, "width");
        if (w) w.value = snapResolutionDim(widgetValue(w));
    });
    hookWidget(node, "height", () => {
        const w = widgetByName(node, "height");
        if (w) w.value = snapResolutionDim(widgetValue(w));
    });
    if (!node._mmxRefineOnWidgetChanged) {
        node._mmxRefineOnWidgetChanged = true;
        const prev = node.onWidgetChanged;
        node.onWidgetChanged = function (name, ...rest) {
            const r = prev?.apply(this, [name, ...rest]);
            if (
                name === "mode"
                || name === "upscale_method"
                || name === "aspect_ratio"
                || name === "megapixels"
                || name === "n_tiles"
                || name === "refine_seams"
            ) {
                migrateRefineWidgets(this);
                syncRefineWidgetVisibility(this);
            }
            return r;
        };
    }
}

const DIRECTOR_SAMPLE_COMFY_WIDGETS = [
    "bd_grp_sample",
    "seed",
    "control_after_generate",
    "control after generate",
    "bd_grp_advanced",
    "steps",
    "sampler",
    "scheduler",
    "shift_video",
    "shift_audio",
    "live_tae_vae",
    "bd_grp_perf",
    "clear_vram_between_segments",
];

const DIRECTOR_REFINE_COMFY_WIDGETS = [
    "bd_grp_refine",
    "refine_enable",
    "refine_mode",
    "refine_upscale_method",
    "refine_latent_upscale_model",
    "refine_sampler",
    "refine_passes",
    "refine_sample_steps",
    "refine_scheduler",
    "refine_denoise",
    "refine_extra_steps",
    "refine_start_at_sigma",
    "refine_end_at_sigma",
    "refine_spacing",
    "refine_seed_mode",
    "refine_aspect_ratio",
    "refine_megapixels",
    "refine_width",
    "refine_height",
    "refine_skip_fl2v",
    "refine_tile",
    "refine_n_tiles",
    "refine_tile_axis",
    "refine_tile_overlap",
    "refine_max_size_for_no_tile",
    "refine_seams",
    "refine_seam_steps",
    "refine_model",
    "refine_model_r2v",
    "upscale_model",
    "selflift_model_hires",
    "bd_grp_selflift",
    "selflift_enable",
    "selflift_split_mode",
    "selflift_highres_steps",
    "selflift_transition_step",
    "selflift_lowres_scale",
    "selflift_latent_upscale_model",
    "selflift_native_low_carry",
    "selflift_sampler_mode",
    "selflift_rho",
    "selflift_w_min",
    "selflift_w_max",
    "selflift_latent_upsample",
    "selflift_enable_latent_chunking",
    "selflift_enable_tiling",
    "selflift_tile_count",
    "selflift_tile_overlap",
    "bd_grp_semantic_bridge",
    "semantic_bridge_enable",
    "semantic_bridge_adapter",
    "semantic_bridge_alpha",
    "semantic_bridge_magnitude_match",
    "bd_grp_face_refine",
    "face_refine_enable",
    "face_refine_detector",
    "face_refine_confidence",
    "face_refine_crop_factor",
    "face_refine_canvas_width",
    "face_refine_canvas_height",
    "face_refine_canvas_mode",
    "face_refine_select",
    "face_refine_denoise",
    "face_refine_steps",
    "face_refine_sampler",
    "face_refine_scheduler",
    "face_refine_seed_mode",
    "face_refine_paste_region",
    "face_refine_mask_dilation",
    "face_refine_feather",
    "face_refine_colour_match",
    "face_refine_blend",
    "face_refine_follow_director",
    "face_refine_sigmas",
    "clear_vram_before_face_refine",
    "clear_vram_before_refine",
    "export_pre_face_refine",
];

function directorHasNamedLink(node, name) {
    const inp = (node?.inputs || []).find((item) => String(item?.name) === name);
    if (!inp) return false;
    if (inp.link != null) return true;
    return Array.isArray(inp.links) && inp.links.length > 0;
}

function directorHasRefineLink(node) {
    return directorHasNamedLink(node, "refine");
}

function readDirectorRefineMode(node) {
    const raw = String(widgetValue(widgetByName(node, "refine_mode")) ?? "").toLowerCase();
    if (raw.includes("latent_upscale") || raw.includes("latent")) return "latent_upscale";
    if (raw.includes("upscale")) return "upscale";
    if (raw.includes("refine")) return "refine";
    return "refine";
}

const DIRECTOR_HIDDEN_INPUT_SLOTS = [
    "upscale_model",
    "selflift_model_hires",
    "face_refine_sigmas",
];

function stripHiddenDirectorInputDefs(nodeData) {
    if (!nodeData) return;
    for (const group of [nodeData.input?.optional, nodeData.input?.required, nodeData.inputs]) {
        if (!group || typeof group !== "object") continue;
        for (const name of DIRECTOR_HIDDEN_INPUT_SLOTS) delete group[name];
    }
}

function hideDirectorInputSlots(node) {
    if (!node?.inputs) return;
    for (const name of DIRECTOR_HIDDEN_INPUT_SLOTS) {
        for (;;) {
            const idx = node.inputs.findIndex((item) => String(item?.name) === name);
            if (idx < 0) break;
            if (typeof node.removeInput === "function") node.removeInput(idx);
            else node.inputs.splice(idx, 1);
        }
    }
}

function hideDirectorRefineComfyWidgets(node) {
    if (!node) return;
    hideDirectorInputSlots(node);
    for (const name of DIRECTOR_SAMPLE_COMFY_WIDGETS) {
        setWidgetVisible(node, name, false);
    }
    for (const name of DIRECTOR_REFINE_COMFY_WIDGETS) {
        setWidgetVisible(node, name, false);
    }
    const seed = widgetByName(node, "seed");
    for (const linked of seed?.linkedWidgets || []) {
        if (linked?.name) setWidgetVisible(node, linked.name, false);
        else {
            linked.hidden = true;
            if (!linked.options) linked.options = {};
            linked.options.hidden = true;
            linked.computeSize = () => [0, -4];
        }
    }
}

function controlAfterWidget(node) {
    return widgetByName(node, "control_after_generate")
        || widgetByName(node, "control after generate");
}

export function closePassPanels(editor, except) {
    if (except !== "sample") {
        editor._mmxSamplePanelOpen = false;
        editor.samplePanelEl?.classList.add("hidden");
        editor.sampleCfgBtn?.classList.remove("active");
    }
    if (except !== "refine") {
        editor._mmxRefinePanelOpen = false;
        editor.refinePanelEl?.classList.add("hidden");
    }
    if (except !== "selflift") {
        editor._mmxSelfLiftPanelOpen = false;
        editor.selfLiftPanelEl?.classList.add("hidden");
    }
    if (except !== "face") {
        editor._mmxFaceRefinePanelOpen = false;
        editor.faceRefinePanelEl?.classList.add("hidden");
    }
    if (except !== "semantic") {
        editor._mmxSemanticBridgePanelOpen = false;
        editor.semanticBridgePanelEl?.classList.add("hidden");
    }
    if (except !== "preview") {
        editor._mmxPreviewPanelOpen = false;
        editor.previewPanelEl?.classList.add("hidden");
    }
    if (except !== "continuity") {
        editor._mmxContinuityPanelOpen = false;
        editor.segmentContinuityPanelEl?.classList.add("hidden");
        if (editor.segmentContinuityPanelEl) editor.segmentContinuityPanelEl.hidden = true;
    }
    syncFirstPassButtons(editor);
}

function syncFirstPassButtons(editor) {
    const selfLiftOn = isTruthyFlag(editor?.node?.widgets?.find((item) => item.name === "selflift_enable")?.value);
    const selfLiftButton = editor?.selfLiftBarEl?.querySelector("button");
    const sampleButton = editor?.sampleCfgBtn;
    selfLiftButton?.classList.toggle("active", selfLiftOn);
    // SelfLift is an alternate first-pass path. When it is off, the normal
    // first-pass tab remains the active mode even if its drawer is collapsed.
    sampleButton?.classList.toggle("active", !selfLiftOn);
}

function normModelName(name) {
    return String(name || "").replaceAll("\\", "/").trim();
}

function fillModelSelect(select, values, current) {
    if (!select) return;
    const cur = normModelName(current);
    const items = [];
    const seen = new Set();
    for (const raw of values || []) {
        const value = normModelName(raw);
        if (!value || seen.has(value)) continue;
        seen.add(value);
        items.push(value);
    }
    if (cur && !seen.has(cur)) items.unshift(cur);
    select.innerHTML = optionHtml(items, cur || items[0] || "");
    if (cur) select.value = cur;
}

let _drawerModelsPromise = null;

function loadDrawerModels(force = false) {
    if (force) _drawerModelsPromise = null;
    if (!_drawerModelsPromise) {
        _drawerModelsPromise = api.fetchApi("/minimax/director/list_drawer_models")
            .then(async (resp) => {
                const data = resp.ok ? await resp.json() : {};
                return {
                    detectors: Array.isArray(data.detectors) ? data.detectors.map(normModelName).filter(Boolean) : [],
                    latent_upscale_models: Array.isArray(data.latent_upscale_models) ? data.latent_upscale_models.map(normModelName).filter(Boolean) : [],
                    semantic_bridge_adapters: Array.isArray(data.semantic_bridge_adapters) ? data.semantic_bridge_adapters.map(normModelName).filter(Boolean) : [],
                };
            })
            .catch(() => ({ detectors: [], latent_upscale_models: [], semantic_bridge_adapters: [] }));
    }
    return _drawerModelsPromise;
}

function applyDrawerModelsToEditor(editor, data) {
    const node = editor?.node;
    if (!node || !data) return;
    const apply = (widgetName, values, select) => {
        const widget = node.widgets?.find((x) => x.name === widgetName);
        if (widget) {
            if (!widget.options) widget.options = {};
            if (values.length) widget.options.values = values;
            if (widget.value) widget.value = normModelName(widget.value);
        }
        if (select) fillModelSelect(select, values, widget?.value);
    };
    apply("selflift_latent_upscale_model", data.latent_upscale_models, editor.selfLiftPanelEl?.querySelector(`[data-w="selflift_latent_upscale_model"]`));
    apply("semantic_bridge_adapter", data.semantic_bridge_adapters || [], editor.semanticBridgePanelEl?.querySelector(`[data-w="semantic_bridge_adapter"]`));
    apply("face_refine_detector", data.detectors, editor.faceRefinePanelEl?.querySelector(`[data-w="face_refine_detector"]`));
    apply("refine_latent_upscale_model", data.latent_upscale_models, editor.refinePanelEl?.querySelector(`[data-w="refine_latent_upscale_model"]`));
}

function refreshDrawerModelSelects(editor, force = false) {
    return loadDrawerModels(force).then((data) => applyDrawerModelsToEditor(editor, data));
}

export function mountDirectorSelfLiftPanel(editor) {
    const node = editor?.node;
    const bar = editor?.outputBarEl;
    if (!node || !bar || editor._mmxSelfLiftPanelMounted) return;
    editor._mmxSelfLiftPanelMounted = true;
    const wrap = document.createElement("span");
    wrap.className = "bd-out-refine-wrap";
    wrap.innerHTML = `<button type="button" class="bd-btn" data-r="selflift-cfg">SelfLift</button>`;
    const tools = bar.querySelector(".bd-live-preview-tools");
    if (tools) bar.insertBefore(wrap, tools); else bar.appendChild(wrap);
    const panel = document.createElement("div");
    panel.className = "bd-refine-panel hidden";
    panel.setAttribute("data-r", "selflift-panel");
    panel.innerHTML = [
        `<div class="bd-refine-group">渐进采样</div>`,
        `<label class="bd-refine-field row"><input type="checkbox" data-w="selflift_enable"><span>启用 SelfLift</span></label>`,
        refinePanelFieldHtml("split", "SelfLift split", "分段方式", `<select data-w="selflift_split_mode"><option value="highres_steps">按高清步数</option><option value="transition_step">按转场步数</option></select>`),
        refinePanelFieldHtml("high", "SelfLift high steps", "高清步数", `<input type="number" data-w="selflift_highres_steps" min="1" max="64" step="1">`),
        refinePanelFieldHtml("trans", "SelfLift transition", "转场步数", `<input type="number" data-w="selflift_transition_step" min="1" max="200" step="1">`),
        refinePanelFieldHtml("scale", "SelfLift low scale", "低清倍率", `<input type="number" data-w="selflift_lowres_scale" min="0.25" max="1" step="0.05">`),
        refinePanelFieldHtml("samp", "SelfLift sampler", "采样器", `<select data-w="selflift_sampler_mode"><option value="euler">Euler</option><option value="follow_director">跟随导演台</option></select>`),
        `<label class="bd-refine-field row"><input type="checkbox" data-w="selflift_native_low_carry"><span>跨段低清承接</span></label>`,
        `<div class="bd-refine-group">提升 / 3D</div>`,
        refinePanelFieldHtml("model", "SelfLift model", "3D 权重", `<select data-w="selflift_latent_upscale_model"></select>`),
        refinePanelFieldHtml("up", "SelfLift upsample", "插值方式", `<select data-w="selflift_latent_upsample"><option value="bilinear">双线性</option><option value="nearest">最近邻</option></select>`),
        refinePanelFieldHtml("rho", "SelfLift rho", "像素锚混合 rho", `<input type="number" data-w="selflift_rho" min="0" max="1" step="0.05">`),
        refinePanelFieldHtml("wmin", "SelfLift w_min", "低频权重 w_min", `<input type="number" data-w="selflift_w_min" min="0" max="1" step="0.05">`),
        refinePanelFieldHtml("wmax", "SelfLift w_max", "高频权重 w_max", `<input type="number" data-w="selflift_w_max" min="0" max="1" step="0.05">`),
        `<label class="bd-refine-field row"><input type="checkbox" data-w="selflift_enable_latent_chunking"><span>3D 时间分块</span></label>`,
        `<div class="bd-refine-group">高清分块</div>`,
        `<label class="bd-refine-field row"><input type="checkbox" data-w="selflift_enable_tiling"><span>启用高清分块</span></label>`,
        refinePanelFieldHtml("tiles", "SelfLift tiles", "分块数", `<input type="number" data-w="selflift_tile_count" min="1" max="8" step="1">`),
        refinePanelFieldHtml("overlap", "SelfLift overlap", "重叠像素", `<input type="number" data-w="selflift_tile_overlap" min="0" max="2048" step="64">`),
    ].join("");
    applyFieldTooltips(panel, {
        selflift_enable: "启用 SelfLift：低清首采、3D latent 提升，再进行高清收尾。",
        split: "选择按高清步数切分，或按论文中的 transition step 切分低清/高清阶段。",
        high: "高清收尾阶段使用的步数；仅 highres_steps 模式生效。",
        trans: "低清阶段结束并转入高清阶段的步数 k；仅 transition_step 模式生效。",
        scale: "低清首采画布相对导演台画布的倍率；越低越省显存，但细节更依赖提升阶段。",
        samp: "SelfLift 的采样器。Euler 是论文路径；跟随导演台则使用一采采样器（必须也是 Euler）。",
        selflift_native_low_carry: "跨段传递上一段的 native 低清尾部，帮助连续镜头减少闪烁和接缝。",
        model: "SelfLift 3D latent 提升使用的权重文件。",
        up: "低清 cond 与提升过渡态的插值方式。",
        rho: "像素锚混合比例；0 表示只使用 3D latent 提升。",
        wmin: "rho 大于 0 时，低频区域的像素锚权重。",
        wmax: "rho 大于 0 时，高频残差区域的像素锚权重。",
        selflift_enable_latent_chunking: "将 3D latent 提升按时间分块以降低显存；可能改变段间一致性。",
        selflift_enable_tiling: "仅对高清收尾启用空间分块；低清阶段不会分块。",
        tiles: "高清收尾的空间分块数量。",
        overlap: "高清空间块之间的重叠像素，用于减轻拼接接缝。",
    });
    bar.after(panel); editor.selfLiftBarEl = wrap; editor.selfLiftPanelEl = panel;
    const write = (name, value) => { const src = node; const w = src?.widgets?.find((x) => x.name === name); if (!w) return; w.value = value; w.callback?.(value); src?.setDirtyCanvas?.(true, true); };
    const sync = () => {
        const src = node;
        panel.classList.remove("linked");
        const modelSelect = panel.querySelector(`[data-w="selflift_latent_upscale_model"]`);
        if (modelSelect && src) {
            const modelWidget = src.widgets?.find((x) => x.name === "selflift_latent_upscale_model");
            const current = String(modelWidget?.value ?? "");
            const values = modelWidget?.options?.values?.length
                ? modelWidget.options.values
                : (current ? [current] : [""]);
            fillModelSelect(modelSelect, values, current);
        }
        for (const el of panel.querySelectorAll("[data-w]")) { const w = src?.widgets?.find((x) => x.name === el.getAttribute("data-w")); if (!w) continue; if (el.type === "checkbox") el.checked = isTruthyFlag(w.value); else if (w.value != null) el.value = w.value; }
        const split = String(panel.querySelector(`[data-w="selflift_split_mode"]`)?.value || "highres_steps");
        const rho = Number(panel.querySelector(`[data-w="selflift_rho"]`)?.value || 0);
        const tiling = !!panel.querySelector(`[data-w="selflift_enable_tiling"]`)?.checked;
        const setShow = (id, on) => { const el = panel.querySelector(`[data-show="${id}"]`); if (el) el.classList.toggle("hidden", !on); };
        setShow("high", split !== "transition_step");
        setShow("trans", split === "transition_step");
        setShow("wmin", rho > 1e-8);
        setShow("wmax", rho > 1e-8);
        setShow("tiles", tiling);
        setShow("overlap", tiling);
        syncFirstPassButtons(editor);
    };
    wrap.querySelector("[data-r=selflift-cfg]").addEventListener("click", () => {
        const next = !editor._mmxSelfLiftPanelOpen;
        closePassPanels(editor, next ? "selflift" : "");
        editor._mmxSelfLiftPanelOpen = next;
        panel.classList.toggle("hidden", !next);
        const after = () => { sync(); editor.resizeNodeForContentMinChange?.(); };
        if (next) refreshDrawerModelSelects(editor).finally(after);
        else after();
    });
    refreshDrawerModelSelects(editor).finally(sync);
    panel.addEventListener("change", (e) => {
        const el = e.target?.closest?.("[data-w]");
        if (!el) return;
        write(el.getAttribute("data-w"), el.type === "checkbox" ? String(!!el.checked) : el.type === "number" ? Number(el.value) : el.value);
        sync();
        if (editor._mmxSelfLiftPanelOpen) editor.resizeNodeForContentMinChange?.();
    });
    sync();
}

export function mountDirectorSemanticBridgePanel(editor) {
    const node = editor?.node;
    const bar = editor?.outputBarEl;
    if (!node || !bar || editor._mmxSemanticBridgePanelMounted) return;
    editor._mmxSemanticBridgePanelMounted = true;
    const wrap = document.createElement("span");
    wrap.className = "bd-out-refine-wrap";
    wrap.innerHTML = `<button type="button" class="bd-btn" data-r="semantic-bridge-cfg">Semantic Bridge</button>`;
    const tools = bar.querySelector(".bd-live-preview-tools");
    if (tools) bar.insertBefore(wrap, tools); else bar.appendChild(wrap);
    const panel = document.createElement("div");
    panel.className = "bd-refine-panel hidden";
    panel.setAttribute("data-r", "semantic-bridge-panel");
    panel.innerHTML = [
        `<div class="bd-refine-group">Semantic Bridge 语义增强</div>`,
        `<label class="bd-refine-field row"><input type="checkbox" data-w="semantic_bridge_enable"><span>启用 Semantic Bridge</span></label>`,
        refinePanelFieldHtml("adapter", "Semantic Bridge adapter", "权重文件", `<select data-w="semantic_bridge_adapter"></select>`),
        refinePanelFieldHtml("alpha", "Semantic Bridge alpha", "混合强度 alpha", `<input type="number" data-w="semantic_bridge_alpha" min="0" max="1" step="0.01">`),
        `<label class="bd-refine-field row"><input type="checkbox" data-w="semantic_bridge_magnitude_match"><span>匹配向量模长</span></label>`,
        `<div class="bd-refine-help">仅改写一采 conditioning token；未启用或未找到权重时保持原流程。</div>`,
    ].join("");
    applyFieldTooltips(panel, {
        semantic_bridge_enable: "启用后对一采 conditioning token 应用 Semantic Bridge student MLP。",
        adapter: "放在 ComfyUI/models/semantic_bridge/ 下的 .safetensors/.pt/.pth 权重。",
        alpha: "残差混合强度，默认 0.15。",
        semantic_bridge_magnitude_match: "将 student 输出的向量模长对齐到原 hidden。",
    });
    bar.after(panel); editor.semanticBridgeBarEl = wrap; editor.semanticBridgePanelEl = panel;
    const sync = () => {
        const select = panel.querySelector(`[data-w="semantic_bridge_adapter"]`);
        const w = node.widgets?.find((x) => x.name === "semantic_bridge_adapter");
        const values = w?.options?.values?.length ? w.options.values : (w?.value ? [w.value] : []);
        if (select) fillModelSelect(select, values, w?.value);
        for (const el of panel.querySelectorAll("[data-w]")) {
            const item = node.widgets?.find((x) => x.name === el.getAttribute("data-w"));
            if (!item) continue;
            if (el.type === "checkbox") el.checked = isTruthyFlag(item.value); else if (item.value != null) el.value = item.value;
        }
        wrap.querySelector("button")?.classList.toggle("active", isTruthyFlag(node.widgets?.find((x) => x.name === "semantic_bridge_enable")?.value));
    };
    editor.syncSemanticBridgePanelFromWidgets = sync;
    wrap.querySelector("[data-r=semantic-bridge-cfg]").addEventListener("click", () => {
        const next = !editor._mmxSemanticBridgePanelOpen;
        closePassPanels(editor, next ? "semantic" : "");
        editor._mmxSemanticBridgePanelOpen = next; panel.classList.toggle("hidden", !next);
        if (next) refreshDrawerModelSelects(editor).finally(() => { sync(); editor.resizeNodeForContentMinChange?.(); }); else sync();
    });
    refreshDrawerModelSelects(editor).finally(sync);
    panel.addEventListener("change", (event) => {
        const el = event.target?.closest?.("[data-w]"); if (!el) return;
        const item = node.widgets?.find((x) => x.name === el.getAttribute("data-w")); if (!item) return;
        item.value = el.type === "checkbox" ? String(!!el.checked) : el.type === "number" ? Number(el.value) : el.value;
        item.callback?.(item.value); node.setDirtyCanvas?.(true, true); sync();
    });
}

export function mountDirectorFaceRefinePanel(editor) {
    const node = editor?.node;
    const bar = editor?.outputBarEl;
    if (!node || !bar || editor._mmxFaceRefinePanelMounted) return;
    editor._mmxFaceRefinePanelMounted = true;
    const wrap = document.createElement("span");
    wrap.className = "bd-out-refine-wrap";
    wrap.innerHTML = `<button type="button" class="bd-btn" data-r="face-refine-cfg">FaceRefine</button>`;
    const tools = bar.querySelector(".bd-live-preview-tools");
    if (tools) bar.insertBefore(wrap, tools); else bar.appendChild(wrap);
    const faceSamplerVals = comboValues(node, "face_refine_sampler", ["euler"]);
    const faceSchedulerVals = comboValues(node, "face_refine_scheduler", ["simple"]);
    const panel = document.createElement("div");
    panel.className = "bd-refine-panel hidden";
    panel.setAttribute("data-r", "face-refine-panel");
    panel.innerHTML = [
        `<div class="bd-refine-group">开关</div>`,
        `<label class="bd-refine-field row"><input type="checkbox" data-w="face_refine_enable"><span>启用 FaceRefine</span></label>`,
        `<label class="bd-refine-field row"><input type="checkbox" data-w="clear_vram_before_face_refine"><span>FaceRefine 前清理显存</span></label>`,
        `<label class="bd-refine-field row"><input type="checkbox" data-w="clear_vram_before_refine"><span>二采前清理显存</span></label>`,
        `<label class="bd-refine-field row"><input type="checkbox" data-w="export_pre_face_refine"><span>输出修脸前画面</span></label>`,
        `<div class="bd-refine-group">人脸检测</div>`,
        refinePanelFieldHtml("detector", "FaceRefine detector", "检测模型", `<select data-w="face_refine_detector"></select>`),
        refinePanelFieldHtml("confidence", "FaceRefine confidence", "置信度", `<input type="number" data-w="face_refine_confidence" min="0.05" max="0.95" step="0.01">`),
        refinePanelFieldHtml("crop", "FaceRefine crop", "裁剪倍率", `<input type="number" data-w="face_refine_crop_factor" min="1.2" max="8" step="0.1">`),
        refinePanelFieldHtml("canvas", "FaceRefine canvas", "画布模式", `<select data-w="face_refine_canvas_mode"><option value="manual">手动</option><option value="auto_capped_768">自动（上限 768）</option></select>`),
        refinePanelFieldHtml("canvas-width", "FaceRefine canvas width", "画布宽度", `<input type="number" data-w="face_refine_canvas_width" min="128" max="1344" step="32">`),
        refinePanelFieldHtml("canvas-height", "FaceRefine canvas height", "画布高度", `<input type="number" data-w="face_refine_canvas_height" min="128" max="1344" step="32">`),
        refinePanelFieldHtml("select", "FaceRefine select", "目标选择", `<select data-w="face_refine_select"><option value="largest_face">最大脸</option><option value="centre_most">最靠近中心</option></select>`),
        `<div class="bd-refine-group">采样设置</div>`,
        `<label class="bd-refine-field row"><input type="checkbox" data-w="face_refine_follow_director"><span>跟随导演台</span></label>`,
        refinePanelFieldHtml("denoise", "FaceRefine denoise", "去噪强度", `<input type="number" data-w="face_refine_denoise" min="0.02" max="1" step="0.01">`),
        refinePanelFieldHtml("steps", "FaceRefine steps", "步数", `<input type="number" data-w="face_refine_steps" min="1" max="50" step="1">`),
        refinePanelFieldHtml("sampler", "FaceRefine sampler", "采样器", `<select data-w="face_refine_sampler">${optionHtml(faceSamplerVals, widgetValue(widgetByName(node, "face_refine_sampler")))}</select>`),
        refinePanelFieldHtml("scheduler", "FaceRefine scheduler", "调度器", `<select data-w="face_refine_scheduler">${optionHtml(faceSchedulerVals, widgetValue(widgetByName(node, "face_refine_scheduler")))}</select>`),
        refinePanelFieldHtml("seed", "FaceRefine seed", "种子模式", `<select data-w="face_refine_seed_mode"><option value="inherit">继承导演台</option><option value="offset">按段偏移</option></select>`),
        `<div class="bd-refine-group">贴回设置</div>`,
        refinePanelFieldHtml("paste", "FaceRefine paste", "贴回区域", `<select data-w="face_refine_paste_region"><option value="face_only">仅脸部</option><option value="face_ellipse">脸部椭圆</option><option value="full_crop">完整裁剪</option></select>`),
        refinePanelFieldHtml("dilation", "FaceRefine dilation", "遮罩扩张", `<input type="number" data-w="face_refine_mask_dilation" min="0" max="256" step="2">`),
        refinePanelFieldHtml("feather", "FaceRefine feather", "羽化", `<input type="number" data-w="face_refine_feather" min="0" max="256" step="2">`),
        refinePanelFieldHtml("colour", "FaceRefine colour", "色彩匹配", `<input type="number" data-w="face_refine_colour_match" min="0" max="1" step="0.05">`),
        refinePanelFieldHtml("blend", "FaceRefine blend", "混合强度", `<input type="number" data-w="face_refine_blend" min="0" max="1" step="0.05">`),
    ].join("");
    applyFieldTooltips(panel, {
        face_refine_enable: "启用 FaceRefine：检测人脸、局部重采并贴回最终画面。",
        clear_vram_before_face_refine: "在 FaceRefine 开始前卸载采样模型并清理显存，降低峰值显存。",
        clear_vram_before_refine: "在一采完成、二采开始前卸载模型并清理显存。",
        export_pre_face_refine: "额外输出贴回修脸前的画面，便于与最终结果对比。",
        detector: "人脸检测模型；需要放在 ComfyUI/models/ultralytics/ 或其子目录。",
        confidence: "人脸检测置信度阈值；降低可抓到更小或更侧的脸，但误检概率会上升。",
        crop: "人脸裁剪范围相对脸高的倍率；更大能保留更多头发和表情上下文。",
        canvas: "FaceRefine 裁剪画布模式；手动使用指定宽高，自动则按裁剪适配并限制上限。",
        "canvas-width": "FaceRefine 裁剪画布宽度。",
        "canvas-height": "FaceRefine 裁剪画布高度。",
        select: "选择跟踪目标：最大脸，或最靠近画面中心的脸。",
        denoise: "FaceRefine 局部重采去噪强度；越高变化越明显。",
        steps: "FaceRefine 局部重采步数；更高通常更细致但更慢。跟随导演台时使用一采步数。",
        sampler: "FaceRefine 局部重采使用的采样算法。跟随导演台时使用一采采样器。",
        scheduler: "FaceRefine 局部重采使用的 sigma 调度曲线。跟随导演台时使用一采调度器。",
        seed: "FaceRefine 种子模式：继承导演台种子，或按段偏移。",
        paste: "将修复结果贴回原画面的区域；仅脸部最保守，完整裁剪改动范围最大。",
        face_refine_follow_director: "使用导演台一采的步数、采样器和调度器。",
        dilation: "贴回遮罩向外扩张的像素范围。",
        feather: "贴回遮罩边缘的羽化半径，减轻矩形边界。",
        colour: "修复结果与原画面的色彩匹配强度。",
        blend: "修复结果贴回原画面的混合强度。",
    });
    bar.after(panel);
    editor.faceRefineBarEl = wrap;
    editor.faceRefinePanelEl = panel;
    const sync = () => {
        const detector = panel.querySelector(`[data-w="face_refine_detector"]`);
        const detectorWidget = node.widgets?.find((x) => x.name === "face_refine_detector");
        const detectorValue = String(detectorWidget?.value || "face_yolov8m.pt");
        const detectorValues = detectorWidget?.options?.values?.length
            ? detectorWidget.options.values
            : [detectorValue];
        fillModelSelect(detector, detectorValues, detectorValue);
        for (const el of panel.querySelectorAll("[data-w]")) {
            const w = node.widgets?.find((x) => x.name === el.getAttribute("data-w"));
            if (w && el.type === "select-one" && w.options?.values?.length && el.getAttribute("data-w") !== "face_refine_detector") {
                fillModelSelect(el, w.options.values, w.value);
            }
            if (!w) continue;
            if (el.type === "checkbox") el.checked = isTruthyFlag(w.value);
            else if (w.value != null) el.value = w.value;
        }
        const followDirector = !!panel.querySelector(`[data-w="face_refine_follow_director"]`)?.checked;
        const canvasMode = String(panel.querySelector(`[data-w="face_refine_canvas_mode"]`)?.value || "manual");
        const setShow = (id, on) => { const field = panel.querySelector(`[data-show="${id}"]`); if (field) field.classList.toggle("hidden", !on); };
        setShow("steps", !followDirector);
        setShow("sampler", !followDirector);
        setShow("scheduler", !followDirector);
        setShow("canvas-width", canvasMode === "manual");
        setShow("canvas-height", canvasMode === "manual");
        wrap.querySelector("button")?.classList.toggle("active", isTruthyFlag(node.widgets?.find((x) => x.name === "face_refine_enable")?.value));
    };
    wrap.querySelector("[data-r=face-refine-cfg]").addEventListener("click", () => {
        const next = !editor._mmxFaceRefinePanelOpen;
        closePassPanels(editor, next ? "face" : "");
        editor._mmxFaceRefinePanelOpen = next;
        panel.classList.toggle("hidden", !next);
        const after = () => { sync(); editor.resizeNodeForContentMinChange?.(); };
        if (next) refreshDrawerModelSelects(editor).finally(after);
        else after();
    });
    refreshDrawerModelSelects(editor).finally(sync);
    panel.addEventListener("change", (event) => {
        const el = event.target?.closest?.("[data-w]");
        if (!el) return;
        const w = node.widgets?.find((x) => x.name === el.getAttribute("data-w"));
        if (!w) return;
        w.value = el.type === "checkbox" ? String(!!el.checked) : el.type === "number" ? Number(el.value) : el.value;
        w.callback?.(w.value);
        node.setDirtyCanvas?.(true, true);
        sync();
    });
    sync();
}


export function mountDirectorSamplePanel(editor) {
    const node = editor?.node;
    const bar = editor?.outputBarEl;
    if (!node || !bar || editor._mmxSamplePanelMounted) return;
    editor._mmxSamplePanelMounted = true;
    hideDirectorRefineComfyWidgets(node);

    const wrap = document.createElement("span");
    wrap.className = "bd-out-refine-wrap";
    wrap.innerHTML = `<button type="button" class="bd-btn" data-r="sample-cfg" data-i18n="widget.sampleConfig">${t("widget.sampleConfig")}</button>`;
    const tools = bar.querySelector(".bd-live-preview-tools");
    if (tools) bar.insertBefore(wrap, tools);
    else bar.appendChild(wrap);

    const samplerVals = comboValues(node, "sampler", ["res_multistep"]);
    const schedulerVals = comboValues(node, "scheduler", ["simple"]);
    const ctrlW = controlAfterWidget(node);
    const ctrlVals = comboValues(node, ctrlW?.name || "control_after_generate", ["fixed", "increment", "decrement", "randomize"]);

    const panel = document.createElement("div");
    panel.className = "bd-refine-panel hidden";
    panel.setAttribute("data-r", "sample-panel");
    panel.innerHTML = [
        refinePanelFieldHtml("seed", "widget.seed", "种子",
            `<input type="number" data-w="seed" min="0" max="18446744073709551615" step="1">`),
        refinePanelFieldHtml("ctrl", "widget.controlAfterGenerate", "生成前后定制",
            `<select data-w="${ctrlW?.name || "control_after_generate"}">${optionHtml(ctrlVals, widgetValue(ctrlW))}</select>`),
        refinePanelFieldHtml("steps", "widget.steps", "步数",
            `<input type="number" data-w="steps" min="1" max="200" step="1">`),
        refinePanelFieldHtml("sampler", "widget.sampler", "采样器",
            `<select data-w="sampler">${optionHtml(samplerVals, widgetValue(widgetByName(node, "sampler")))}</select>`),
        refinePanelFieldHtml("scheduler", "widget.scheduler", "调度器",
            `<select data-w="scheduler">${optionHtml(schedulerVals, widgetValue(widgetByName(node, "scheduler")))}</select>`),
        refinePanelFieldHtml("shiftv", "widget.shiftVideo", "shift video",
            `<input type="number" data-w="shift_video" min="0.01" max="100" step="0.01">`),
        refinePanelFieldHtml("shifta", "widget.shiftAudio", "shift audio",
            `<input type="number" data-w="shift_audio" min="0.01" max="100" step="0.01">`),
    ].join("");
    applyFieldTooltips(panel, {
        seed: "控制本次一采的随机起点；相同设置和种子可复现相近结果。",
        ctrl: "生成后如何处理种子：固定、递增、递减或随机化。",
        steps: "一采去噪步数；步数越高通常细节越充分，但耗时和显存占用也会增加。",
        sampler: "一采使用的采样算法，决定噪声到画面的推进方式。",
        scheduler: "一采的 sigma 调度曲线，决定各采样步的噪声分布。",
        shiftv: "视频 latent 的时间/运动 shift 参数，影响运动节奏和时序分布。",
        shifta: "音频 latent 的时间 shift 参数，影响声音与画面时间轴的配合。",
    });
    bar.after(panel);
    editor.sampleBarEl = wrap;
    editor.samplePanelEl = panel;
    editor.sampleCfgBtn = wrap.querySelector('[data-r="sample-cfg"]');
    editor.syncSamplePanelFromWidgets = () => syncSamplePanelFromWidgets(editor);

    editor.sampleCfgBtn.addEventListener("click", () => {
        const next = !editor._mmxSamplePanelOpen;
        closePassPanels(editor, next ? "sample" : "");
        editor._mmxSamplePanelOpen = next;
        panel.classList.toggle("hidden", !next);
        syncFirstPassButtons(editor);
        if (next) syncSamplePanelFromWidgets(editor);
        editor.resizeNodeForContentMinChange?.();
    });
    const onSamplePanelEdit = (e) => {
        const el = e.target?.closest?.("[data-w]");
        if (!el) return;
        const name = el.getAttribute("data-w");
        const value = el.type === "number" ? Number(el.value) : el.value;
        writeRefineWidget(node, name, value);
        if (
            name === "seed"
            || name === "steps"
            || name === "sampler"
            || name === "scheduler"
            || name === "shift_video"
            || name === "shift_audio"
        ) {
            // Keep the same snapshot that the configure/settle restore path
            // reads.  `_mmxSampleSnap` was never consumed there, so changing
            // the custom panel could be overwritten by the previous value
            // when the node was queued or reconfigured.
            node._mmxSampleWidgetSnap = node._mmxSampleWidgetSnap || {};
            node._mmxSampleWidgetSnap[name] = value;
        }
        // Custom panel writes `widget.value` without the Comfy callback that
        // `hookDirectorSampleWidgetSnapshots` uses, so the snapshot dirty
        // check never saw first-pass edits and 更新/重置 stayed dim.
        editor._snapshotSelector?.notifyChanged?.();
    };
    panel.addEventListener("change", onSamplePanelEdit);
    panel.addEventListener("input", onSamplePanelEdit);
    panel.addEventListener("keydown", (e) => e.stopPropagation());
    syncSamplePanelFromWidgets(editor);
}

function syncSamplePanelFromWidgets(editor) {
    const node = editor?.node;
    const panel = editor?.samplePanelEl;
    if (!node || !panel) return;
    hideDirectorRefineComfyWidgets(node);
    for (const el of panel.querySelectorAll("[data-w]")) {
        const name = el.getAttribute("data-w");
        const raw = widgetValue(widgetByName(node, name));
        if (raw != null && raw !== "") el.value = raw;
    }
}

function comboValues(node, name, fallback) {
    const w = widgetByName(node, name);
    const raw = w?.options?.values;
    if (!Array.isArray(raw) || !raw.length) return fallback;
    return raw.map((item) => {
        if (item && typeof item === "object") return String(item.content ?? item.value ?? "");
        return String(item);
    }).filter(Boolean);
}

function optionHtml(values, current) {
    const cur = String(current ?? "");
    return values.map((value) => {
        const safe = String(value).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
        const sel = String(value) === cur ? " selected" : "";
        return `<option value="${safe}"${sel}>${safe}</option>`;
    }).join("");
}

function writeRefineWidget(node, name, value) {
    const w = widgetByName(node, name);
    if (!w) return;
    w.value = value;
}

function syncDirectorBuiltinRefineWidgets(node) {
    if (!node || !DIRECTOR_CLASSES.has(node.comfyClass || node.type || "")) return;
    hideDirectorRefineComfyWidgets(node);
    const editor = node._minimaxEditor;
    if (editor) {
        syncRefinePanelFromWidgets(editor);
        editor.syncSemanticBridgePanelFromWidgets?.();
    }
}

function hookDirectorBuiltinRefine(node) {
    hideDirectorRefineComfyWidgets(node);
}

function refinePanelFieldHtml(id, labelKey, fallback, inner) {
    const useI18n = String(labelKey || "").startsWith("widget.");
    const text = useI18n ? t(labelKey) : fallback;
    const attr = useI18n ? ` data-i18n="${labelKey}"` : "";
    return `<label class="bd-refine-field" data-show="${id}">`
        + `<span${attr}>${text}</span>${inner}</label>`;
}

function applyFieldTooltips(panel, descriptions) {
    if (!panel || !descriptions) return;
    for (const [key, description] of Object.entries(descriptions)) {
        const selector = `[data-show="${key}"], [data-w="${key}"], [data-r="${key}"]`;
        const field = panel.querySelector(selector);
        if (!field) continue;
        const host = field.closest("label") || field;
        host.title = description;
        field.title = description;
    }
}

export function mountDirectorRefinePanel(editor) {
    const node = editor?.node;
    const bar = editor?.outputBarEl;
    if (!node || !bar || editor._mmxRefinePanelMounted) return;
    editor._mmxRefinePanelMounted = true;
    hideDirectorRefineComfyWidgets(node);

    const wrap = document.createElement("span");
    wrap.className = "bd-out-refine-wrap";
    wrap.innerHTML = `
        <button type="button" class="bd-btn" data-r="refine-cfg" data-i18n="widget.refineConfig">${t("widget.refineConfig")}</button>
        <span class="bd-refine-mp-inline hidden" data-r="refine-mp-inline">
            <span data-i18n="widget.refineUpscaleMegapixels">${t("widget.refineUpscaleMegapixels")}</span>
            <input type="number" data-r="refine-mp-inline-input" min="0" max="16" step="0.1">
        </span>
    `;
    const tools = bar.querySelector(".bd-live-preview-tools");
    if (tools) bar.insertBefore(wrap, tools);
    else bar.appendChild(wrap);

    const modeVals = comboValues(node, "refine_mode", ["refine", "upscale", "latent_upscale"]);
    const samplerVals = comboValues(node, "refine_sampler", ["euler"]);
    const schedulerVals = comboValues(node, "refine_scheduler", ["simple"]);
    const methodVals = comboValues(node, "refine_upscale_method", ["h3_latent", "lanczos", "nvidia_rtx_vsr"]);
    const latentVals = comboValues(node, "refine_latent_upscale_model", []);
    const seedVals = comboValues(node, "refine_seed_mode", ["inherit", "offset"]);
    const spacingVals = comboValues(node, "refine_spacing", ["cosine", "linear", "exponential"]);
    const axisVals = comboValues(node, "refine_tile_axis", ["auto", "H", "W"]);
    const aspectVals = [
        FOLLOW_DIRECTOR_ASPECT,
        ...RESOLUTION_ASPECTS.map(([label]) => label),
        CUSTOM_ASPECT_RATIO,
    ];

    const panel = document.createElement("div");
    panel.className = "bd-refine-panel hidden";
    panel.setAttribute("data-r", "refine-panel");
    panel.innerHTML = [
        `<div class="bd-refine-group">二采采样</div>`,
        `<label class="bd-refine-field row">`
            + `<input type="checkbox" data-r="refine-enable">`
            + `<span data-i18n="widget.refineEnable">${t("widget.refineEnable")}</span></label>`,
        refinePanelFieldHtml("mode", "widget.refineMode", "模式",
            `<select data-w="refine_mode">${optionHtml(modeVals, widgetValue(widgetByName(node, "refine_mode")))}</select>`),
        refinePanelFieldHtml("sampler", "widget.refineSampler", "二采采样器",
            `<select data-w="refine_sampler">${optionHtml(samplerVals, widgetValue(widgetByName(node, "refine_sampler")))}</select>`),
        refinePanelFieldHtml("passes", "widget.refinePasses", "精修次数",
            `<input type="number" data-w="refine_passes" min="1" max="9999" step="1">`),
        refinePanelFieldHtml("steps", "widget.refineSampleSteps", "二采步数",
            `<input type="number" data-w="refine_sample_steps" min="1" max="200" step="1">`),
        refinePanelFieldHtml("scheduler", "widget.refineScheduler", "二采调度器",
            `<select data-w="refine_scheduler">${optionHtml(schedulerVals, widgetValue(widgetByName(node, "refine_scheduler")))}</select>`),
        refinePanelFieldHtml("denoise", "widget.refineDenoise", "二采 denoise",
            `<input type="number" data-w="refine_denoise" min="0" max="1" step="0.01">`),
        refinePanelFieldHtml("seed", "widget.refineSeedMode", "种子模式",
            `<select data-w="refine_seed_mode">${optionHtml(seedVals, widgetValue(widgetByName(node, "refine_seed_mode")))}</select>`),
        `<label class="bd-refine-field row" data-show="skip"><input type="checkbox" data-w="refine_skip_fl2v"><span data-i18n="widget.refineSkipFl2v">跳过 fl2v</span></label>`,
        `<div class="bd-refine-group" data-show="canvas-section">画幅</div>`,
        refinePanelFieldHtml("aspect", "widget.refineAspectRatio", "比例",
            `<select data-w="refine_aspect_ratio">${optionHtml(aspectVals, widgetValue(widgetByName(node, "refine_aspect_ratio")) || FOLLOW_DIRECTOR_ASPECT)}</select>`),
        refinePanelFieldHtml("mp", "widget.refineMegapixels", "百万像素",
            `<input type="number" data-w="refine_megapixels" min="0" max="16" step="0.1">`),
        refinePanelFieldHtml("width", "widget.refineWidth", "宽",
            `<input type="number" data-w="refine_width" min="0" max="8192" step="32">`),
        refinePanelFieldHtml("height", "widget.refineHeight", "高",
            `<input type="number" data-w="refine_height" min="0" max="8192" step="32">`),
        `<div class="bd-refine-group" data-show="upscale-section">放大</div>`,
        refinePanelFieldHtml("method", "widget.refineUpscaleMethod", "放大方式",
            `<select data-w="refine_upscale_method">${optionHtml(methodVals, widgetValue(widgetByName(node, "refine_upscale_method")))}</select>`),
        refinePanelFieldHtml("h3model", "widget.refineLatentModel", "H3 latent 放大",
            `<select data-w="refine_latent_upscale_model">${optionHtml(latentVals, widgetValue(widgetByName(node, "refine_latent_upscale_model")))}</select>`),
        `<div class="bd-refine-group" data-show="extra-section">低噪加步</div>`,
        refinePanelFieldHtml("extra", "widget.refineExtraSteps", "低噪加步",
            `<input type="number" data-w="refine_extra_steps" min="0" max="15" step="1">`),
        refinePanelFieldHtml("extra-start", "widget.refineStartAtSigma", "加步起始 sigma",
            `<input type="number" data-w="refine_start_at_sigma" min="0" max="20" step="0.01">`),
        refinePanelFieldHtml("extra-end", "widget.refineEndAtSigma", "加步结束 sigma",
            `<input type="number" data-w="refine_end_at_sigma" min="0" max="5" step="0.01">`),
        refinePanelFieldHtml("extra-curve", "widget.refineSpacing", "加步曲线",
            `<select data-w="refine_spacing">${optionHtml(spacingVals, widgetValue(widgetByName(node, "refine_spacing")))}</select>`),
        `<div class="bd-refine-group" data-show="tile-section">分块</div>`,
        `<label class="bd-refine-field row" data-show="tile"><input type="checkbox" data-w="refine_tile"><span data-i18n="widget.refineTile">分块</span></label>`,
        refinePanelFieldHtml("tiles", "widget.refineNTiles", "分块数",
            `<input type="number" data-w="refine_n_tiles" min="1" max="8" step="1">`),
        refinePanelFieldHtml("axis", "widget.refineTileAxis", "分块轴",
            `<select data-w="refine_tile_axis">${optionHtml(axisVals, widgetValue(widgetByName(node, "refine_tile_axis")))}</select>`),
        refinePanelFieldHtml("overlap", "widget.refineTileOverlap", "重叠",
            `<input type="number" data-w="refine_tile_overlap" min="0" max="32" step="1">`),
        refinePanelFieldHtml("notile", "widget.refineMaxSizeNoTile", "不分块上限",
            `<input type="number" data-w="refine_max_size_for_no_tile" min="8" max="256" step="1">`),
        `<label class="bd-refine-field row" data-show="seams"><input type="checkbox" data-w="refine_seams"><span data-i18n="widget.refineSeams">接缝精修</span></label>`,
        refinePanelFieldHtml("seam-steps", "widget.refineSeamSteps", "接缝步数",
            `<input type="number" data-w="refine_seam_steps" min="1" max="25" step="1">`),
    ].join("");
    applyFieldTooltips(panel, {
        "refine-enable": "启用二采；关闭时直接使用一采结果。",
        mode: "二采模式：同尺寸精修、放大后二采，或只做 latent 放大。",
        sampler: "二采使用的采样算法。",
        passes: "二采重复精修的次数；次数越高耗时越长。",
        steps: "每次二采的采样步数；更高通常更细致，但更慢。",
        scheduler: "二采的 sigma 调度曲线。",
        denoise: "二采去噪强度；越高改动越大，越低越保留一采内容。",
        seed: "二采种子来源：继承一采种子，或按段偏移。",
        skip: "对 FL2V 镜头跳过二采/放大，保留首尾帧约束下的一采结果。",
        aspect: "二采或放大的目标宽高比；跟随导演台表示不单独改比例。",
        mp: "二采/放大的目标像素量；数值越大画布越大、显存和耗时越高。",
        width: "自定义目标画布宽度，自动对齐到模型要求的尺寸倍数。",
        height: "自定义目标画布高度，自动对齐到模型要求的尺寸倍数。",
        method: "放大方式：H3 latent、像素插值或 NVIDIA RTX VSR。",
        h3model: "H3 latent 放大使用的 3D latent 权重。",
        extra: "低噪阶段额外增加的采样步数；0 表示关闭。",
        "extra-start": "低噪加步开始生效的 sigma 阈值。",
        "extra-end": "低噪加步结束生效的 sigma 阈值。",
        "extra-curve": "低噪加步在起止 sigma 之间的插值曲线。",
        tile: "将高清二采拆成多个空间块，以降低峰值显存。",
        tiles: "高清二采的空间分块数量。",
        axis: "空间分块方向；auto 自动选择较长的 latent 轴。",
        overlap: "相邻空间块的 latent 重叠宽度，用于减轻拼接接缝。",
        notile: "低于该 latent 尺寸时不分块，直接整幅采样。",
        seams: "启用额外的接缝精修，改善分块边界。",
        "seam-steps": "接缝精修使用的末尾采样步数。",
    });
    bar.after(panel);
    editor.refineBarEl = wrap;
    editor.refinePanelEl = panel;

    const enableCb = panel.querySelector('[data-r="refine-enable"]');
    const cfgBtn = wrap.querySelector('[data-r="refine-cfg"]');
    editor.refineEnableEl = enableCb;
    editor.refineCfgBtn = cfgBtn;
    editor.refineMpInline = wrap.querySelector('[data-r="refine-mp-inline"]');
    editor.refineMpInlineInput = wrap.querySelector('[data-r="refine-mp-inline-input"]');

    enableCb.addEventListener("change", () => {
        writeRefineWidget(node, "refine_enable", !!enableCb.checked);
        applyDirectorRefinePassDefaults(editor, directorRefineActive(node));
        rememberDirectorRefineActive(node);
        updateRefinePanelVisibility(editor);
        editor.updateOutputPreview?.();
        editor.updateDomWidgetHeight?.();
    });
    cfgBtn.addEventListener("click", () => {
        const next = !editor._mmxRefinePanelOpen;
        closePassPanels(editor, next ? "refine" : "");
        editor._mmxRefinePanelOpen = next;
        const after = () => {
            updateRefinePanelVisibility(editor);
            editor.resizeNodeForContentMinChange?.();
        };
        if (next) refreshDrawerModelSelects(editor).finally(after);
        else after();
    });
    refreshDrawerModelSelects(editor);
    panel.addEventListener("change", (e) => {
        const el = e.target?.closest?.("[data-w]");
        if (!el) return;
        const name = el.getAttribute("data-w");
        const value = el.type === "checkbox" ? !!el.checked : (el.type === "number" ? Number(el.value) : el.value);
        writeRefineWidget(node, name, value);
        updateRefinePanelVisibility(editor);
        editor.updateOutputPreview?.();
        editor.updateDomWidgetHeight?.();
    });
    panel.addEventListener("keydown", (e) => e.stopPropagation());
    editor.refineMpInlineInput?.addEventListener("change", () => {
        const value = Number(editor.refineMpInlineInput.value);
        writeRefineWidget(node, "refine_megapixels", value);
        const panelInput = panel.querySelector('[data-w="refine_megapixels"]');
        if (panelInput) panelInput.value = editor.refineMpInlineInput.value;
        editor.updateOutputPreview?.();
    });
    editor.refineMpInlineInput?.addEventListener("keydown", (e) => e.stopPropagation());

    syncRefinePanelFromWidgets(editor);
    updateRefinePanelVisibility(editor);
}

function syncRefinePanelFromWidgets(editor) {
    const node = editor?.node;
    const panel = editor?.refinePanelEl;
    if (!node || !panel) return;
    hideDirectorRefineComfyWidgets(node);
    if (editor.refineEnableEl) {
        editor.refineEnableEl.checked = boolWidgetValue(node, "refine_enable");
        editor.refineEnableEl.disabled = directorHasRefineLink(node);
    }
    if (editor.refineBarEl) {
        editor.refineBarEl.classList.toggle("linked", directorHasRefineLink(node));
    }
    for (const el of panel.querySelectorAll("[data-w]")) {
        const name = el.getAttribute("data-w");
        const raw = widgetValue(widgetByName(node, name));
        if (el.type === "checkbox") el.checked = isTruthyFlag(raw);
        else if (raw != null && raw !== "") el.value = raw;
    }
    updateRefinePanelVisibility(editor);
}

function syncRefineMpInline(editor) {
    const node = editor?.node;
    const wrap = editor?.refineMpInline;
    const input = editor?.refineMpInlineInput;
    if (!node || !wrap || !input) return;
    const mode = String(widgetValue(widgetByName(node, "refine_mode")) ?? "refine").toLowerCase();
    const enabled = directorHasRefineLink(node) || boolWidgetValue(node, "refine_enable");
    const upscale = mode.includes("upscale") && !mode.includes("latent");
    wrap.classList.toggle("hidden", !(enabled && upscale));
    const mp = widgetValue(widgetByName(node, "refine_megapixels"));
    if (mp != null && mp !== "" && document.activeElement !== input) input.value = mp;
}

function updateRefinePanelVisibility(editor) {
    const node = editor?.node;
    const panel = editor?.refinePanelEl;
    if (!panel || !node) return;
    const linked = directorHasRefineLink(node);
    const enabled = linked || boolWidgetValue(node, "refine_enable");
    const open = !!editor._mmxRefinePanelOpen && !linked;
    panel.classList.toggle("hidden", !open);
    editor.refineCfgBtn?.classList.toggle("active", enabled);
    if (editor.refineCfgBtn) editor.refineCfgBtn.disabled = linked;
    syncRefineMpInline(editor);
    const mode = String(widgetValue(widgetByName(node, "refine_mode")) ?? "refine").toLowerCase();
    const upscale = mode.includes("upscale") && !mode.includes("latent");
    const latentOnly = mode.includes("latent");
    const needsCanvas = upscale || latentOnly;
    const method = String(widgetValue(widgetByName(node, "refine_upscale_method")) ?? "").toLowerCase();
    const tileOn = !latentOnly && boolWidgetValue(node, "refine_tile");
    const extraOn = Number(widgetValue(widgetByName(node, "refine_extra_steps")) || 0) > 0;
    const aspect = widgetValue(widgetByName(node, "refine_aspect_ratio"));
    const custom = isCustomAspect(aspect);
    const show = {
        mode: true,
        method: upscale,
        h3model: latentOnly || (upscale && method === "h3_latent"),
        sampler: !latentOnly,
        passes: !latentOnly,
        steps: !latentOnly,
        scheduler: !latentOnly,
        denoise: !latentOnly,
        extra: !latentOnly,
        "extra-start": !latentOnly && extraOn,
        "extra-end": !latentOnly && extraOn,
        "extra-curve": !latentOnly && extraOn,
        seed: !latentOnly,
        aspect: needsCanvas,
        mp: needsCanvas && !custom,
        width: needsCanvas && custom,
        height: needsCanvas && custom,
        skip: true,
        "canvas-section": needsCanvas,
        "upscale-section": needsCanvas,
        "extra-section": !latentOnly,
        "tile-section": !latentOnly,
        tile: !latentOnly,
        tiles: tileOn,
        axis: tileOn,
        overlap: tileOn,
        notile: tileOn,
        seams: tileOn,
        "seam-steps": tileOn && boolWidgetValue(node, "refine_seams"),
    };
    for (const field of panel.querySelectorAll("[data-show]")) {
        const key = field.getAttribute("data-show");
        field.classList.toggle("hidden", show[key] === false);
    }
}

function rememberDirectorRefineActive(node) {
    if (!node) return;
    node._mmxRefineActive = directorRefineActive(node);
}

function maybeApplyDirectorPassDefaults(node) {
    const now = directorRefineActive(node);
    if (node._mmxRefineActive === now) return;
    const prev = node._mmxRefineActive;
    node._mmxRefineActive = now;
    if (prev === undefined) return;
    applyDirectorRefinePassDefaults(node._minimaxEditor, now);
}

app.registerExtension({
    name: "ComfyUI.H3_D_NEOBuiltinRefine",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (DIRECTOR_CLASSES.has(nodeData?.name)) {
            stripHiddenDirectorInputDefs(nodeData);
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function (...args) {
                const r = onNodeCreated?.apply(this, args);
                hideDirectorRefineComfyWidgets(this);
                rememberDirectorRefineActive(this);
                queueMicrotask(() => hideDirectorRefineComfyWidgets(this));
                return r;
            };
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (...args) {
                const r = onConfigure?.apply(this, args);
                hideDirectorRefineComfyWidgets(this);
                rememberDirectorRefineActive(this);
                queueMicrotask(() => {
                    hideDirectorRefineComfyWidgets(this);
                    syncDirectorBuiltinRefineWidgets(this);
                });
                return r;
            };
            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function (...args) {
                return onWidgetChanged?.apply(this, args);
            };
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (...args) {
                const result = onConnectionsChange?.apply(this, args);
                syncDirectorBuiltinRefineWidgets(this);
                maybeApplyDirectorPassDefaults(this);
                return result;
            };
            return;
        }
    },
    nodeCreated(node) {
        if (DIRECTOR_CLASSES.has(node?.comfyClass || node?.type || "")) {
            hideDirectorRefineComfyWidgets(node);
            rememberDirectorRefineActive(node);
        }
    },
    loadedGraphNode(node) {
        if (DIRECTOR_CLASSES.has(node?.comfyClass || node?.type || "")) {
            hideDirectorRefineComfyWidgets(node);
            rememberDirectorRefineActive(node);
        }
    },
    afterConfigureGraph() {
        const graph = app.graph ?? app.canvas?.graph;
        for (const node of graph?._nodes ?? graph?.nodes ?? []) {
            if (DIRECTOR_CLASSES.has(node?.comfyClass || node?.type || "")) {
                hideDirectorRefineComfyWidgets(node);
                syncDirectorBuiltinRefineWidgets(node);
                rememberDirectorRefineActive(node);
            }
        }
    },
});
