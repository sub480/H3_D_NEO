import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';
import {
    SOURCE_ASPECT_RATIO, DEFAULT_ASPECT_RATIO, normalizeAspectRatioLabel,
    resolutionFromSelector, groupSourceVideoDimensions, groupSourceDimensions, groupSourceImageRef,
} from '../web/js/minimax_gen_timeline.js';
import { aspectDisplayLabel } from '../web/js/minimax_i18n.js';

const timelineCode = readFileSync(new URL('../web/js/minimax_timeline.js', import.meta.url), 'utf8');
const batchCode = readFileSync(new URL('../web/js/minimax_image_batch.js', import.meta.url), 'utf8');
const helpers = await import('../web/js/minimax_gen_timeline.js');
const dimensionsStart = timelineCode.indexOf('function snapDim(');
const dimensionsEnd = timelineCode.indexOf('\n}', timelineCode.indexOf('function resolveOutputDimensions(')) + 2;
const resolveOutputDimensions = vm.runInNewContext(
    `${timelineCode.slice(dimensionsStart, dimensionsEnd)}; resolveOutputDimensions`,
);

// Execute real editor methods without loading the ComfyUI app or creating a browser.
function method(name, extra = {}) {
    const start = timelineCode.indexOf(`\n    ${name}(`) + 5;
    assert.ok(start > 4, name);
    const end = timelineCode.indexOf('\n    }', start) + 6;
    const context = vm.createContext({ ...helpers, aspectDisplayLabel, resolveOutputDimensions, document: {}, ...extra });
    return vm.runInContext(`({ ${timelineCode.slice(start, end)} })`, context)[name];
}

function group(width, height, taskType = 'v2v') {
    return { taskType, sourceVideo: { mediaKind: 'video', totalFrames: 5,
        video: { width, height }, videoClips: [{ width, height, sourceFrameCount: 5 }] } };
}

test('selector preserves canonical source option, MP math and mandatory 32 grid', () => {
    assert.equal(normalizeAspectRatioLabel(SOURCE_ASPECT_RATIO), SOURCE_ASPECT_RATIO);
    assert.equal(normalizeAspectRatioLabel('16:9'), DEFAULT_ASPECT_RATIO);
    assert.equal(aspectDisplayLabel(SOURCE_ASPECT_RATIO), '与源素材一致');
    assert.equal(resolutionFromSelector(SOURCE_ASPECT_RATIO, 1), null);
    for (const [width, height] of [[1920, 1080], [1080, 1920], [1234, 567], [1, 10000]]) {
        for (const mp of [0.1, 0.4, 1, 16]) {
            const resolved = resolutionFromSelector(SOURCE_ASPECT_RATIO, mp, 8, { width, height });
            assert.equal(resolved.multiple, 32);
            assert.equal(resolved.width % 32, 0);
            assert.equal(resolved.height % 32, 0);
            assert.ok(resolved.width >= 32 && resolved.height >= 32);
        }
    }
    assert.deepEqual(resolutionFromSelector(SOURCE_ASPECT_RATIO, 0.4, 32, { width: 1080, height: 1920 }),
        { width: 480, height: 864, megapixels: 0.4, aspectRatio: SOURCE_ASPECT_RATIO, multiple: 32 });
});

test('dimensions come from each group and its selected source clip', () => {
    const seg = group(1920, 1080);
    seg.sourceVideo.videoClips.push({ width: 1080, height: 1920, sourceFrameCount: 5 });
    seg.sourceVideo.rangeStart = 5;
    assert.deepEqual(groupSourceVideoDimensions(seg), { width: 1080, height: 1920 });
    seg.sourceVideo.rangeStart = 0;
    seg.sourceVideo.video.frameMap = [{ clip: 1, frame: 0 }];
    assert.deepEqual(groupSourceVideoDimensions(seg), { width: 1080, height: 1920 });
    assert.equal(groupSourceVideoDimensions({ ...seg, taskType: 't2v' }), null);
    seg.sourceVideo.mediaKind = 'image';
    assert.equal(groupSourceVideoDimensions(seg), null);
});

test('source selection preserves nonvideo fallback canvas and survives workflow roundtrip', () => {
    const editor = { timeline: { output: { width: 1024, height: 768, megapixels: 0.4 },
        segments: [group(1080, 1920)] }, selectedIndex: 0, outAspect: {}, outMp: {} };
    const apply = method('applyResolutionSelector');
    const resolved = apply.call(editor, SOURCE_ASPECT_RATIO, 1);
    assert.equal(resolved.width, 1024);
    assert.equal(editor.timeline.output.height, 768);
    assert.equal(editor.timeline.output.megapixels, 1);
    const restored = JSON.parse(JSON.stringify(editor.timeline));
    restored.output.aspectRatio = normalizeAspectRatioLabel(restored.output.aspectRatio);
    assert.equal(restored.output.aspectRatio, SOURCE_ASPECT_RATIO);
    assert.equal(restored.output.width, 1024);
    const payload = method('buildTimelinePayload', {
        normalizeOutputContinuity: value => value, stripTimelineContinuityRootFields() {},
        stripTimelineEphemeralFields() {}, sanitizeSegmentForPayload: value => value,
        isSegmentContinuityFromPrev: () => false, isSegmentContinuityForcePrevCache: () => false,
    }).call({ timeline: restored, _persistCurrentBatchWorkspace() {},
        getFrameRate: () => 24, _runSelectionPayload: () => ({}) });
    assert.equal(payload.output.aspectRatio, SOURCE_ASPECT_RATIO);
    assert.equal(payload.output.mode, 'fixed');
    assert.equal(payload.output.width, 1024);
});

test('preview changes on source replacement and group selection, not top-level video', () => {
    const preview = method('_firstPassSize');
    const editor = { timeline: { output: { aspectRatio: SOURCE_ASPECT_RATIO, megapixels: 0.4,
        width: 1024, height: 768 }, video: { width: 9999, height: 1 },
        segments: [group(1920, 1080), group(1080, 1920, 'rv2v'), { taskType: 't2v' }] },
        selectedIndex: 0, getSourceDimensions() { throw Error('wrong-group lookup'); } };
    assert.equal(preview.call(editor).width, 864);
    editor.selectedIndex = 1;
    assert.equal(preview.call(editor).width, 480);
    editor.timeline.segments[1] = group(1234, 567);
    assert.ok(preview.call(editor).width > preview.call(editor).height);
    editor.selectedIndex = 2;
    assert.equal(preview.call(editor).width, 1024);
    assert.equal(preview.call(editor).height, 768);
    assert.match(batchCode, /editor\.selectedIndex = next;\s*editor\.updateOutputPreview\?\.\(\);/);
    let refreshed = 0;
    method('renderImageBatchGroups', { renderImageBatchGroups() {} }).call({
        updateOutputPreview() { refreshed++; },
    });
    assert.equal(refreshed, 1);
});

test('source option is visible in image-source and video-source modes without changing stored choice', () => {
    const option = {};
    const editor = { timeline: { output: { aspectRatio: SOURCE_ASPECT_RATIO } },
        outAspect: { querySelector: () => option, classList: { toggle() {} } },
        isImageBatch: () => true, isGenMode: () => false, isFl2vMode: () => false,
        applyResolutionSelector() {} };
    const update = method('updateOutputModeUI');
    for (const key of ['mixed', 'i2v', 'fl2v', 'v2v', 'rv2v']) {
        editor.getTaskKey = () => key;
        update.call(editor);
        assert.equal(option.hidden, false);
        assert.equal(option.disabled, false);
    }
    editor.getTaskKey = () => 't2v';
    update.call(editor);
    assert.equal(option.hidden, true);
    assert.equal(option.disabled, true);
    assert.equal(editor.timeline.output.aspectRatio, SOURCE_ASPECT_RATIO);
});

test('legacy source resolution preview ignores target MP budget', () => {
    const seg = { ...group(320, 640), videoResolution: 'source' };
    const editor = { selectedIndex: 0, timeline: { segments: [seg],
        output: { aspectRatio: SOURCE_ASPECT_RATIO, megapixels: 16, width: 864, height: 480 } } };
    const size = method('_firstPassSize').call(editor);
    assert.equal(size.width, 320);
    assert.equal(size.height, 640);
});

test('image sources and FL2V first/last priority drive preview', () => {
    const landscape = { imageFile: 'first.png', width: 160, height: 90 };
    const portrait = { imageFile: 'last.png', width: 90, height: 160 };
    const cases = [
        [{ taskType: 'i2v', genImage: landscape, videoResolution: 'source' }, landscape],
        [{ taskType: 'i2v', genImage: portrait }, portrait],
        [{ taskType: 'fl2v', startImage: landscape }, landscape],
        [{ taskType: 'fl2v', endImage: portrait }, portrait],
        [{ taskType: 'fl2v', startImage: landscape, endImage: portrait }, landscape],
    ];
    const editor = { selectedIndex: 0, timeline: {
        output: { aspectRatio: SOURCE_ASPECT_RATIO, megapixels: 0.4, width: 1024, height: 768 },
        segments: cases.map(([seg]) => seg),
    } };
    const preview = method('_firstPassSize');
    for (const [index, [seg, ref]] of cases.entries()) {
        assert.equal(groupSourceImageRef(seg), ref);
        assert.deepEqual(groupSourceDimensions(seg), { width: ref.width, height: ref.height });
        editor.selectedIndex = index;
        const size = preview.call(editor);
        assert.equal(size.width, ref.width > ref.height ? 864 : 480);
        assert.equal(size.height, ref.width > ref.height ? 480 : 864);
    }
    editor.selectedIndex = 4;
    editor.timeline.segments[4].startImage = null;
    assert.equal(preview.call(editor).width, 480);
    editor.timeline.segments[4].startImage = { imageFile: 'unknown.png' };
    assert.equal(preview.call(editor).width, 1024);
    assert.equal(groupSourceDimensions(editor.timeline.segments[4]), null);
    assert.equal(SOURCE_ASPECT_RATIO, '与原视频一致');
});

function topFunction(code, name, extra = {}) {
    const start = code.indexOf(`function ${name}(`);
    assert.ok(start >= 0, name);
    const end = code.indexOf('\n}', start) + 2;
    return vm.runInNewContext(`${code.slice(start, end)}; ${name}`, { ...helpers, ...extra });
}

test('payload sanitizer preserves source dimensions and inline tail images', () => {
    const sanitizeSourceImage = topFunction(timelineCode, 'sanitizeSourceImage');
    const sanitize = topFunction(timelineCode, 'sanitizeSegmentForPayload', {
        sanitizeSourceImage, sanitizeRefImage: x => x, sanitizeRefAudio: x => x, sanitizeRefVideo: x => x,
    });
    const seg = sanitize({ taskType: 'fl2v',
        genImage: { imageFile: 'source.png', width: 90, height: 160 },
        startImage: { imageFile: 'first.png', width: 160, height: 90 },
        endImage: { imageB64: 'inline', width: 90, height: 160 },
    });
    const restored = JSON.parse(JSON.stringify(seg));
    assert.equal(restored.genImage.width, 90);
    assert.equal(restored.startImage.width, 160);
    assert.equal(restored.endImage.imageB64, 'inline');
    assert.equal(restored.endImage.height, 160);
});

test('source image metadata refreshes preview and rejects stale callbacks', () => {
    const sync = topFunction(batchCode, 'syncSegSourceImageDimensions');
    const seg = { taskType: 'i2v', genImage: { imageFile: 'first.png' } };
    let previews = 0;
    let writes = 0;
    const editor = { timeline: { segments: [seg] },
        updateOutputPreview() { previews++; }, scheduleTimelineSync() { writes++; } };
    sync(editor, seg, 'genImage', 'first.png', 90, 160);
    assert.equal(seg.genImage.width, 90);
    assert.equal(previews, 1);
    assert.equal(writes, 1);
    sync(editor, seg, 'genImage', 'first.png', 90, 160);
    assert.equal(previews, 1);
    seg.genImage = { imageFile: 'replacement.png' };
    sync(editor, seg, 'genImage', 'first.png', 160, 90);
    assert.equal(seg.genImage.width, undefined);
    editor.timeline.segments = [];
    sync(editor, seg, 'genImage', 'replacement.png', 160, 90);
    assert.equal(seg.genImage.width, undefined);
});
