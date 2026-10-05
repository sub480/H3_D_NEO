import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';
import * as helpers from '../web/js/minimax_gen_timeline.js';

const timelineCode = readFileSync(new URL('../web/js/minimax_timeline.js', import.meta.url), 'utf8');
const batchCode = readFileSync(new URL('../web/js/minimax_image_batch.js', import.meta.url), 'utf8');
const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
const plain = value => JSON.parse(JSON.stringify(value));

// Run the real functions, with UI/normalization side effects stubbed, not a browser.
function topFunction(code, name, extra = {}) {
    const start = code.indexOf(`function ${name}(`);
    assert.ok(start >= 0, name);
    const end = code.indexOf('\n}', start) + 2;
    return vm.runInNewContext(`${code.slice(start, end)}; ${name}`, { ...helpers, clamp, ...extra });
}

function method(name, extra = {}) {
    const start = timelineCode.indexOf(`\n    ${name}(`) + 5;
    assert.ok(start > 4, name);
    const end = timelineCode.indexOf('\n    }', start) + 6;
    return vm.runInNewContext(`({ ${timelineCode.slice(start, end)} })`, {
        ...helpers, clamp, remapRunSelectionAfterDelete, ...extra,
    })[name];
}

const remapRunSelectionAfterDelete = topFunction(batchCode, 'remapRunSelectionAfterDelete');
const batchDelete = topFunction(batchCode, 'deleteImageBatchGroup', {
    remapRunSelectionAfterDelete,
    flushBatchPromptInputs: editor => editor.events.push('prompt'),
    flushBatchDurationInputs: editor => editor.events.push('duration'),
    normalizeImageBatchSegments: editor => editor.events.push('normalize'),
});

function editorFor(selection = [0, 1, 2, 3]) {
    return {
        timeline: { segments: ['a', 'b', 'c', 'd'].map((id, i) => ({
            id, taskType: ['t2v', 'fl2v', 'i2v', 'r2v'][i], start: i * 4, length: 4,
        })), runSelection: selection, runSelectEnabled: true },
        selectedIndex: 1, currentFrame: 0, events: [],
        renderImageBatchGroups() { this.events.push('render'); },
        commit() { this.events.push('commit'); },
        updateVideoNameLabel() {}, resizeNodeForContentMinChange() {},
    };
}

test('batch deletion preserves selected group identity for first, middle and last groups', () => {
    for (const index of [0, 1, 3]) {
        const editor = editorFor();
        const expectedIds = editor.timeline.segments.map(seg => seg.id).filter((_, i) => i !== index);
        editor.selectedIndex = 3;
        batchDelete(editor, index);
        assert.deepEqual(plain(editor.timeline.runSelection), [0, 1, 2]);
        assert.deepEqual(editor.timeline.runSelection.map(i => editor.timeline.segments[i].id), expectedIds);
        assert.equal(editor.selectedIndex, 2);
        assert.deepEqual(editor.events, ['prompt', 'duration', 'normalize', 'render', 'commit']);
    }
});

test('disabled run selection still remaps sparse selections and never selects a replacement', () => {
    const editor = editorFor([1, 3]);
    editor.timeline.runSelectEnabled = false;
    batchDelete(editor, 1);
    assert.deepEqual(plain(editor.timeline.runSelection), [2]);
    assert.equal(editor.timeline.segments[2].id, 'd');
    const onlyDeleted = editorFor([1]);
    batchDelete(onlyDeleted, 1);
    assert.deepEqual(plain(onlyDeleted.timeline.runSelection), []);
});

test('invalid, externally owned and final-group deletes do not change selection or drafts', () => {
    for (const index of [-1, 4, 1.5, undefined]) {
        const editor = editorFor();
        const before = plain(editor.timeline);
        batchDelete(editor, index);
        assert.deepEqual(plain(editor.timeline), before);
        assert.deepEqual(editor.events, []);
    }
    for (const lock of ['hasExternalI2vGroups', 'hasExternalR2vGroups', 'lastGroup']) {
        const editor = editorFor([0]);
        if (lock === 'lastGroup') editor.timeline.segments.length = 1;
        else editor[lock] = () => true;
        const before = plain(editor.timeline);
        batchDelete(editor, 0);
        assert.deepEqual(plain(editor.timeline), before);
        assert.deepEqual(editor.events, []);
    }
});

test('actual mixed FL2V group deletion routes through the shared batch path', () => {
    const editor = editorFor([1, 3]);
    Object.assign(editor, {
        getTaskKey: () => 'mixed', getDirectorMode: () => 'prompt_batch',
        isGenMode: method('isGenMode'), isImageBatch: method('isImageBatch'),
        isFl2vMode: method('isFl2vMode'), usesBatchTimeline: method('usesBatchTimeline'),
        getTotalFrames() { return this.timeline.segments.length * 4; },
        updateDomWidgetHeight() {}, scheduleRender() {},
    });
    method('deleteSelectedSegment', { deleteImageBatchGroup: batchDelete }).call(editor);
    assert.deepEqual(plain(editor.timeline.runSelection), [2]);
    assert.equal(editor.timeline.segments[2].id, 'd');
    assert.equal(editor.events.filter(event => event === 'commit').length, 1);
});

test('gen deletion also remaps before commit and ignores an invalid selected group', () => {
    const remove = method('genDeleteSelectedSegment');
    const editor = editorFor([1, 3]);
    remove.call(editor);
    assert.deepEqual(plain(editor.timeline.runSelection), [2]);
    assert.deepEqual(editor.events, ['commit']);
    const before = plain(editor.timeline);
    editor.selectedIndex = -1;
    remove.call(editor);
    assert.deepEqual(plain(editor.timeline), before);
});

test('legacy video deletion remaps indices independently of frame removal', () => {
    const editor = editorFor([1, 3]);
    editor.timeline.segments.forEach(seg => { seg.length = 0; });
    Object.assign(editor, {
        isGenMode: () => false, usesBatchTimeline: () => false,
        isImageBatch: () => false, isFl2vMode: () => false,
        getTotalFrames: () => 16, compactSegmentsAfterDelete() {},
        _prefetchSegmentThumbs() {}, _syncStagePreview() {}, updateStageVisibility() {},
    });
    method('deleteSelectedSegment', { THUMB_PREFETCH_BATCH: 2 }).call(editor);
    assert.deepEqual(plain(editor.timeline.runSelection), [2]);
    assert.equal(editor.timeline.segments[2].id, 'd');
});

test('deleting a split point remaps the removed right group', () => {
    const editor = editorFor([1, 3]);
    Object.assign(editor, {
        isGenMode: () => false, isImageBatch: () => false,
        selectedSplitFrame: 4, getEditableSplitFrames: () => [4, 8, 12],
        updateSelectionUI() {}, updateSplitPointUI() {}, scheduleRender() {},
    });
    method('deleteSelectedSplitPoint').call(editor);
    assert.deepEqual(plain(editor.timeline.runSelection), [2]);
    assert.equal(editor.timeline.segments[0].length, 8);
    assert.equal(editor.timeline.segments[2].id, 'd');
});

test('split deletion preserves identity when imported groups need sorting', () => {
    const editor = editorFor();
    const groups = editor.timeline.segments;
    editor.timeline.segments = [groups[3], groups[1], groups[0], groups[2]];
    editor.timeline.runSelection = [0, 3]; // d and c before sorting.
    Object.assign(editor, {
        isGenMode: () => false, isImageBatch: () => false,
        selectedSplitFrame: 4, getEditableSplitFrames: () => [4, 8, 12],
        updateSelectionUI() {}, updateSplitPointUI() {}, scheduleRender() {},
    });
    method('deleteSelectedSplitPoint').call(editor);
    assert.deepEqual(plain(editor.timeline.runSelection), [1, 2]);
    assert.deepEqual(plain(editor.timeline.runSelection.map(i => editor.timeline.segments[i].id)), ['c', 'd']);
});

const parseTimeline = topFunction(timelineCode, 'parseTimeline', {
    uid: () => 'new-id', coerceTimelineFps: value => value,
    normalizeOutputContinuity: value => value, sanitizeRefVideo: value => value,
    normalizeAudioMode: value => value, stripTimelineContinuityRootFields() {},
    stripTimelineEphemeralFields() {}, clampLivePreviewSpeed: value => value,
    LIVE_PREVIEW_SPEED_DEFAULT: 1, DEFAULT_CONTINUITY_FRAMES: 4, MIN_SEG: 4,
});

test('test baseline drops obsolete common flags without introducing false or changing prompts', () => {
    const build = method('buildTimelinePayload', {
        normalizeOutputContinuity: value => value, stripTimelineContinuityRootFields() {},
        stripTimelineEphemeralFields() {}, sanitizeSegmentForPayload: value => value,
    });
    for (const flags of [{}, { commonEnabled: false }, { commonEnabled: true },
        { common_enabled: false }, { common_enabled: true }]) {
        const raw = { timelineMode: 'prompt_batch', global: { taskType: 'mixed', prompt: 'global', ...flags },
            output: { aspectRatio: helpers.SOURCE_ASPECT_RATIO },
            segments: [{ id: 'empty', start: 0, length: 4, prompt: '' },
                { id: 'local', start: 4, length: 4, prompt: 'local' }] };
        const parsed = parseTimeline(plain(raw), 8, 24);
        const editor = { timeline: parsed, _persistCurrentBatchWorkspace() {},
            getFrameRate: () => 24, _runSelectionPayload: () => ({}) };
        const restored = parseTimeline(plain(build.call(editor)), 8, 24);
        for (const key of ['commonEnabled', 'common_enabled']) {
            assert.equal(Object.hasOwn(restored.global, key), false);
        }
        assert.equal(restored.global.prompt, 'global');
        assert.equal(restored.segments[0].prompt, '');
        assert.equal(restored.segments[1].prompt, 'local');
        assert.equal(restored.output.aspectRatio, helpers.SOURCE_ASPECT_RATIO);
    }
});
