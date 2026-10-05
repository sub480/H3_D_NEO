import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';
import * as helpers from '../web/js/minimax_gen_timeline.js';

const code = readFileSync(new URL('../web/js/minimax_timeline.js', import.meta.url), 'utf8');

function method(name, extra = {}) {
    const start = code.indexOf(`\n    ${name}(`) + 5;
    assert.ok(start > 4, name);
    const end = code.indexOf('\n    }', start) + 6;
    return vm.runInNewContext(`({ ${code.slice(start, end)} })`, {
        ...helpers, H3_FPS: 24, DEFAULT_CONTINUITY_FRAMES: 4,
        isVideoEditTaskKey: key => key === 'v2v' || key === 'rv2v',
        isContinuityEligible: () => false, isContinuityEnabled: () => false,
        ...extra,
    })[name];
}

function editor(tasks = ['v2v'], enabled = false) {
    const widget = { value: enabled };
    return {
        timeline: { global: { taskType: 'mixed' }, output: { exportSourceImages: enabled },
            segments: tasks.map(taskType => ({ taskType })) },
        getTaskKey: () => 'mixed', isMixedMode: () => true,
        hasMixedVideoEditGroup: method('hasMixedVideoEditGroup'),
        showsOutputSourceImages: method('showsOutputSourceImages'),
        syncExportSourceImagesUI: method('syncExportSourceImagesUI'),
        syncFromWidgets: method('syncFromWidgets'),
        exportSourceImagesCb: { checked: false },
        exportSourceImagesWrap: { classList: { toggle(_, hidden) { this.hidden = hidden; } } },
        widget: name => name === 'export_source_images' ? widget : null,
        getTotalFrames: () => 24, isFl2vMode: () => false,
        syncOutputToWidgets() {},
    };
}

test('mixed V2V and RV2V groups expose source export and preserve its enabled value', () => {
    for (const task of ['v2v', 'rv2v']) {
        const e = editor(['t2v', task], true);
        assert.equal(e.showsOutputSourceImages(), true);
        e.syncExportSourceImagesUI();
        e.syncFromWidgets();
        assert.equal(e.exportSourceImagesWrap.classList.hidden, false);
        assert.equal(e.exportSourceImagesCb.checked, true);
        assert.equal(e.widget('export_source_images').value, true);
        assert.equal(e.timeline.output.exportSourceImages, true);
    }
});

test('export stays opt-in and non-source tasks do not enable source decoding', () => {
    const off = editor(['v2v']);
    off.syncExportSourceImagesUI();
    off.syncFromWidgets();
    assert.equal(off.widget('export_source_images').value, false);
    for (const task of ['t2v', 'i2v', 'fl2v', 'r2v']) {
        const e = editor([task], true);
        e.syncExportSourceImagesUI();
        e.syncFromWidgets();
        assert.equal(e.exportSourceImagesWrap.classList.hidden, true);
        assert.equal(e.widget('export_source_images').value, false);
        assert.equal(e.timeline.output.exportSourceImages, true);
    }
});

test('saved source preference survives queue synchronization before checkbox restoration', () => {
    const e = editor(['rv2v'], true);
    assert.equal(e.exportSourceImagesCb.checked, false);
    e.syncFromWidgets();
    assert.equal(e.timeline.output.exportSourceImages, true);
    assert.equal(e.widget('export_source_images').value, true);
    e.syncExportSourceImagesUI();
    assert.equal(e.exportSourceImagesCb.checked, true);
});

test('group type switches retain preference and source widget follows eligibility', () => {
    const e = editor(['v2v'], true);
    e.syncExportSourceImagesUI();
    e.timeline.segments[0].taskType = 't2v';
    e.syncExportSourceImagesUI();
    e.syncFromWidgets();
    assert.equal(e.widget('export_source_images').value, false);
    e.timeline.segments[0].taskType = 'rv2v';
    e.syncFromWidgets();
    e.syncExportSourceImagesUI();
    assert.equal(e.widget('export_source_images').value, true);
    assert.equal(e.exportSourceImagesCb.checked, true);
});

test('checkbox change handler persists both checked and unchecked states', () => {
    const handler = code.indexOf('this.exportSourceImagesCb.onchange =');
    const start = code.lastIndexOf('        if (this.exportSourceImagesCb) {', handler);
    assert.ok(start >= 0);
    const end = code.indexOf('\n        }', start) + '\n        }'.length;
    const bind = vm.runInNewContext(`(function() { ${code.slice(start, end)} })`);
    const e = editor(['v2v']);
    e.commit = () => e.syncFromWidgets();
    bind.call(e);
    for (const checked of [true, false]) {
        e.exportSourceImagesCb.checked = checked;
        e.exportSourceImagesCb.onchange();
        assert.equal(e.timeline.output.exportSourceImages, checked);
        assert.equal(e.widget('export_source_images').value, checked);
    }
});

test('rendering groups refreshes source export visibility after add/delete/type changes', () => {
    const e = editor(['t2v'], true);
    e.updateOutputPreview = () => {};
    const render = method('renderImageBatchGroups', { renderImageBatchGroups() {} });
    render.call(e);
    assert.equal(e.exportSourceImagesWrap.classList.hidden, true);
    e.timeline.segments.push({ taskType: 'v2v' });
    render.call(e);
    assert.equal(e.exportSourceImagesWrap.classList.hidden, false);
    e.timeline.segments.pop();
    render.call(e);
    assert.equal(e.exportSourceImagesWrap.classList.hidden, true);
});
