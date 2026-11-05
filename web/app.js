const $ = selector => document.querySelector(selector);
const fmt = value => value == null ? '—' : Number(value).toLocaleString('en-US', {maximumFractionDigits: 1, minimumFractionDigits: 1});
const names = {seasonal_naive: 'Seasonal naive', last_value_naive: 'Last-value naive', holt_winters: 'Holt-Winters', ridge: 'Ridge', hist_gradient_boosting: 'Histogram gradient boosting'};
const modelName = name => names[name] || name;
const escapeHTML = value => String(value).replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
const shortDate = value => new Date(`${value}T00:00:00Z`).toLocaleDateString('en-GB', {day: 'numeric', month: 'short', timeZone: 'UTC'});
const horizon = $('#horizon');
for (let i = 1; i <= 14; i++) horizon.add(new Option(`${i} day${i > 1 ? 's' : ''}`, i, i === 14, i === 14));

async function get(url) {
    const response = await fetch(url);
    if (!response.ok) throw Error(`${url}: ${response.status}`);
    return response.json();
}

// Size charts to the card, preserving readable phone labels. Null values end a
// line segment: unavailable rolling windows are never drawn as zero error.
const charts = new Map();
function drawChart(el, chart) {
    const {series, dates, unit, points = false} = chart;
    const all = series.flatMap(item => item.values.filter(value => value != null));
    if (!all.length) { el.textContent = 'No chart data available.'; return; }
    const width = Math.max(240, Math.floor(el.clientWidth));
    const height = 260, left = 53, right = 18, top = 32, bottom = 42;
    const upper = Math.max(...all) * 1.12 || 1;
    const magnitude = 10 ** Math.floor(Math.log10(upper / 3));
    const tickStep = Math.ceil(upper / 3 / magnitude) * magnitude;
    const ceiling = Math.ceil(upper / tickStep) * tickStep;
    const x = i => dates.length === 1 ? (left + width - right) / 2 : left + i * (width - left - right) / (dates.length - 1);
    const y = value => height - bottom - value * (height - top - bottom) / ceiling;
    const ticks = Array.from({length: Math.round(ceiling / tickStep) + 1}, (_, i) => tickStep * i);
    const grid = ticks.map(value => `<line x1="${left}" y1="${y(value)}" x2="${width - right}" y2="${y(value)}" class="chart-grid"/><text x="${left - 8}" y="${y(value) + 4}" text-anchor="end">${Math.round(value).toLocaleString('en-US')}</text>`).join('');
    const tickCount = Math.min(dates.length, width < 500 ? 3 : 5);
    const indices = [...new Set(Array.from({length: tickCount}, (_, i) => tickCount === 1 ? 0 : Math.round(i * (dates.length - 1) / (tickCount - 1))))];
    const dateTicks = indices.map(i => `<text x="${x(i)}" y="${height - 16}" text-anchor="${dates.length === 1 ? 'middle' : i === 0 ? 'start' : i === dates.length - 1 ? 'end' : 'middle'}">${shortDate(dates[i])}</text>`).join('');
    const lines = series.map(item => {
        let path = '', active = false;
        item.values.forEach((value, i) => {
            if (value == null) { active = false; return; }
            path += `${active ? 'L' : 'M'}${x(i)},${y(value)} `;
            active = true;
        });
        const markers = points ? item.values.map((value, i) => value == null ? '' : `<circle cx="${x(i)}" cy="${y(value)}" r="4" fill="${item.color}" tabindex="0" role="img" aria-label="${escapeHTML(`${dates[i]}: ${fmt(value)} ${unit}`)}"><title>${escapeHTML(`${dates[i]}: ${fmt(value)} ${unit}`)}</title></circle>`).join('') : '';
        return `<path d="${path}" fill="none" stroke="${item.color}" stroke-width="2.5" stroke-linejoin="round"/>${markers}`;
    }).join('');
    el.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHTML(`${series.map(item => item.name).join(' and ')}; ${dates[0]} to ${dates.at(-1)}; ${unit}`)}"><text x="${left}" y="15">${escapeHTML(unit)}</text>${grid}${dateTicks}${lines}</svg><div class="chart-legend">${series.map(item => `<span><i style="background:${item.color}" aria-hidden="true"></i>${escapeHTML(item.name)}</span>`).join('')}</div>`;
}
const chartResize = new ResizeObserver(entries => {
    for (const entry of entries) {
        const saved = charts.get(entry.target);
        if (saved && saved.width !== Math.floor(entry.contentRect.width)) {
            saved.width = Math.floor(entry.contentRect.width);
            drawChart(entry.target, saved.chart);
        }
    }
});
function lineChart(el, chart) {
    charts.set(el, {chart, width: Math.floor(el.clientWidth)});
    drawChart(el, chart);
    chartResize.observe(el);
}

let forecastController;
async function forecast() {
    forecastController?.abort();
    const controller = new AbortController();
    forecastController = controller;
    $('#forecast-status').textContent = 'Loading historical forecast…';
    $('#forecast-chart').setAttribute('aria-busy', 'true');
    try {
        const response = await fetch('/predict', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify({horizon: Number(horizon.value)}), signal: controller.signal});
        const data = await response.json();
        if (!response.ok) throw Error('Prediction unavailable. Please try again.');
        if (controller.signal.aborted) return;
        lineChart($('#forecast-chart'), {series: [{name: 'Predicted rentals', color: '#0d8065', values: data.predictions.map(point => point.value)}], dates: data.predictions.map(point => point.date), unit: 'Rentals per day', points: true});
        $('#forecast-values').innerHTML = data.predictions.map(point => `<tr><td><time datetime="${escapeHTML(point.date)}">${escapeHTML(point.date)}</time></td><td>${fmt(point.value)}</td></tr>`).join('');
        $('#forecast-details').hidden = false;
        $('#forecast-meta').textContent = `${modelName(data.model_name)} · ${data.model_version} · ${data.disclaimer}`;
        $('#forecast-status').textContent = data.horizon === 1 ? `${shortDate(data.forecast_start)} 2012 · ${fmt(data.predictions[0].value)} predicted rentals` : `${data.horizon} daily predictions · ${shortDate(data.forecast_start)}–${shortDate(data.forecast_end)} 2012`;
    } catch (error) {
        if (error.name === 'AbortError') return;
        charts.delete($('#forecast-chart'));
        $('#forecast-chart').textContent = 'Could not load the forecast. Change Days ahead to retry.';
        $('#forecast-values').replaceChildren();
        $('#forecast-details').hidden = true;
        $('#forecast-meta').textContent = '';
        $('#forecast-status').textContent = 'Forecast unavailable.';
    } finally {
        if (forecastController === controller) $('#forecast-chart').setAttribute('aria-busy', 'false');
    }
}

async function load() {
    try {
        const [model, comparison, horizons, monitor] = await Promise.all(['/model', '/comparison', '/horizons', '/monitoring'].map(get));
        const selected = comparison.find(row => row.selected);
        $('#selected-badge').textContent = `Selected: ${modelName(model.model_id)}`;
        const bestTest = comparison.filter(row => row.mae_test != null).sort((a, b) => a.mae_test - b.mae_test)[0];
        $('#selection-reason').textContent = `${modelName(model.model_id)} was selected using validation only: ${fmt(selected.mae_validation)} rentals/day MAE, ${model.validation_improvement_pct.toFixed(2)}% lower than seasonal naive, passing the fixed 1% improvement gate. ${bestTest.model !== model.model_id ? `${modelName(bestTest.model)} scored better on the held-out test, but the model choice stays frozen to keep that evaluation fair.` : 'The model choice stayed frozen before the held-out test was evaluated.'}`;
        $('#model-rows').innerHTML = comparison.map(row => `<tr class="${row.selected ? 'selected' : ''}"><th scope="row">${escapeHTML(modelName(row.model))}${row.selected ? '<span class="row-tag">Selected</span>' : ''}${row.status_test === 'failed' ? ' (failed)' : ''}</th><td>${fmt(row.mae_validation)}</td><td>${fmt(row.mae_test)}</td><td>${fmt(row.rmse_test)}</td><td>${fmt(row.wape_test)}%</td><td>${fmt(row.test_improvement_vs_seasonal_pct)}%</td></tr>`).join('');
        const bars = horizons.filter(row => row.split === 'test' && row.model === model.model_id && ['days_1_7', 'days_8_14'].includes(row.period));
        const max = Math.max(...bars.map(row => row.mae));
        $('#horizon-model').textContent = `${modelName(model.model_id)} · held-out 14-day backtest · average absolute error in rentals per day`;
        $('#horizon-chart').innerHTML = bars.map(row => `<div class="bar-row"><span>${row.period === 'days_1_7' ? 'Days 1–7' : 'Days 8–14'}</span><div class="bar-track" aria-hidden="true"><div class="bar-fill" style="width:${max ? row.mae / max * 100 : 0}%"></div></div><b>${fmt(row.mae)}</b></div>`).join('');
        $('#monitor-stats').innerHTML = [['Paired replay days', monitor.paired_days], ['Selected day-ahead MAE', fmt(monitor.selected_day_ahead_mae)], ['Seasonal day-ahead MAE', fmt(monitor.baseline_day_ahead_mae)], ['Selected latest 28-day MAE', fmt(monitor.latest_selected_rolling_28_mae)]].map(([label, value]) => `<div class="stat"><b>${value}</b><small>${label}</small></div>`).join('');
        lineChart($('#monitor-chart'), {series: [{name: `${modelName(model.model_id)} · rolling 28-day MAE`, color: '#0d8065', values: monitor.records.map(row => row.selected_rolling_28_mae)}, {name: 'Seasonal naive · rolling 28-day MAE', color: '#ac671c', values: monitor.records.map(row => row.baseline_rolling_28_mae)}], dates: monitor.records.map(row => row.target_date), unit: 'MAE · rentals per day'});
        const firstRolling = monitor.records.find(row => row.selected_rolling_28_mae != null);
        $('#monitor-note').textContent = `Rolling averages need 28 paired days. The blank opening period is expected: the first average is available on ${shortDate(firstRolling.target_date)} 2012 (day 28). These are historical replay metrics, not live alerts.`;
        $('#monitor-flag').textContent = monitor.baseline_selected ? `Baseline selected · change from initial 28-day error: ${monitor.absolute_drift ? 'flagged' : 'not flagged'}.` : `Error versus seasonal naive: ${monitor.relative_deterioration ? 'flagged' : 'not flagged'} · change from initial 28-day error: ${monitor.absolute_drift ? 'flagged' : 'not flagged'}.`;
        $('#model-meta').textContent = `Version ${model.version} · ${model.training_rows} training days · Source SHA-256 ${model.data_sha256}`;
        await forecast();
    } catch (error) {
        $('#release-error').hidden = false;
        $('#release-error').textContent = `Unable to load the historical release (${error.message}). Reload the page to retry.`;
        $('#forecast-status').textContent = 'Release unavailable.';
    }
}
horizon.addEventListener('change', forecast);
load();
