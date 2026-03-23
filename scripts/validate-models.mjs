/**
 * DrAgnes Model Validation Script
 *
 * Runs both HuggingFace ViT models against HAM10000 test images
 * and computes per-class confusion matrices, sensitivity, specificity,
 * PPV, NPV, and compares against DermaSensor benchmarks.
 *
 * Usage: node scripts/validate-models.mjs
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const CACHE_DIR = path.join(PROJECT_DIR, '.cache', 'ham10000');
const CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'];

// Models
const MODEL1_ID = 'Anwarkh1/Skin_Cancer-Image_Classification';
const MODEL2_ID = 'Jayanth2002/dinov2-base-finetuned-SkinDisease';

// New HF router endpoint (api-inference.huggingface.co is 410 Gone)
const MODEL1_URL = `https://router.huggingface.co/hf-inference/models/${MODEL1_ID}`;
const MODEL2_URL = `https://router.huggingface.co/hf-inference/models/${MODEL2_ID}`;

// ---------- Resolve HF token ----------
function findHFToken() {
    // Try environment variable first
    if (process.env.HF_TOKEN) return process.env.HF_TOKEN;
    if (process.env.HUGGINGFACE_TOKEN) return process.env.HUGGINGFACE_TOKEN;

    // Try dragnes .env
    const localEnv = path.join(PROJECT_DIR, '.env');
    if (fs.existsSync(localEnv)) {
        const token = extractEnvVar(localEnv, ['HF_TOKEN', 'HUGGINGFACE_TOKEN', 'HuggingFace_Key']);
        if (token) return token;
    }

    // Try root project .env
    const rootEnv = path.resolve(PROJECT_DIR, '..', '..', '.env');
    if (fs.existsSync(rootEnv)) {
        const token = extractEnvVar(rootEnv, ['HF_TOKEN', 'HUGGINGFACE_TOKEN', 'HuggingFace_Key']);
        if (token) return token;
    }

    // Try ~/.cache/huggingface/token
    const hfCache = path.join(process.env.HOME || '', '.cache', 'huggingface', 'token');
    if (fs.existsSync(hfCache)) return fs.readFileSync(hfCache, 'utf8').trim();

    return null;
}

function extractEnvVar(filePath, keys) {
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        for (const line of content.split('\n')) {
            if (line.startsWith('#') || !line.includes('=')) continue;
            const eqIdx = line.indexOf('=');
            const key = line.substring(0, eqIdx).trim();
            const val = line.substring(eqIdx + 1).trim().replace(/^["']|["']$/g, '');
            if (keys.includes(key) && val.length > 0) return val;
        }
    } catch { /* ignore */ }
    return null;
}

// ---------- Label mappings ----------
// Anwarkh1 model: returns underscore-separated labels from HAM10000 training
// Verified labels: actinic_keratoses, basal_cell_carcinoma,
//   benign_keratosis-like_lesions, dermatofibroma, melanoma,
//   melanocytic_Nevi, vascular_lesions
const ANWARKH1_MAP = {
    // Exact labels returned by the model (underscore format)
    'actinic_keratoses': 'akiec',
    'basal_cell_carcinoma': 'bcc',
    'benign_keratosis-like_lesions': 'bkl',
    'dermatofibroma': 'df',
    'melanoma': 'mel',
    'melanocytic_Nevi': 'nv',
    'melanocytic_nevi': 'nv',   // lowercase variant
    'vascular_lesions': 'vasc',
    // Space-separated fallbacks
    'actinic keratoses': 'akiec',
    'basal cell carcinoma': 'bcc',
    'benign keratosis-like lesions': 'bkl',
    'benign keratosis': 'bkl',
    'melanocytic nevi': 'nv',
    'melanocytic nevus': 'nv',
    'vascular lesion': 'vasc',
    'vascular lesions': 'vasc',
    // Short form fallbacks
    'akiec': 'akiec', 'bcc': 'bcc', 'bkl': 'bkl', 'df': 'df',
    'mel': 'mel', 'nv': 'nv', 'vasc': 'vasc',
};

// DINOv2 skin disease model (replacement for actavkid which is 404)
// Labels: Melanoma, Basal Cell Carcinoma, actinic keratosis,
//   pigmented benign keratosis, nevus, dermatofibroma, vascular lesion,
//   squamous cell carcinoma, seborrheic keratosis, etc.
const MODEL2_MAP = {
    'Melanoma': 'mel',
    'melanoma': 'mel',
    'Basal Cell Carcinoma': 'bcc',
    'basal cell carcinoma': 'bcc',
    'actinic keratosis': 'akiec',
    'squamous cell carcinoma': 'akiec',    // SCC -> akiec family
    'pigmented benign keratosis': 'bkl',
    'seborrheic keratosis': 'bkl',         // also benign keratosis family
    'benign keratosis': 'bkl',
    'nevus': 'nv',
    'melanocytic nevus': 'nv',
    'dermatofibroma': 'df',
    'vascular lesion': 'vasc',
    // Broad skin disease labels that don't map well (treated as benign)
    'Neurofibromatosis': 'nv',
    'Leprosy Lepromatous': 'df',
    'Epidermolysis Bullosa Pruriginosa': 'nv',
    // Short form fallbacks
    'akiec': 'akiec', 'bcc': 'bcc', 'bkl': 'bkl', 'df': 'df',
    'mel': 'mel', 'nv': 'nv', 'vasc': 'vasc',
};

// ---------- HuggingFace API call ----------
async function classifyWithHF(imageBuffer, modelUrl, apiKey, retries = 2) {
    const headers = { 'Content-Type': 'image/jpeg' };
    if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;

    for (let attempt = 0; attempt <= retries; attempt++) {
        try {
            const response = await fetch(modelUrl, {
                method: 'POST',
                headers,
                body: imageBuffer,
            });

            if (response.status === 503) {
                // Model is loading
                const body = await response.json().catch(() => ({}));
                const wait = body.estimated_time || 30;
                console.log(`  Model loading, waiting ${Math.ceil(wait)}s...`);
                await sleep(Math.ceil(wait) * 1000);
                continue;
            }

            if (response.status === 429) {
                // Rate limited
                const retryAfter = parseInt(response.headers.get('retry-after') || '5', 10);
                console.log(`  Rate limited, waiting ${retryAfter}s...`);
                await sleep(retryAfter * 1000);
                continue;
            }

            if (!response.ok) {
                const text = await response.text();
                return { error: `${response.status}: ${text.substring(0, 200)}`, results: [] };
            }

            const results = await response.json();
            return { results: Array.isArray(results) ? results : [], error: null };
        } catch (err) {
            if (attempt < retries) {
                await sleep(2000);
                continue;
            }
            return { error: err.message, results: [] };
        }
    }
    return { error: 'Max retries exceeded', results: [] };
}

// ---------- SvelteKit server API call ----------
async function classifyViaServer(imagePath, endpoint) {
    const imageBuffer = fs.readFileSync(imagePath);
    const blob = new Blob([imageBuffer], { type: 'image/jpeg' });
    const formData = new FormData();
    formData.append('image', blob, path.basename(imagePath));

    try {
        const response = await fetch(`http://localhost:5173${endpoint}`, {
            method: 'POST',
            body: formData,
        });
        if (!response.ok) {
            const text = await response.text();
            return { error: `${response.status}: ${text.substring(0, 200)}`, results: [] };
        }
        const data = await response.json();
        return data;
    } catch (err) {
        return { error: err.message, results: [] };
    }
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

function mapResults(results, labelMap) {
    const probs = {};
    CLASSES.forEach(c => probs[c] = 0);
    let unmapped = [];
    for (const r of (results || [])) {
        const label = r.label || r.Label || '';
        // Try exact match, then lowercase, then with underscores replaced
        const canonical = labelMap[label]
            || labelMap[label.toLowerCase()]
            || labelMap[label.replace(/_/g, ' ')]
            || labelMap[label.replace(/_/g, ' ').toLowerCase()];
        if (canonical) {
            probs[canonical] = Math.max(probs[canonical], r.score || 0);
        } else {
            unmapped.push(label);
        }
    }
    if (unmapped.length > 0) {
        // Only log once per unique unmapped label
        const unique = [...new Set(unmapped)];
        if (!mapResults._warned) mapResults._warned = new Set();
        for (const u of unique) {
            if (!mapResults._warned.has(u)) {
                console.log(`  WARNING: unmapped label "${u}"`);
                mapResults._warned.add(u);
            }
        }
    }
    // Normalize
    const total = Object.values(probs).reduce((a, b) => a + b, 0);
    if (total > 0) CLASSES.forEach(c => probs[c] /= total);
    return probs;
}

// ---------- Metrics ----------
function computeMetrics(confusion, totalValid, classes) {
    const metrics = {};
    for (const cls of classes) {
        const tp = confusion[cls][cls];
        const fn = Object.values(confusion[cls]).reduce((a, b) => a + b, 0) - tp;
        const fp = classes.reduce((a, c) => a + (c !== cls ? confusion[c][cls] : 0), 0);
        const tn = totalValid - tp - fn - fp;

        const sens = (tp + fn) > 0 ? tp / (tp + fn) : 0;
        const spec = (tn + fp) > 0 ? tn / (tn + fp) : 0;
        const ppv = (tp + fp) > 0 ? tp / (tp + fp) : 0;
        const npv = (tn + fn) > 0 ? tn / (tn + fn) : 0;

        metrics[cls] = { tp, fn, fp, tn, sens, spec, ppv, npv, n: tp + fn };
    }
    return metrics;
}

function printConfusionMatrix(confusion, classes) {
    console.log(`  ${''.padEnd(8)} ${classes.map(c => c.padEnd(8)).join('')}`);
    for (const actual of classes) {
        const row = classes.map(pred => String(confusion[actual][pred]).padEnd(8)).join('');
        console.log(`  ${actual.padEnd(8)} ${row}`);
    }
}

function printMetrics(metrics, classes) {
    console.log(`  ${'Class'.padEnd(8)} ${'Sens'.padEnd(8)} ${'Spec'.padEnd(8)} ${'PPV'.padEnd(8)} ${'NPV'.padEnd(8)} ${'N'.padEnd(6)}`);
    console.log(`  ${'-'.repeat(44)}`);
    for (const cls of classes) {
        const m = metrics[cls];
        console.log(
            `  ${cls.padEnd(8)} ` +
            `${(m.sens * 100).toFixed(1).padStart(5)}%  ` +
            `${(m.spec * 100).toFixed(1).padStart(5)}%  ` +
            `${(m.ppv * 100).toFixed(1).padStart(5)}%  ` +
            `${(m.npv * 100).toFixed(1).padStart(5)}%  ` +
            `${String(m.n).padEnd(6)}`
        );
    }
}

// ---------- Main ----------
async function main() {
    console.log('='.repeat(70));
    console.log('  DrAgnes Model Validation — HAM10000 Test Set');
    console.log('='.repeat(70));

    // Find HF token
    const apiKey = findHFToken();
    if (!apiKey) {
        console.error('\nERROR: No HuggingFace API key found.');
        console.error('Set HF_TOKEN in environment or in .env file.');
        process.exit(1);
    }
    console.log(`\nHF Token: ${apiKey.substring(0, 5)}...${apiKey.substring(apiKey.length - 4)} (${apiKey.length} chars)`);

    // Warm up models — send one test image to each to ensure they load
    console.log('\nWarming up models...');
    const warmupImg = fs.readFileSync(path.join(CACHE_DIR, 'train', 'nv', fs.readdirSync(path.join(CACHE_DIR, 'train', 'nv'))[0]));

    const [w1, w2] = await Promise.all([
        classifyWithHF(warmupImg, MODEL1_URL, apiKey, 3),
        classifyWithHF(warmupImg, MODEL2_URL, apiKey, 3),
    ]);

    if (w1.error) console.log(`  Model 1 (Anwarkh1) warmup error: ${w1.error}`);
    else {
        console.log(`  Model 1 (Anwarkh1) ready. Labels: ${w1.results.map(r => r.label).join(', ')}`);
    }
    if (w2.error) console.log(`  Model 2 (DINOv2) warmup error: ${w2.error}`);
    else {
        console.log(`  Model 2 (DINOv2) ready. Labels: ${w2.results.map(r => r.label).join(', ')}`);
    }

    // If both models failed warmup, try via server
    let useDirectHF = true;
    if (w1.error && w2.error) {
        console.log('\n  Direct HF calls failed, trying via localhost:5173 server...');
        const serverTest = await classifyViaServer(
            path.join(CACHE_DIR, 'train', 'nv', fs.readdirSync(path.join(CACHE_DIR, 'train', 'nv'))[0]),
            '/api/classify'
        );
        if (!serverTest.error && serverTest.results?.length > 0) {
            console.log('  Server proxy works! Using server endpoints.');
            useDirectHF = false;
        } else {
            console.error('\n  FATAL: Neither direct HF nor server proxy work. Cannot proceed.');
            console.error('  Server test result:', JSON.stringify(serverTest).substring(0, 300));
            process.exit(1);
        }
    }

    // Collect test images with ground truth labels
    let testImages = [];

    // Check for test split first (from previous training run)
    for (const cls of CLASSES) {
        const testPath = path.join(CACHE_DIR, 'test', cls);
        const trainPath = path.join(CACHE_DIR, 'train', cls);

        if (fs.existsSync(testPath)) {
            const files = fs.readdirSync(testPath).filter(f =>
                f.endsWith('.jpg') || f.endsWith('.png') || f.endsWith('.jpeg')
            );
            for (const file of files) {
                testImages.push({ path: path.join(testPath, file), label: cls });
            }
        } else if (fs.existsSync(trainPath)) {
            // No test split — use last 15% of train data
            const files = fs.readdirSync(trainPath)
                .filter(f => f.endsWith('.jpg') || f.endsWith('.png') || f.endsWith('.jpeg'))
                .sort(); // deterministic ordering
            const testStart = Math.floor(files.length * 0.85);
            for (const file of files.slice(testStart)) {
                testImages.push({ path: path.join(trainPath, file), label: cls });
            }
        }
    }

    // Also check flat structure
    if (testImages.length === 0) {
        for (const cls of CLASSES) {
            const clsPath = path.join(CACHE_DIR, cls);
            if (fs.existsSync(clsPath)) {
                const files = fs.readdirSync(clsPath)
                    .filter(f => f.endsWith('.jpg') || f.endsWith('.png'))
                    .sort();
                const testStart = Math.floor(files.length * 0.85);
                for (const file of files.slice(testStart)) {
                    testImages.push({ path: path.join(clsPath, file), label: cls });
                }
            }
        }
    }

    console.log(`\nFound ${testImages.length} test images`);
    console.log('Per class:', CLASSES.map(c =>
        `${c}: ${testImages.filter(i => i.label === c).length}`
    ).join(', '));

    if (testImages.length === 0) {
        console.error('No test images found! Check .cache/ham10000/ directory structure.');
        function listDir(dir, depth = 0) {
            if (depth > 2) return;
            try {
                const entries = fs.readdirSync(dir, { withFileTypes: true });
                for (const entry of entries.slice(0, 10)) {
                    console.log(' '.repeat(depth * 2) + (entry.isDirectory() ? '[DIR] ' : '') + entry.name);
                    if (entry.isDirectory()) listDir(path.join(dir, entry.name), depth + 1);
                }
                if (entries.length > 10) console.log(' '.repeat(depth * 2) + `... and ${entries.length - 10} more`);
            } catch { /* ignore */ }
        }
        listDir(CACHE_DIR);
        process.exit(1);
    }

    // Sample max 30 per class for rate limit safety
    const MAX_PER_CLASS = 30;
    const sampledImages = [];
    for (const cls of CLASSES) {
        const clsImages = testImages.filter(i => i.label === cls);
        // Use deterministic sort (by filename) then take first MAX_PER_CLASS
        // (these are the last 15% already, so they form a consistent test set)
        clsImages.sort((a, b) => path.basename(a.path).localeCompare(path.basename(b.path)));
        sampledImages.push(...clsImages.slice(0, MAX_PER_CLASS));
    }

    console.log(`\nSampled ${sampledImages.length} images (max ${MAX_PER_CLASS} per class)`);
    console.log('Per class:', CLASSES.map(c =>
        `${c}: ${sampledImages.filter(i => i.label === c).length}`
    ).join(', '));

    // ---------- Run both models on each image ----------
    const results = [];
    let completed = 0;
    let model1Errors = 0;
    let model2Errors = 0;
    const startTime = Date.now();

    console.log('\nRunning classifications...\n');

    for (const img of sampledImages) {
        const imageBuffer = fs.readFileSync(img.path);

        let result1, result2;

        if (useDirectHF) {
            // Call HF directly (faster, no server overhead)
            [result1, result2] = await Promise.all([
                classifyWithHF(imageBuffer, MODEL1_URL, apiKey),
                classifyWithHF(imageBuffer, MODEL2_URL, apiKey),
            ]);
        } else {
            // Call via SvelteKit server
            const [r1, r2] = await Promise.allSettled([
                classifyViaServer(img.path, '/api/classify'),
                classifyViaServer(img.path, '/api/classify-v2'),
            ]);
            result1 = r1.status === 'fulfilled' ? r1.value : { error: r1.reason?.message, results: [] };
            result2 = r2.status === 'fulfilled' ? r2.value : { error: r2.reason?.message, results: [] };
        }

        if (result1.error) model1Errors++;
        if (result2.error) model2Errors++;

        const probs1 = result1.results?.length ? mapResults(result1.results, ANWARKH1_MAP) : null;
        const probs2 = result2.results?.length ? mapResults(result2.results, MODEL2_MAP) : null;

        results.push({
            label: img.label,
            model1Probs: probs1,
            model2Probs: probs2,
            model1Top: probs1 ? CLASSES.reduce((a, b) => (probs1[a] > probs1[b] ? a : b)) : null,
            model2Top: probs2 ? CLASSES.reduce((a, b) => (probs2[a] > probs2[b] ? a : b)) : null,
            model1Raw: result1.results?.slice(0, 3) || [],
            model2Raw: result2.results?.slice(0, 3) || [],
        });

        completed++;
        if (completed % 10 === 0 || completed === sampledImages.length) {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
            const rate = (completed / (Date.now() - startTime) * 1000).toFixed(1);
            console.log(
                `  [${completed}/${sampledImages.length}] ` +
                `M1-err: ${model1Errors} | M2-err: ${model2Errors} | ` +
                `${elapsed}s elapsed | ${rate} img/s`
            );
        }

        // Delay to avoid rate limiting
        await sleep(250);
    }

    // ---------- Print results ----------
    console.log('\n' + '='.repeat(70));
    console.log('  MODEL VALIDATION RESULTS');
    console.log('='.repeat(70));

    for (const [modelName, topKey, probKey, rawKey, labelMap] of [
        ['Anwarkh1/Skin_Cancer-Image_Classification (ViT-Base, 85.8M)', 'model1Top', 'model1Probs', 'model1Raw', ANWARKH1_MAP],
        ['Jayanth2002/dinov2-base-finetuned-SkinDisease (DINOv2-Base, 86M)', 'model2Top', 'model2Probs', 'model2Raw', MODEL2_MAP],
    ]) {
        const validResults = results.filter(r => r[topKey] !== null);

        console.log(`\n${'='.repeat(70)}`);
        console.log(`  ${modelName}`);
        console.log('='.repeat(70));
        console.log(`  Valid predictions: ${validResults.length}/${results.length}`);
        console.log(`  Error rate: ${((results.length - validResults.length) / results.length * 100).toFixed(1)}%`);

        if (validResults.length === 0) {
            console.log('  NO VALID PREDICTIONS -- model may be unavailable or rate limited');
            // Show raw errors for debugging
            const errSample = results.find(r => r[topKey] === null);
            if (errSample) console.log('  Sample error:', JSON.stringify(errSample[rawKey]).substring(0, 200));
            continue;
        }

        // Show sample raw label mapping
        console.log(`\n  Sample raw labels from model:`);
        const sampleWithResults = results.find(r => r[rawKey]?.length > 0);
        if (sampleWithResults) {
            for (const r of sampleWithResults[rawKey]) {
                const mapped = labelMap[r.label] || labelMap[r.label?.toLowerCase()] || '???';
                console.log(`    "${r.label}" (score: ${r.score?.toFixed(4)}) -> ${mapped}`);
            }
        }

        // Confusion matrix
        const confusion = {};
        CLASSES.forEach(c => { confusion[c] = {}; CLASSES.forEach(c2 => confusion[c][c2] = 0); });

        let correct = 0;
        for (const r of validResults) {
            const predicted = r[topKey];
            const actual = r.label;
            confusion[actual][predicted]++;
            if (predicted === actual) correct++;
        }

        const accuracy = correct / validResults.length;
        console.log(`\n  Overall accuracy: ${(accuracy * 100).toFixed(1)}% (${correct}/${validResults.length})`);

        // Per-class metrics
        const metrics = computeMetrics(confusion, validResults.length, CLASSES);

        console.log(`\n  Per-class metrics:`);
        printMetrics(metrics, CLASSES);

        // Melanoma specifically
        const mel = metrics['mel'];
        console.log(`\n  *** MELANOMA SENSITIVITY: ${(mel.sens * 100).toFixed(1)}% (${mel.tp}/${mel.n}) ***`);
        console.log(`  *** MELANOMA SPECIFICITY: ${(mel.spec * 100).toFixed(1)}% ***`);
        console.log(`  *** MELANOMA PPV: ${(mel.ppv * 100).toFixed(1)}% ***`);
        console.log(`  *** MELANOMA NPV: ${(mel.npv * 100).toFixed(1)}% ***`);
        console.log(`  DermaSensor benchmark: sensitivity 95.5%, specificity 32.5%`);
        console.log(`  Status: ${mel.sens >= 0.90 ? 'PASS (>=90%)' : mel.sens >= 0.80 ? 'MARGINAL (80-90%)' : 'FAIL (<80%)'}`);

        // BCC (second most important malignancy)
        const bcc = metrics['bcc'];
        console.log(`\n  BCC SENSITIVITY: ${(bcc.sens * 100).toFixed(1)}% (${bcc.tp}/${bcc.n})`);

        // Confusion matrix
        console.log(`\n  Confusion Matrix (rows=actual, cols=predicted):`);
        printConfusionMatrix(confusion, CLASSES);
    }

    // ---------- Dual-model ensemble ----------
    console.log(`\n${'='.repeat(70)}`);
    console.log('  DUAL-MODEL ENSEMBLE');
    console.log('='.repeat(70));

    const dualResults = results.filter(r => r.model1Probs && r.model2Probs);
    console.log(`  Both models available: ${dualResults.length}/${results.length}`);

    if (dualResults.length > 0) {
        const dualConfusion = {};
        CLASSES.forEach(c => { dualConfusion[c] = {}; CLASSES.forEach(c2 => dualConfusion[c][c2] = 0); });

        let dualCorrect = 0;
        let agreements = 0;

        for (const r of dualResults) {
            // Ensemble strategy:
            // For melanoma: weight actavkid (larger model, published 89% recall) at 0.7
            // For other classes: weight Anwarkh1 (fine-tuned specifically on HAM10000) at 0.6
            const ensemble = {};
            for (const cls of CLASSES) {
                if (cls === 'mel') {
                    ensemble[cls] = r.model2Probs[cls] * 0.7 + r.model1Probs[cls] * 0.3;
                } else {
                    ensemble[cls] = r.model1Probs[cls] * 0.6 + r.model2Probs[cls] * 0.4;
                }
            }
            // Normalize
            const total = Object.values(ensemble).reduce((a, b) => a + b, 0);
            if (total > 0) CLASSES.forEach(c => ensemble[c] /= total);

            const predicted = CLASSES.reduce((a, b) => ensemble[a] > ensemble[b] ? a : b);
            dualConfusion[r.label][predicted]++;
            if (predicted === r.label) dualCorrect++;
            if (r.model1Top === r.model2Top) agreements++;
        }

        const dualAccuracy = dualCorrect / dualResults.length;
        console.log(`  Overall accuracy: ${(dualAccuracy * 100).toFixed(1)}% (${dualCorrect}/${dualResults.length})`);
        console.log(`  Model agreement: ${(agreements / dualResults.length * 100).toFixed(1)}%`);

        const dualMetrics = computeMetrics(dualConfusion, dualResults.length, CLASSES);

        console.log(`\n  Per-class metrics:`);
        printMetrics(dualMetrics, CLASSES);

        const melDual = dualMetrics['mel'];
        console.log(`\n  *** ENSEMBLE MELANOMA SENSITIVITY: ${(melDual.sens * 100).toFixed(1)}% (${melDual.tp}/${melDual.n}) ***`);
        console.log(`  *** ENSEMBLE MELANOMA SPECIFICITY: ${(melDual.spec * 100).toFixed(1)}% ***`);

        console.log(`\n  Confusion Matrix (rows=actual, cols=predicted):`);
        printConfusionMatrix(dualConfusion, CLASSES);
    }

    // ---------- Summary comparison ----------
    console.log(`\n${'='.repeat(70)}`);
    console.log('  COMPARISON vs DermaSensor BENCHMARKS');
    console.log('='.repeat(70));

    const model1Valid = results.filter(r => r.model1Top !== null);
    const model2Valid = results.filter(r => r.model2Top !== null);

    console.log(`\n  ${''.padEnd(24)} ${'Mel Sens'.padEnd(10)} ${'Mel Spec'.padEnd(10)} ${'Overall'.padEnd(10)} ${'N'.padEnd(6)}`);
    console.log(`  ${'-'.repeat(60)}`);

    // Model 1
    if (model1Valid.length > 0) {
        const c1 = {};
        CLASSES.forEach(c => { c1[c] = {}; CLASSES.forEach(c2 => c1[c][c2] = 0); });
        for (const r of model1Valid) c1[r.label][r.model1Top]++;
        const m1 = computeMetrics(c1, model1Valid.length, CLASSES);
        const acc1 = model1Valid.filter(r => r.model1Top === r.label).length / model1Valid.length;
        console.log(
            `  ${'Anwarkh1 ViT-Base'.padEnd(24)} ` +
            `${(m1.mel.sens * 100).toFixed(1).padStart(5)}%    ` +
            `${(m1.mel.spec * 100).toFixed(1).padStart(5)}%    ` +
            `${(acc1 * 100).toFixed(1).padStart(5)}%    ` +
            `${model1Valid.length}`
        );
    }

    // Model 2
    if (model2Valid.length > 0) {
        const c2 = {};
        CLASSES.forEach(c => { c2[c] = {}; CLASSES.forEach(c2_ => c2[c][c2_] = 0); });
        for (const r of model2Valid) c2[r.label][r.model2Top]++;
        const m2 = computeMetrics(c2, model2Valid.length, CLASSES);
        const acc2 = model2Valid.filter(r => r.model2Top === r.label).length / model2Valid.length;
        console.log(
            `  ${'DINOv2-Base SkinDis.'.padEnd(24)} ` +
            `${(m2.mel.sens * 100).toFixed(1).padStart(5)}%    ` +
            `${(m2.mel.spec * 100).toFixed(1).padStart(5)}%    ` +
            `${(acc2 * 100).toFixed(1).padStart(5)}%    ` +
            `${model2Valid.length}`
        );
    }

    // Ensemble
    if (dualResults.length > 0) {
        const ce = {};
        CLASSES.forEach(c => { ce[c] = {}; CLASSES.forEach(c2 => ce[c][c2] = 0); });
        for (const r of dualResults) {
            const ensemble = {};
            for (const cls of CLASSES) {
                if (cls === 'mel') {
                    ensemble[cls] = r.model2Probs[cls] * 0.7 + r.model1Probs[cls] * 0.3;
                } else {
                    ensemble[cls] = r.model1Probs[cls] * 0.6 + r.model2Probs[cls] * 0.4;
                }
            }
            const total = Object.values(ensemble).reduce((a, b) => a + b, 0);
            if (total > 0) CLASSES.forEach(c => ensemble[c] /= total);
            const predicted = CLASSES.reduce((a, b) => ensemble[a] > ensemble[b] ? a : b);
            ce[r.label][predicted]++;
        }
        const me = computeMetrics(ce, dualResults.length, CLASSES);
        const accE = dualResults.filter(r => {
            const ensemble = {};
            for (const cls of CLASSES) {
                if (cls === 'mel') {
                    ensemble[cls] = r.model2Probs[cls] * 0.7 + r.model1Probs[cls] * 0.3;
                } else {
                    ensemble[cls] = r.model1Probs[cls] * 0.6 + r.model2Probs[cls] * 0.4;
                }
            }
            const total = Object.values(ensemble).reduce((a, b) => a + b, 0);
            if (total > 0) CLASSES.forEach(c => ensemble[c] /= total);
            const predicted = CLASSES.reduce((a, b) => ensemble[a] > ensemble[b] ? a : b);
            return predicted === r.label;
        }).length / dualResults.length;
        console.log(
            `  ${'Dual-Model Ensemble'.padEnd(24)} ` +
            `${(me.mel.sens * 100).toFixed(1).padStart(5)}%    ` +
            `${(me.mel.spec * 100).toFixed(1).padStart(5)}%    ` +
            `${(accE * 100).toFixed(1).padStart(5)}%    ` +
            `${dualResults.length}`
        );
    }

    console.log(
        `  ${'DermaSensor (benchmark)'.padEnd(24)} ` +
        `${' 95.5'}%    ` +
        `${' 32.5'}%    ` +
        `${'   --'}     ` +
        `--`
    );

    console.log(`\n${'='.repeat(70)}`);
    console.log(`  Total elapsed: ${((Date.now() - startTime) / 1000).toFixed(1)}s`);
    console.log('='.repeat(70));

    // Save raw results to JSON for later analysis
    const outputPath = path.join(PROJECT_DIR, 'scripts', 'validation-results.json');
    fs.writeFileSync(outputPath, JSON.stringify({
        timestamp: new Date().toISOString(),
        totalImages: sampledImages.length,
        model1Errors,
        model2Errors,
        results: results.map(r => ({
            label: r.label,
            model1Top: r.model1Top,
            model2Top: r.model2Top,
            model1Probs: r.model1Probs,
            model2Probs: r.model2Probs,
        })),
    }, null, 2));
    console.log(`\nRaw results saved to: ${outputPath}`);
}

main().catch(err => {
    console.error('FATAL ERROR:', err);
    process.exit(1);
});
