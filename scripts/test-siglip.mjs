/**
 * DrAgnes SigLIP / SwinV2 / ViT-Melanoma Model Benchmark
 *
 * Tests three dermatology-specific models against HAM10000 test images:
 *   1. skintaglabs/siglip-skin-lesion-classifier  (SigLIP 400M, MIT)
 *   2. TriDat/swinv2-base-patch4-window12-192-22k-finetuned-lora-ISIC-2019  (SwinV2, ISIC-2019)
 *   3. UnipaPolitoUnimore/vit-large-patch32-384-melanoma  (ViT-Large melanoma-focused)
 *
 * Usage: node scripts/test-siglip.mjs
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const CACHE_DIR = path.join(PROJECT_DIR, '.cache', 'ham10000');
const CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'];
const MAX_PER_CLASS = 30;
const DELAY_MS = 300;

// -------- Models --------
const MODELS = [
    {
        id: 'skintaglabs/siglip-skin-lesion-classifier',
        shortName: 'SigLIP-SkinTag',
        url: 'https://router.huggingface.co/hf-inference/models/skintaglabs/siglip-skin-lesion-classifier',
        labelMap: {
            // SigLIP zero-shot or fine-tuned labels - map common derm labels
            // Exact labels unknown a-priori; we will discover and map them
            'melanoma': 'mel',
            'Melanoma': 'mel',
            'mel': 'mel',
            'MEL': 'mel',
            'basal cell carcinoma': 'bcc',
            'Basal Cell Carcinoma': 'bcc',
            'BCC': 'bcc',
            'bcc': 'bcc',
            'basal_cell_carcinoma': 'bcc',
            'actinic keratosis': 'akiec',
            'Actinic Keratosis': 'akiec',
            'actinic keratoses': 'akiec',
            'akiec': 'akiec',
            'squamous cell carcinoma': 'akiec',
            'Squamous Cell Carcinoma': 'akiec',
            'benign keratosis': 'bkl',
            'Benign Keratosis': 'bkl',
            'benign keratosis-like lesions': 'bkl',
            'benign_keratosis-like_lesions': 'bkl',
            'seborrheic keratosis': 'bkl',
            'Seborrheic Keratosis': 'bkl',
            'solar lentigo': 'bkl',
            'bkl': 'bkl',
            'pigmented benign keratosis': 'bkl',
            'dermatofibroma': 'df',
            'Dermatofibroma': 'df',
            'df': 'df',
            'melanocytic nevi': 'nv',
            'melanocytic nevus': 'nv',
            'Melanocytic Nevi': 'nv',
            'melanocytic_nevi': 'nv',
            'melanocytic_Nevi': 'nv',
            'nevus': 'nv',
            'Nevus': 'nv',
            'nv': 'nv',
            'NV': 'nv',
            'vascular lesion': 'vasc',
            'vascular lesions': 'vasc',
            'Vascular Lesion': 'vasc',
            'Vascular Lesions': 'vasc',
            'vascular_lesions': 'vasc',
            'vasc': 'vasc',
            // ISIC-style labels
            'AKIEC': 'akiec',
            'BKL': 'bkl',
            'DF': 'df',
            'VASC': 'vasc',
        },
    },
    {
        id: 'TriDat/swinv2-base-patch4-window12-192-22k-finetuned-lora-ISIC-2019',
        shortName: 'SwinV2-ISIC2019',
        url: 'https://router.huggingface.co/hf-inference/models/TriDat/swinv2-base-patch4-window12-192-22k-finetuned-lora-ISIC-2019',
        labelMap: {
            // ISIC-2019 has 8 classes: MEL, NV, BCC, AK, BKL, DF, VASC, SCC
            // Model may use various label formats
            'MEL': 'mel', 'mel': 'mel', 'Melanoma': 'mel', 'melanoma': 'mel',
            'NV': 'nv', 'nv': 'nv', 'Melanocytic nevus': 'nv', 'melanocytic nevus': 'nv', 'Nevus': 'nv', 'nevus': 'nv',
            'BCC': 'bcc', 'bcc': 'bcc', 'Basal cell carcinoma': 'bcc', 'basal cell carcinoma': 'bcc',
            'AK': 'akiec', 'ak': 'akiec', 'Actinic keratosis': 'akiec', 'actinic keratosis': 'akiec',
            'AKIEC': 'akiec', 'akiec': 'akiec',
            'BKL': 'bkl', 'bkl': 'bkl', 'Benign keratosis': 'bkl', 'benign keratosis': 'bkl',
            'DF': 'df', 'df': 'df', 'Dermatofibroma': 'df', 'dermatofibroma': 'df',
            'VASC': 'vasc', 'vasc': 'vasc', 'Vascular lesion': 'vasc', 'vascular lesion': 'vasc',
            'SCC': 'akiec', 'scc': 'akiec', 'Squamous cell carcinoma': 'akiec', 'squamous cell carcinoma': 'akiec',
            // Possible label_ prefix or numeric labels
            'LABEL_0': 'akiec', 'LABEL_1': 'bcc', 'LABEL_2': 'bkl', 'LABEL_3': 'df',
            'LABEL_4': 'mel', 'LABEL_5': 'nv', 'LABEL_6': 'vasc', 'LABEL_7': 'akiec', // SCC -> akiec
        },
    },
    {
        id: 'UnipaPolitoUnimore/vit-large-patch32-384-melanoma',
        shortName: 'ViT-L-Melanoma',
        url: 'https://router.huggingface.co/hf-inference/models/UnipaPolitoUnimore/vit-large-patch32-384-melanoma',
        labelMap: {
            // Binary melanoma classifier: melanoma vs benign
            // May return labels like 'melanoma' / 'benign' or 'LABEL_0' / 'LABEL_1'
            'melanoma': 'mel', 'Melanoma': 'mel', 'MELANOMA': 'mel',
            'malignant': 'mel', 'Malignant': 'mel',
            'benign': '_benign', 'Benign': '_benign', 'BENIGN': '_benign',
            'not melanoma': '_benign', 'Not Melanoma': '_benign',
            // Numeric labels - we will discover mapping on warmup
            'LABEL_0': '_benign', 'LABEL_1': 'mel',
        },
        isBinary: true, // special handling: only measures mel sensitivity
    },
];

// -------- Resolve HF token --------
function findHFToken() {
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

// -------- HF API call --------
function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

async function classifyWithHF(imageBuffer, modelUrl, apiKey, retries = 3) {
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
                const body = await response.json().catch(() => ({}));
                const wait = Math.min(body.estimated_time || 30, 60);
                console.log(`    Model loading (503), waiting ${Math.ceil(wait)}s... (attempt ${attempt + 1}/${retries + 1})`);
                await sleep(Math.ceil(wait) * 1000);
                continue;
            }

            if (response.status === 429) {
                const retryAfter = parseInt(response.headers.get('retry-after') || '10', 10);
                console.log(`    Rate limited (429), waiting ${retryAfter}s...`);
                await sleep(retryAfter * 1000);
                continue;
            }

            if (!response.ok) {
                const text = await response.text();
                return { error: `${response.status}: ${text.substring(0, 300)}`, results: [] };
            }

            const results = await response.json();
            return { results: Array.isArray(results) ? results : [], error: null };
        } catch (err) {
            if (attempt < retries) {
                await sleep(3000);
                continue;
            }
            return { error: err.message, results: [] };
        }
    }
    return { error: 'Max retries exceeded', results: [] };
}

// -------- Label mapping --------
const _warnedLabels = new Set();

function mapLabel(results, labelMap) {
    if (!results || results.length === 0) return null;

    // Find the top-scoring result that maps to a known class
    let bestScore = -1;
    let bestClass = null;

    const probs = {};
    CLASSES.forEach(c => probs[c] = 0);

    for (const r of results) {
        const label = r.label || r.Label || '';
        const score = r.score || 0;

        // Try multiple normalizations
        const canonical =
            labelMap[label] ||
            labelMap[label.toLowerCase()] ||
            labelMap[label.replace(/_/g, ' ')] ||
            labelMap[label.replace(/_/g, ' ').toLowerCase()] ||
            labelMap[label.trim()];

        if (canonical && !canonical.startsWith('_')) {
            probs[canonical] = Math.max(probs[canonical], score);
            if (score > bestScore) {
                bestScore = score;
                bestClass = canonical;
            }
        } else if (canonical && canonical.startsWith('_')) {
            // Special markers like _benign - track but don't assign to a HAM class
            if (score > bestScore) {
                bestScore = score;
                bestClass = canonical;
            }
        } else {
            if (!_warnedLabels.has(label)) {
                console.log(`    WARNING: unmapped label "${label}" (score: ${score.toFixed(4)})`);
                _warnedLabels.add(label);
            }
        }
    }

    return { topClass: bestClass, probs, topScore: bestScore };
}

// -------- Metrics --------
function computeMetrics(confusion, totalValid, classes) {
    const metrics = {};
    for (const cls of classes) {
        const tp = confusion[cls]?.[cls] || 0;
        const fn = Object.values(confusion[cls] || {}).reduce((a, b) => a + b, 0) - tp;
        const fp = classes.reduce((a, c) => a + (c !== cls ? (confusion[c]?.[cls] || 0) : 0), 0);
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
        const row = classes.map(pred => String(confusion[actual]?.[pred] || 0).padEnd(8)).join('');
        console.log(`  ${actual.padEnd(8)} ${row}`);
    }
}

// -------- Collect test images --------
function collectTestImages() {
    const testImages = [];

    for (const cls of CLASSES) {
        const testPath = path.join(CACHE_DIR, 'test', cls);
        const trainPath = path.join(CACHE_DIR, 'train', cls);

        if (fs.existsSync(testPath)) {
            const files = fs.readdirSync(testPath)
                .filter(f => /\.(jpg|jpeg|png)$/i.test(f))
                .sort();
            for (const file of files) {
                testImages.push({ path: path.join(testPath, file), label: cls });
            }
        } else if (fs.existsSync(trainPath)) {
            // No test split - use last 15%
            const files = fs.readdirSync(trainPath)
                .filter(f => /\.(jpg|jpeg|png)$/i.test(f))
                .sort();
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
                    .filter(f => /\.(jpg|jpeg|png)$/i.test(f))
                    .sort();
                const testStart = Math.floor(files.length * 0.85);
                for (const file of files.slice(testStart)) {
                    testImages.push({ path: path.join(clsPath, file), label: cls });
                }
            }
        }
    }

    return testImages;
}

function sampleImages(testImages) {
    const sampled = [];
    for (const cls of CLASSES) {
        const clsImages = testImages.filter(i => i.label === cls);
        clsImages.sort((a, b) => path.basename(a.path).localeCompare(path.basename(b.path)));
        sampled.push(...clsImages.slice(0, MAX_PER_CLASS));
    }
    return sampled;
}

// -------- Run one model --------
async function runModel(model, images, apiKey) {
    console.log(`\n  Running ${model.shortName} (${images.length} images)...`);
    const startTime = Date.now();
    const results = [];
    let errors = 0;

    // Clear warning cache for each model
    _warnedLabels.clear();

    for (let i = 0; i < images.length; i++) {
        const img = images[i];
        const imageBuffer = fs.readFileSync(img.path);
        const result = await classifyWithHF(imageBuffer, model.url, apiKey);

        if (result.error) {
            errors++;
            results.push({ label: img.label, predicted: null, raw: [], error: result.error });
        } else {
            const mapped = mapLabel(result.results, model.labelMap);
            results.push({
                label: img.label,
                predicted: mapped?.topClass || null,
                topScore: mapped?.topScore || 0,
                probs: mapped?.probs || null,
                raw: result.results?.slice(0, 5) || [],
                error: null,
            });
        }

        if ((i + 1) % 20 === 0 || i === images.length - 1) {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
            console.log(`    [${i + 1}/${images.length}] errors: ${errors} | ${elapsed}s elapsed`);
        }

        // Rate limit delay
        if (i < images.length - 1) {
            await sleep(DELAY_MS);
        }
    }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`    Done in ${elapsed}s (${errors} errors)`);

    return results;
}

// -------- Print report for one model --------
function printReport(model, results) {
    console.log(`\n--- ${model.id} ---`);

    // Show sample raw labels for debugging
    const sampleWithResults = results.find(r => r.raw?.length > 0);
    if (sampleWithResults) {
        console.log(`  Sample raw labels from API:`);
        for (const r of sampleWithResults.raw.slice(0, 7)) {
            const mapped = model.labelMap[r.label] || model.labelMap[r.label?.toLowerCase()] || '???';
            console.log(`    "${r.label}" -> ${mapped} (score: ${(r.score || 0).toFixed(4)})`);
        }
    }

    if (model.isBinary) {
        // Binary melanoma classifier: only mel sensitivity/specificity
        printBinaryReport(model, results);
        return;
    }

    // Filter valid predictions (mapped to a real HAM class)
    const valid = results.filter(r => r.predicted && CLASSES.includes(r.predicted));
    console.log(`  Valid predictions: ${valid.length}/${results.length}`);

    if (valid.length === 0) {
        console.log(`  NO VALID PREDICTIONS - model may be unavailable or labels could not be mapped`);
        const errSample = results.find(r => r.error);
        if (errSample) console.log(`  Sample error: ${errSample.error?.substring(0, 300)}`);
        console.log(`\n  *** MELANOMA SENSITIVITY: N/A ***`);
        console.log(`  Status: FAIL (no data)`);
        return;
    }

    // Overall accuracy
    const correct = valid.filter(r => r.predicted === r.label).length;
    const accuracy = (correct / valid.length * 100).toFixed(1);
    console.log(`  Overall accuracy: ${accuracy}%`);

    // Build confusion matrix
    const confusion = {};
    CLASSES.forEach(c => { confusion[c] = {}; CLASSES.forEach(c2 => confusion[c][c2] = 0); });
    for (const r of valid) {
        confusion[r.label][r.predicted]++;
    }

    // Per-class metrics
    const metrics = computeMetrics(confusion, valid.length, CLASSES);

    console.log(`\n  Per-class metrics:`);
    console.log(`  ${'Class'.padEnd(8)} ${'Sens'.padEnd(8)} ${'Spec'.padEnd(8)} ${'N'.padEnd(6)}`);
    console.log(`  ${'-'.repeat(28)}`);
    for (const cls of CLASSES) {
        const m = metrics[cls];
        console.log(
            `  ${cls.padEnd(8)} ` +
            `${(m.sens * 100).toFixed(1).padStart(5)}%  ` +
            `${(m.spec * 100).toFixed(1).padStart(5)}%  ` +
            `${String(m.n).padEnd(6)}`
        );
    }

    // Melanoma headline
    const mel = metrics['mel'];
    const melSens = (mel.sens * 100).toFixed(1);
    const status = mel.sens >= 0.90 ? 'PASS (>=90%)' : mel.sens >= 0.80 ? 'MARGINAL (80-90%)' : 'FAIL (<80%)';
    console.log(`\n  *** MELANOMA SENSITIVITY: ${melSens}% (${mel.tp}/${mel.n}) ***`);
    console.log(`  *** MELANOMA SPECIFICITY: ${(mel.spec * 100).toFixed(1)}% ***`);
    console.log(`  Status: ${status}`);

    // Confusion matrix
    console.log(`\n  Confusion Matrix (rows=actual, cols=predicted):`);
    printConfusionMatrix(confusion, CLASSES);
}

function printBinaryReport(model, results) {
    // Binary classifier: mel vs not-mel
    // Predictions can be 'mel' or '_benign' or null
    const valid = results.filter(r => r.predicted !== null);
    console.log(`  Valid predictions: ${valid.length}/${results.length}`);

    if (valid.length === 0) {
        console.log(`  NO VALID PREDICTIONS`);
        console.log(`\n  *** MELANOMA SENSITIVITY: N/A ***`);
        console.log(`  Status: FAIL (no data)`);
        return;
    }

    // For binary: predicted is 'mel' or '_benign'
    let tp = 0, fn = 0, fp = 0, tn = 0;
    for (const r of valid) {
        const actualMel = r.label === 'mel';
        const predictedMel = r.predicted === 'mel';

        if (actualMel && predictedMel) tp++;
        else if (actualMel && !predictedMel) fn++;
        else if (!actualMel && predictedMel) fp++;
        else tn++;
    }

    const melSens = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const melSpec = (tn + fp) > 0 ? tn / (tn + fp) : 0;

    const melTotal = tp + fn;
    const nonMelTotal = tn + fp;

    console.log(`\n  Binary classification (melanoma vs benign):`);
    console.log(`  Melanoma images: ${melTotal} | Non-melanoma images: ${nonMelTotal}`);
    console.log(`  TP: ${tp} | FN: ${fn} | FP: ${fp} | TN: ${tn}`);

    // Per-class breakdown (which classes get misclassified as mel?)
    console.log(`\n  Per-class predicted-as-melanoma rate:`);
    console.log(`  ${'Class'.padEnd(8)} ${'PredMel'.padEnd(10)} ${'PredBen'.padEnd(10)} ${'MelRate'.padEnd(10)} ${'N'.padEnd(6)}`);
    console.log(`  ${'-'.repeat(44)}`);
    for (const cls of CLASSES) {
        const clsResults = valid.filter(r => r.label === cls);
        const predMel = clsResults.filter(r => r.predicted === 'mel').length;
        const predBen = clsResults.length - predMel;
        const melRate = clsResults.length > 0 ? (predMel / clsResults.length * 100).toFixed(1) : '0.0';
        console.log(
            `  ${cls.padEnd(8)} ` +
            `${String(predMel).padStart(4)}      ` +
            `${String(predBen).padStart(4)}      ` +
            `${melRate.padStart(5)}%    ` +
            `${clsResults.length}`
        );
    }

    const status = melSens >= 0.90 ? 'PASS (>=90%)' : melSens >= 0.80 ? 'MARGINAL (80-90%)' : 'FAIL (<80%)';
    console.log(`\n  *** MELANOMA SENSITIVITY: ${(melSens * 100).toFixed(1)}% (${tp}/${melTotal}) ***`);
    console.log(`  *** MELANOMA SPECIFICITY: ${(melSpec * 100).toFixed(1)}% ***`);
    console.log(`  Status: ${status}`);
}

// -------- Main --------
async function main() {
    console.log('='.repeat(70));
    console.log('  DrAgnes Model Benchmark: SigLIP / SwinV2 / ViT-Melanoma');
    console.log('='.repeat(70));

    // Find HF token
    const apiKey = findHFToken();
    if (!apiKey) {
        console.error('\nERROR: No HuggingFace API key found.');
        console.error('Set HF_TOKEN env var, or put HuggingFace_Key in /Users/stuartkerr/RuVector_New/RuVector/.env');
        process.exit(1);
    }
    console.log(`\nHF Token: ${apiKey.substring(0, 5)}...${apiKey.substring(apiKey.length - 4)} (${apiKey.length} chars)`);

    // Collect and sample test images
    const allTestImages = collectTestImages();
    console.log(`\nFound ${allTestImages.length} test images (last 15% of each class)`);
    console.log('Per class:', CLASSES.map(c =>
        `${c}: ${allTestImages.filter(i => i.label === c).length}`
    ).join(', '));

    if (allTestImages.length === 0) {
        console.error('No test images found! Check .cache/ham10000/ directory.');
        process.exit(1);
    }

    const sampledImages = sampleImages(allTestImages);
    console.log(`\nSampled ${sampledImages.length} images (max ${MAX_PER_CLASS} per class)`);
    console.log('Per class:', CLASSES.map(c =>
        `${c}: ${sampledImages.filter(i => i.label === c).length}`
    ).join(', '));

    // Warm up all models with one test image
    console.log('\nWarming up models...');
    const warmupBuffer = fs.readFileSync(sampledImages[0].path);

    for (const model of MODELS) {
        console.log(`  Warming up ${model.shortName}...`);
        const w = await classifyWithHF(warmupBuffer, model.url, apiKey, 3);
        if (w.error) {
            console.log(`    Warmup error: ${w.error}`);
        } else {
            const labels = w.results.map(r => `"${r.label}" (${(r.score || 0).toFixed(4)})`).join(', ');
            console.log(`    Ready. Labels: ${labels}`);
        }
    }

    // Run each model
    const allResults = {};
    const totalStart = Date.now();

    for (const model of MODELS) {
        allResults[model.id] = await runModel(model, sampledImages, apiKey);
    }

    // Print reports
    console.log('\n' + '='.repeat(70));
    console.log('  RESULTS');
    console.log('='.repeat(70));

    for (const model of MODELS) {
        printReport(model, allResults[model.id]);
    }

    // Summary comparison
    console.log(`\n${'='.repeat(70)}`);
    console.log('  SUMMARY COMPARISON');
    console.log('='.repeat(70));
    console.log(`\n  ${'Model'.padEnd(28)} ${'MelSens'.padEnd(10)} ${'MelSpec'.padEnd(10)} ${'Accuracy'.padEnd(10)} ${'Valid/N'.padEnd(10)}`);
    console.log(`  ${'-'.repeat(68)}`);

    for (const model of MODELS) {
        const results = allResults[model.id];

        if (model.isBinary) {
            const valid = results.filter(r => r.predicted !== null);
            let tp = 0, fn = 0, fp = 0, tn = 0;
            for (const r of valid) {
                const actualMel = r.label === 'mel';
                const predictedMel = r.predicted === 'mel';
                if (actualMel && predictedMel) tp++;
                else if (actualMel && !predictedMel) fn++;
                else if (!actualMel && predictedMel) fp++;
                else tn++;
            }
            const melSens = (tp + fn) > 0 ? (tp / (tp + fn) * 100).toFixed(1) : 'N/A';
            const melSpec = (tn + fp) > 0 ? (tn / (tn + fp) * 100).toFixed(1) : 'N/A';
            console.log(
                `  ${model.shortName.padEnd(28)} ` +
                `${String(melSens).padStart(5)}%    ` +
                `${String(melSpec).padStart(5)}%    ` +
                `${'(binary)'.padEnd(10)} ` +
                `${valid.length}/${results.length}`
            );
        } else {
            const valid = results.filter(r => r.predicted && CLASSES.includes(r.predicted));
            if (valid.length === 0) {
                console.log(`  ${model.shortName.padEnd(28)} ${'N/A'.padStart(5)}     ${'N/A'.padStart(5)}     ${'N/A'.padEnd(10)} 0/${results.length}`);
                continue;
            }
            const confusion = {};
            CLASSES.forEach(c => { confusion[c] = {}; CLASSES.forEach(c2 => confusion[c][c2] = 0); });
            for (const r of valid) confusion[r.label][r.predicted]++;
            const metrics = computeMetrics(confusion, valid.length, CLASSES);
            const correct = valid.filter(r => r.predicted === r.label).length;
            const accuracy = (correct / valid.length * 100).toFixed(1);
            const melSens = (metrics.mel.sens * 100).toFixed(1);
            const melSpec = (metrics.mel.spec * 100).toFixed(1);
            console.log(
                `  ${model.shortName.padEnd(28)} ` +
                `${melSens.padStart(5)}%    ` +
                `${melSpec.padStart(5)}%    ` +
                `${(accuracy + '%').padEnd(10)} ` +
                `${valid.length}/${results.length}`
            );
        }
    }
    console.log(`  ${'DermaSensor (benchmark)'.padEnd(28)} ${'95.5'.padStart(5)}%    ${'32.5'.padStart(5)}%    ${'--'.padEnd(10)} --`);

    const totalElapsed = ((Date.now() - totalStart) / 1000).toFixed(1);
    console.log(`\n  Total elapsed: ${totalElapsed}s`);
    console.log('='.repeat(70));

    // Save raw results
    const outputPath = path.join(PROJECT_DIR, 'scripts', 'siglip-test-results.json');
    const output = {};
    for (const model of MODELS) {
        output[model.id] = {
            shortName: model.shortName,
            results: allResults[model.id].map(r => ({
                label: r.label,
                predicted: r.predicted,
                topScore: r.topScore,
                raw: r.raw?.slice(0, 3) || [],
            })),
        };
    }
    fs.writeFileSync(outputPath, JSON.stringify({
        timestamp: new Date().toISOString(),
        totalImages: sampledImages.length,
        models: output,
    }, null, 2));
    console.log(`\nRaw results saved to: ${outputPath}`);
}

main().catch(err => {
    console.error('FATAL ERROR:', err);
    process.exit(1);
});
