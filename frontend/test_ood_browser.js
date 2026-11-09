import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { computeMahalanobisDistance, checkFamiliarity, areInputsComplete } from './src/utils/oodBrowser.js';

// Resolve and read ood_browser.json using standard node fs for 100% plain node compatibility
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const jsonPath = path.resolve(__dirname, 'src/data/ood_browser.json');
const oodData = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));

console.log('================================================================');
console.log('Crop Trust Engine - In-Browser Familiarity (OOD) Test Runner');
console.log('Testing 6 test cases from ood_browser.json under plain Node.js');
console.log('================================================================\n');

let allPassed = true;
const TOLERANCE = 1e-4; // Small tolerance for floating point precision

oodData.test_cases.forEach((tc, index) => {
  const caseNum = index + 1;
  const { minD2, nearestCrop } = computeMahalanobisDistance(tc.input, oodData);
  const diff = Math.abs(minD2 - tc.min_d2);
  const distMatches = diff <= TOLERANCE;
  const cropMatches = nearestCrop === tc.nearest_crop;

  // Check status and flagging across risk profiles
  const balancedResult = checkFamiliarity(tc.input, 'balanced', oodData);
  const conservativeResult = checkFamiliarity(tc.input, 'conservative', oodData);
  const boldResult = checkFamiliarity(tc.input, 'bold', oodData);

  const passed = distMatches && cropMatches;
  if (!passed) allPassed = false;

  const resultStatus = passed ? 'PASS' : 'FAIL';
  console.log(`${resultStatus} Case ${caseNum}:`);
  console.log(`  - Expected d²:     ${tc.min_d2.toFixed(6)}`);
  console.log(`  - Computed d²:     ${minD2.toFixed(6)} (diff: ${diff.toExponential(3)})`);
  console.log(`  - Nearest crop:    ${nearestCrop} (expected: ${tc.nearest_crop})`);
  console.log(`  - Status [Balanced]:     ${balancedResult.icon} ${balancedResult.label} (${balancedResult.status})`);
  console.log(`  - Status [Conservative]: ${conservativeResult.icon} ${conservativeResult.label} (${conservativeResult.status})`);
  console.log(`  - Status [Bold]:         ${boldResult.icon} ${boldResult.label} (${boldResult.status})\n`);
});

// Also test the incomplete input check (Requirement 3)
console.log('Checking validation for incomplete / non-numeric inputs...');
const incompleteCases = [
  { N: '', P: 50, K: 50, temperature: 25, humidity: 80, ph: 6.5, rainfall: 100 },
  { N: 50, P: null, K: 50, temperature: 25, humidity: 80, ph: 6.5, rainfall: 100 },
  { N: 50, P: 50, K: 'abc', temperature: 25, humidity: 80, ph: 6.5, rainfall: 100 }
];

let validationPassed = true;
incompleteCases.forEach((ic, idx) => {
  const res = checkFamiliarity(ic, 'balanced', oodData);
  if (res.status !== 'incomplete' || res.label !== 'Enter all 7 values') {
    validationPassed = false;
  }
});

if (validationPassed) {
  console.log('PASS Incomplete inputs validation: returns "Enter all 7 values" when inputs are missing or NaN.\n');
} else {
  allPassed = false;
  console.log('FAIL Incomplete inputs validation\n');
}

console.log('================================================================');
if (allPassed) {
  console.log('ALL TESTS PASSED (6/6 OOD test cases + incomplete input validation)');
  process.exit(0);
} else {
  console.error('TESTS FAILED');
  process.exit(1);
}

