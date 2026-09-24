// ---------------------------------------------------------------------
// script.js
// Handles: threshold slider sync, live metrics fetch, and single
// transaction prediction — all without ever reloading the page.
// ---------------------------------------------------------------------

const thresholdSlider = document.getElementById("threshold");
const thresholdNumber = document.getElementById("threshold-number");
const thresholdValueLabel = document.getElementById("threshold-value");
const form = document.getElementById("transaction-form");

const predictionResult = document.getElementById("prediction-result");
const predictionLabel = document.getElementById("prediction-label");
const predictionProbability = document.getElementById("prediction-probability");

const accuracyValue = document.getElementById("accuracy-value");
const recallValue = document.getElementById("recall-value");
const cmTN = document.getElementById("cm-tn");
const cmFP = document.getElementById("cm-fp");
const cmFN = document.getElementById("cm-fn");
const cmTP = document.getElementById("cm-tp");
const testSetSize = document.getElementById("test-set-size");

let debounceTimer = null;

function getThreshold() {
  return parseFloat(thresholdSlider.value);
}

function syncThresholdInputs(value) {
  thresholdSlider.value = value;
  thresholdNumber.value = value;
  thresholdValueLabel.textContent = parseFloat(value).toFixed(2);
}

// ---------------------------------------------------------------------
// Fetch accuracy / recall / confusion matrix for the current threshold
// ---------------------------------------------------------------------
async function refreshMetrics() {
  const threshold = getThreshold();

  try {
    const response = await fetch("/metrics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ threshold }),
    });

    const data = await response.json();

    if (!response.ok) {
      console.error(data.error);
      return;
    }

    accuracyValue.textContent = (data.accuracy * 100).toFixed(2) + "%";
    recallValue.textContent = (data.recall * 100).toFixed(2) + "%";

    cmTN.textContent = data.confusion_matrix.tn;
    cmFP.textContent = data.confusion_matrix.fp;
    cmFN.textContent = data.confusion_matrix.fn;
    cmTP.textContent = data.confusion_matrix.tp;

    testSetSize.textContent = `Computed on ${data.test_set_size} held-out test transactions.`;
  } catch (err) {
    console.error("Failed to fetch metrics:", err);
  }
}

// Debounce so we don't spam the server while the user is dragging the slider
function scheduleMetricsRefresh() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(refreshMetrics, 150);
}

thresholdSlider.addEventListener("input", () => {
  syncThresholdInputs(thresholdSlider.value);
  scheduleMetricsRefresh();
});

thresholdNumber.addEventListener("input", () => {
  let val = parseFloat(thresholdNumber.value);
  if (isNaN(val)) return;
  val = Math.min(1, Math.max(0, val));
  syncThresholdInputs(val);
  scheduleMetricsRefresh();
});

// ---------------------------------------------------------------------
// Predict a single transaction entered in the form
// ---------------------------------------------------------------------
form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    payload[key] = value;
  }
  payload.threshold = getThreshold();

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      alert(data.error || "Something went wrong.");
      return;
    }

    predictionResult.classList.remove("hidden");
    predictionLabel.textContent = data.prediction === "Fraud"
      ? "⚠️ FRAUD DETECTED"
      : "✅ LEGITIMATE TRANSACTION";
    predictionLabel.className = data.prediction === "Fraud" ? "fraud" : "legit";

    predictionProbability.textContent =
      `Fraud probability: ${(data.probability * 100).toFixed(2)}%  ` +
      `(threshold used: ${data.threshold_used.toFixed(2)})`;
  } catch (err) {
    console.error("Prediction failed:", err);
    alert("Could not reach the server. Is app.py running?");
  }
});

// ---------------------------------------------------------------------
// Initial load
// ---------------------------------------------------------------------
syncThresholdInputs(thresholdSlider.value);
refreshMetrics();
