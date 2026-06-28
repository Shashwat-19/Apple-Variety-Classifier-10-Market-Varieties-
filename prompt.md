You are a senior Full-Stack Engineer specializing in Flask, JavaScript, SQL, and Chart.js.

My Analytics Dashboard loads successfully, but the charts are not rendering correctly.

Current behavior:

* The chart containers appear.
* Axes and labels are visible.
* No bars, lines, or pie slices are displayed.
* This indicates that Chart.js is loading but the datasets are not being populated correctly.

Your task is to find and fix the root cause without changing the existing UI design.

## Debug Everything

Verify the following in order:

1. The backend `/api/analytics` endpoint returns real data.
2. The frontend correctly fetches the API.
3. The JSON structure matches what the JavaScript expects.
4. Labels and dataset arrays contain valid values.
5. Dataset values are numeric, not strings or null.
6. Chart.js receives the correct arrays.
7. Existing chart instances are updated instead of recreated.
8. The charts call `chart.update()` after data changes.
9. Canvas sizing is correct.
10. Chart colors are visible on the dark theme.
11. Browser console contains no JavaScript errors.
12. Network requests return HTTP 200.

## Verify API Response

The endpoint should return data similar to:

{
"total_predictions": 15,
"today_predictions": 3,
"average_confidence": 96.4,
"most_predicted": "Apple 10",

"variety_distribution": {
"Apple 10": 8,
"Apple 3": 5,
"Apple 7": 2
},

"daily_predictions": [
{
"date":"2026-06-01",
"count":2
},
{
"date":"2026-06-02",
"count":5
}
]
}

If the response is empty or incorrect, fix the backend.

## Fix JavaScript

Ensure the chart receives:

const labels = Object.keys(data.variety_distribution);
const values = Object.values(data.variety_distribution).map(Number);

Then update:

chart.data.labels = labels;
chart.data.datasets[0].data = values;
chart.update();

Do NOT recreate the chart every refresh.

## Add Debug Logs

Temporarily log:

console.log("Analytics API:", data);
console.log("Labels:", labels);
console.log("Values:", values);

These logs should confirm that the frontend receives valid data.

## SQL Verification

Verify the backend query uses aggregation similar to:

SELECT predicted_class,
COUNT(*) AS total
FROM predictions
GROUP BY predicted_class;

Do not return duplicate rows.

## Chart Configuration

Ensure datasets include visible styling such as:

backgroundColor
borderColor
borderWidth

Do not use colors that blend into the dark background.

## Empty State

If there are no predictions:

* Show "No prediction data available".
* Do not create an empty chart.
* Do not throw JavaScript errors.

## Final Output

After fixing everything, provide:

1. The exact bug that caused the charts not to render.
2. Every modified file.
3. The corrected backend code.
4. The corrected frontend JavaScript.
5. Confirmation that:

   * Bar Chart works.
   * Pie Chart works.
   * Line Chart works.
   * Dashboard cards update correctly.
   * Analytics update automatically after every new prediction.

Do not use mock data or hardcoded values. Use only real prediction data from the database.
