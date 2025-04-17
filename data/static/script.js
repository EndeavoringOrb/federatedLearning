let selectedRuns = [];
let currentChart = null;

function highlightDifferences(configs) {
    const allKeys = new Set();
    configs.forEach(cfg => Object.keys(cfg).forEach(k => allKeys.add(k)));
    let result = "";

    for (let key of allKeys) {
        let values = configs.map(cfg => JSON.stringify(cfg[key]));
        let isDifferent = new Set(values).size > 1;
        result += isDifferent ? `<span class="diff">${key}: ${values.join(" | ")}</span>\n` : `${key}: ${values[0]}\n`;
    }
    return result;
}

function fetchAndDisplay(runs) {
    $.ajax({
        url: "/get_run_data",
        method: "POST",
        data: JSON.stringify({ runs }),
        contentType: "application/json",
        success: function (data) {
            const ctx = document.getElementById('chart').getContext('2d');
            const datasets = [];
            const configs = [];

            runs.forEach(run => {
                const runData = data[run];
                if (runData.error) return;
                configs.push(runData.config);
                datasets.push(
                    {
                        label: `Run ${run} - current`,
                        data: runData.loss.current_reward,
                        borderColor: 'blue',
                        fill: false
                    },
                    {
                        label: `Run ${run} - mean`,
                        data: runData.loss.mean,
                        borderColor: 'green',
                        fill: false
                    },
                    {
                        label: `Run ${run} - max`,
                        data: runData.loss.max_reward,
                        borderColor: 'red',
                        fill: false
                    }
                );
            });

            if (currentChart) currentChart.destroy();

            currentChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({ length: datasets[0]?.data.length || 0 }, (_, i) => i),
                    datasets
                },
                options: {
                    responsive: true,
                    plugins: {
                        zoom: {
                            pan: {
                                enabled: true,
                                mode: 'x',
                            },
                            zoom: {
                                wheel: {
                                    enabled: true
                                },
                                pinch: {
                                    enabled: true
                                },
                                mode: 'x',
                            }
                        }
                    }
                }
            });

            const configText = highlightDifferences(configs);
            document.getElementById("configDisplay").innerHTML = configText;
        }
    });
}

$(document).ready(function () {
    $(".run-item").click(function () {
        const runId = $(this).data("run").toString();
        if (selectedRuns.includes(runId)) {
            selectedRuns = selectedRuns.filter(r => r !== runId);
            $(this).removeClass("selected");
        } else {
            selectedRuns.push(runId);
            $(this).addClass("selected");
        }
        fetchAndDisplay(selectedRuns);
    });

    $("#themeToggle").click(function () {
        const body = document.body;
        const current = body.getAttribute("data-theme");
        const next = current === "dark" ? "light" : "dark";
        body.setAttribute("data-theme", next);
    });
});