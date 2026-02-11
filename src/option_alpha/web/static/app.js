/**
 * Option Alpha Dashboard - Client-side JavaScript
 *
 * Handles WebSocket connection for scan progress and minor Alpine.js interactivity.
 */

(function () {
    "use strict";

    // -------------------------------------------------------------------
    // WebSocket: Scan Progress
    // -------------------------------------------------------------------

    let ws = null;
    let reconnectTimer = null;
    const RECONNECT_DELAY_MS = 3000;

    function connectWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const url = protocol + "//" + window.location.host + "/ws/scan-progress";

        ws = new WebSocket(url);

        ws.onopen = function () {
            console.log("[WS] Connected to scan-progress");
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }
        };

        ws.onmessage = function (event) {
            try {
                const progress = JSON.parse(event.data);
                updateProgress(progress);
            } catch (err) {
                console.error("[WS] Failed to parse message:", err);
            }
        };

        ws.onclose = function () {
            console.log("[WS] Disconnected, will reconnect in", RECONNECT_DELAY_MS, "ms");
            ws = null;
            reconnectTimer = setTimeout(connectWebSocket, RECONNECT_DELAY_MS);
        };

        ws.onerror = function (err) {
            console.error("[WS] Error:", err);
            ws.close();
        };
    }

    /**
     * Update the progress display from a ScanProgress JSON object.
     */
    function updateProgress(progress) {
        // Update progress bar.
        var bar = document.getElementById("progress-bar");
        var pct = document.getElementById("progress-pct");
        if (bar) {
            bar.style.width = progress.overall_percentage + "%";
        }
        if (pct) {
            pct.textContent = Math.round(progress.overall_percentage) + "%";
        }

        // Update phase list.
        var phaseList = document.getElementById("phase-list");
        if (phaseList && progress.phases) {
            var items = phaseList.querySelectorAll(".phase-item");
            progress.phases.forEach(function (phase, idx) {
                if (items[idx]) {
                    // Update status class.
                    items[idx].className = "phase-item phase-" + phase.status;

                    // Update icon.
                    var icon = items[idx].querySelector(".phase-icon");
                    if (icon) {
                        if (phase.status === "completed") {
                            icon.textContent = "\u2713";  // checkmark
                        } else if (phase.status === "running") {
                            icon.textContent = "\u25B6";  // play
                        } else if (phase.status === "failed") {
                            icon.textContent = "\u2717";  // x
                        } else {
                            icon.textContent = "\u25CB";  // circle
                        }
                    }
                }
            });
        }

        // If scan complete (100%), refresh candidates table.
        if (progress.overall_percentage >= 100) {
            var table = document.getElementById("candidates-table");
            if (table && typeof htmx !== "undefined") {
                // Small delay to let the DB persist finish.
                setTimeout(function () {
                    htmx.ajax("GET", "/candidates", {target: "#candidates-table", swap: "innerHTML"});
                }, 1000);
            }
        }
    }

    // -------------------------------------------------------------------
    // Initialize on DOM ready
    // -------------------------------------------------------------------

    document.addEventListener("DOMContentLoaded", function () {
        connectWebSocket();
    });
})();
