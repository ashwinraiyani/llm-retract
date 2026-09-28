(function () {
  const districts = window.DISTRICT_DATA || [];
  if (!districts.length) {
    return;
  }

  const alertColors = {
    Green: "#2e7d32",
    Yellow: "#efb300",
    Orange: "#ef6c00",
    Red: "#c62828"
  };

  function calculateHhriScore(hhri) {
    const score =
      0.4 * hhri.exposure + 0.35 * hhri.sensitivity + 0.25 * (100 - hhri.adaptiveCapacity);
    return Math.round(score);
  }

  function deriveAlertLevel(hhriScore) {
    if (hhriScore >= 75) return "Red";
    if (hhriScore >= 55) return "Orange";
    if (hhriScore >= 35) return "Yellow";
    return "Green";
  }

  const enrichedDistricts = districts.map((district) => {
    const hhriScore = calculateHhriScore(district.hhri);
    return {
      ...district,
      hhriScore,
      alertLevel: deriveAlertLevel(hhriScore)
    };
  });

  const map = L.map("map", { zoomControl: true }).setView([22.7, 72.0], 7);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(map);

  function markerIcon(color) {
    return L.divIcon({
      className: "custom-marker",
      html: `<span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:${color};border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.25)"></span>`,
      iconSize: [14, 14],
      iconAnchor: [7, 7]
    });
  }

  const summaryCardsEl = document.getElementById("summary-cards");
  const currentMetricsEl = document.getElementById("district-current-metrics");
  const healthIndicatorsEl = document.getElementById("health-indicators");
  const districtNameEl = document.getElementById("district-name");
  const districtEnvironmentEl = document.getElementById("district-environment");
  const districtAlertPillEl = document.getElementById("district-alert-pill");
  const hhriTotalNoteEl = document.getElementById("hhri-total-note");
  const districtSampleLabelEl = document.getElementById("district-sample-label");
  const mobileAlertPreviewEl = document.getElementById("mobile-alert-preview");

  let forecastChart;
  let hhriBreakdownChart;
  let stateHhriChart;

  function renderSummary() {
    const order = ["Green", "Yellow", "Orange", "Red"];
    const counts = order.reduce((acc, level) => {
      acc[level] = enrichedDistricts.filter((d) => d.alertLevel === level).length;
      return acc;
    }, {});

    summaryCardsEl.innerHTML = order
      .map(
        (level) => `
      <article class="summary-card ${level.toLowerCase()}">
        <h3>${level} Alert</h3>
        <p>${counts[level]}</p>
      </article>
    `
      )
      .join("");

    if (stateHhriChart) {
      stateHhriChart.destroy();
    }

    stateHhriChart = new Chart(document.getElementById("stateHhriChart"), {
      type: "bar",
      data: {
        labels: enrichedDistricts.map((d) => d.name),
        datasets: [
          {
            label: "HHRI Score",
            data: enrichedDistricts.map((d) => d.hhriScore),
            backgroundColor: enrichedDistricts.map((d) => alertColors[d.alertLevel])
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            title: { display: true, text: "HHRI (0-100)" }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }

  function renderDetail(district) {
    districtNameEl.textContent = district.name;
    districtEnvironmentEl.textContent = district.environment;

    districtAlertPillEl.textContent = `${district.alertLevel} Alert`;
    districtAlertPillEl.style.background = alertColors[district.alertLevel];

    currentMetricsEl.innerHTML = [
      ["Current Temperature", `${district.current.temperatureC.toFixed(1)}°C`],
      ["Humidity", `${district.current.humidityPct}%`],
      ["Heat Index", `${district.current.heatIndexC.toFixed(1)}°C`]
    ]
      .map(
        ([label, value]) => `
        <article class="metric-card">
          <h4>${label}</h4>
          <p>${value}</p>
        </article>`
      )
      .join("");

    healthIndicatorsEl.innerHTML = [
      ["Estimated Heat-related OPD visits/day", district.healthIndicators.estimatedDailyOpdHeatCases],
      ["Vulnerable Population", `${district.healthIndicators.vulnerablePopulationPct}%`],
      ["Hospital Capacity Utilization", `${district.healthIndicators.hospitalCapacityUtilizationPct}%`]
    ]
      .map(
        ([label, value]) => `
        <article class="metric-card">
          <h4>${label}</h4>
          <p>${value}</p>
        </article>`
      )
      .join("");

    districtSampleLabelEl.textContent = district.dataDisclaimer;

    const forecastLabels = district.forecast.map((f) => f.day);
    const forecastTemp = district.forecast.map((f) => f.temperatureC);
    const forecastHeatIndex = district.forecast.map((f) => f.heatIndexC);

    if (forecastChart) {
      forecastChart.destroy();
    }

    forecastChart = new Chart(document.getElementById("forecastChart"), {
      type: "line",
      data: {
        labels: forecastLabels,
        datasets: [
          {
            label: "Temperature (°C)",
            data: forecastTemp,
            borderColor: "#0f5b8d",
            backgroundColor: "rgba(15,91,141,.15)",
            tension: 0.35
          },
          {
            label: "Heat Index (°C)",
            data: forecastHeatIndex,
            borderColor: "#c62828",
            backgroundColor: "rgba(198,40,40,.14)",
            tension: 0.35
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: "bottom" }
        },
        scales: {
          y: {
            suggestedMin: 35,
            suggestedMax: 52
          }
        }
      }
    });

    if (hhriBreakdownChart) {
      hhriBreakdownChart.destroy();
    }

    hhriBreakdownChart = new Chart(document.getElementById("hhriBreakdownChart"), {
      type: "doughnut",
      data: {
        labels: ["Exposure", "Sensitivity", "Adaptive Capacity"],
        datasets: [
          {
            data: [district.hhri.exposure, district.hhri.sensitivity, district.hhri.adaptiveCapacity],
            backgroundColor: ["#ef6c00", "#f59e0b", "#0284c7"]
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: "bottom" }
        }
      }
    });

    hhriTotalNoteEl.textContent = `Composite HHRI Score: ${district.hhriScore}/100 (derived from Exposure, Sensitivity, and inverse Adaptive Capacity).`;
  }

  function renderMobileMock() {
    const redDistrict = enrichedDistricts
      .filter((d) => d.alertLevel === "Red")
      .sort((a, b) => b.hhriScore - a.hhriScore)[0] || enrichedDistricts[0];

    mobileAlertPreviewEl.innerHTML = `
      <div class="mobile-header">Sample Push/SMS Alert Preview</div>
      <div class="mobile-message">
        <strong>RED ALERT — ${redDistrict.name} District</strong><br/>
        Extreme heat and severe health risk expected in the next 24-48 hours.
        <br/><br/>
        Heat Index: ${redDistrict.current.heatIndexC.toFixed(1)}°C | HHRI: ${redDistrict.hhriScore}/100
        <br/><br/>
        Action: Activate emergency response, extend cooling centre hours, deploy field health workers, and issue public advisories for vulnerable populations.
      </div>
    `;
  }

  enrichedDistricts.forEach((district) => {
    const marker = L.marker([district.lat, district.lng], {
      icon: markerIcon(alertColors[district.alertLevel])
    }).addTo(map);

    marker.bindTooltip(`${district.name} — ${district.alertLevel}`);
    marker.on("click", () => renderDetail(district));
  });

  renderSummary();
  renderMobileMock();
  renderDetail(enrichedDistricts[0]);
})();
