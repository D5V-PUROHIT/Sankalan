document.addEventListener('DOMContentLoaded', () => {
    // Global Application State
    const AppState = {
        currentProfile: null,
        currentPreprocessing: null,
        edaData: null,
        comparisonData: null,
        qualityReport: null,
        trainingHistory: null,
        synthDownloadUrl: '/api/download',
        reportDownloadUrl: '/api/download/report',
        progressPollTimer: null,
        activeTheaterChartId: null
    };

    // Tab Titles & Descriptions
    const tabMeta = {
        'tab-upload': {
            title: 'Dataset Upload & Ingestion',
            desc: 'Load tabular CSV or Excel records to begin deep learning synthesis'
        },
        'tab-overview': {
            title: 'Dataset Profiling & Schema Architecture',
            desc: 'Automated summary statistics, role classification, and preprocessing breakdown'
        },
        'tab-eda': {
            title: 'Exploratory Data Analysis (EDA)',
            desc: 'Analyze empirical distributions, statistical dispersion, and multi-variable correlations'
        },
        'tab-generate': {
            title: 'Train Variational Autoencoder & Synthesize',
            desc: 'Tune latent hyperparameters, train the neural network with live progress, and generate synthetic rows'
        },
        'tab-compare': {
            title: 'Real vs Synthetic Distribution Comparison',
            desc: 'Side-by-side empirical validation verifying preservation of multi-variable relationships'
        },
        'tab-report': {
            title: 'Synthetic Data Quality & Fidelity Benchmark',
            desc: 'Comprehensive statistical evaluation score synthesized from distribution, correlation, and privacy metrics'
        }
    };

    // Plotly Base Layout with High Contrast and Auto Margins (No Cutoff)
    const plotlyLayoutBase = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#0f172a', family: 'Plus Jakarta Sans, sans-serif', size: 12 },
        margin: { t: 45, b: 70, l: 110, r: 40, automargin: true },
        autosize: true,
        xaxis: { 
            automargin: true, 
            tickfont: { size: 12, color: '#334155', weight: 600 },
            titlefont: { size: 13, color: '#0f172a', weight: 700 }
        },
        yaxis: { 
            automargin: true, 
            tickfont: { size: 12, color: '#334155', weight: 600 },
            titlefont: { size: 13, color: '#0f172a', weight: 700 }
        }
    };

    const plotlyConfig = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    // -------------------------------------------------------------
    // Global Zoom & Theater Mode Handlers (Accessible from HTML)
    // -------------------------------------------------------------
    window.zoomChart = function(chartId, factor) {
        const el = document.getElementById(chartId);
        if (!el || !el.layout) return;

        const update = {};
        const xRange = el.layout.xaxis ? el.layout.xaxis.range : null;
        const yRange = el.layout.yaxis ? el.layout.yaxis.range : null;

        if (xRange && xRange.length === 2 && typeof xRange[0] === 'number') {
            const mid = (xRange[0] + xRange[1]) / 2;
            const span = (xRange[1] - xRange[0]) * factor / 2;
            update['xaxis.range'] = [mid - span, mid + span];
            update['xaxis.autorange'] = false;
        }

        if (yRange && yRange.length === 2 && typeof yRange[0] === 'number') {
            const mid = (yRange[0] + yRange[1]) / 2;
            const span = (yRange[1] - yRange[0]) * factor / 2;
            update['yaxis.range'] = [mid - span, mid + span];
            update['yaxis.autorange'] = false;
        }

        if (Object.keys(update).length > 0) {
            Plotly.relayout(el, update);
        }
    };

    window.resetChart = function(chartId) {
        const el = document.getElementById(chartId);
        if (!el) return;
        Plotly.relayout(el, { 'xaxis.autorange': true, 'yaxis.autorange': true });
    };

    window.openTheater = function(sourceChartId, title) {
        const sourceEl = document.getElementById(sourceChartId);
        if (!sourceEl || !sourceEl.data) {
            alert('Chart is still loading or has not been rendered yet.');
            return;
        }

        AppState.activeTheaterChartId = sourceChartId;
        const modal = document.getElementById('theater-modal');
        modal.classList.remove('hidden');

        document.getElementById('theater-title').textContent = title || 'Expanded Visualization';

        // Deep copy data & layout for larger theater display
        const theaterData = JSON.parse(JSON.stringify(sourceEl.data));
        const theaterLayout = JSON.parse(JSON.stringify(sourceEl.layout || plotlyLayoutBase));

        // Enlarge fonts and spacing for theater view
        theaterLayout.font = { ...theaterLayout.font, size: 14 };
        theaterLayout.margin = { t: 50, b: 80, l: 120, r: 50, automargin: true };
        if (theaterLayout.xaxis) theaterLayout.xaxis.tickfont = { size: 13, color: '#1e293b' };
        if (theaterLayout.yaxis) theaterLayout.yaxis.tickfont = { size: 13, color: '#1e293b' };

        Plotly.newPlot('theater-plot', theaterData, theaterLayout, plotlyConfig).then(() => {
            Plotly.Plots.resize(document.getElementById('theater-plot'));
        });

        // Populate theater sidebar stats
        populateTheaterSidebar(sourceChartId);
    };

    window.closeTheater = function() {
        const modal = document.getElementById('theater-modal');
        modal.classList.add('hidden');
        AppState.activeTheaterChartId = null;
    };

    window.zoomTheater = function(factor) {
        window.zoomChart('theater-plot', factor);
    };

    window.resetTheater = function() {
        window.resetChart('theater-plot');
    };

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            window.closeTheater();
        }
    });

    function populateTheaterSidebar(sourceChartId) {
        const metricsContainer = document.getElementById('theater-metrics-content');
        const infoText = document.getElementById('theater-info-text');
        metricsContainer.innerHTML = '';

        if (sourceChartId === 'chart-eda-distribution' && AppState.edaData) {
            const col = document.getElementById('eda-num-select').value;
            const dist = AppState.edaData.column_distributions[col];
            infoText.textContent = `Analyzing numerical dispersion and skewness for "${col}". Density reveals distribution spread.`;
            if (dist) {
                metricsContainer.innerHTML = `
                    <div class="theater-metric-row"><span>Feature</span><strong>${col}</strong></div>
                    <div class="theater-metric-row"><span>Mean</span><strong>${dist.mean}</strong></div>
                    <div class="theater-metric-row"><span>Median</span><strong>${dist.median}</strong></div>
                    <div class="theater-metric-row"><span>Std Dev</span><strong>${dist.std}</strong></div>
                    <div class="theater-metric-row"><span>Min / Max</span><strong>${dist.min} / ${dist.max}</strong></div>
                `;
            }
        } else if (sourceChartId === 'chart-compare-numeric' && AppState.comparisonData) {
            const num = AppState.comparisonData.num_distribution;
            infoText.textContent = `Overlaid empirical comparison between Real (Royal Blue) and VAE Synthetic (Sunset Orange) samples for "${num ? num.column : ''}".`;
            if (num) {
                metricsContainer.innerHTML = `
                    <div class="theater-metric-row"><span>Real Mean</span><strong style="color:var(--real-color, #2563eb)">${num.real_stats.mean}</strong></div>
                    <div class="theater-metric-row"><span>Synth Mean</span><strong style="color:var(--synth-color, #ea580c)">${num.synth_stats.mean}</strong></div>
                    <div class="theater-metric-row"><span>Real Std</span><strong style="color:var(--real-color, #2563eb)">${num.real_stats.std}</strong></div>
                    <div class="theater-metric-row"><span>Synth Std</span><strong style="color:var(--synth-color, #ea580c)">${num.synth_stats.std}</strong></div>
                `;
            }
        } else if (sourceChartId === 'chart-training-loss' && AppState.trainingHistory) {
            const h = AppState.trainingHistory;
            const epochs = h.total_loss.length;
            const finalTotal = h.total_loss[epochs - 1];
            const finalRecon = h.recon_loss[epochs - 1];
            const finalKL = h.kl_loss[epochs - 1];
            infoText.textContent = 'Demonstrates Variational Autoencoder optimization over training epochs. Total Loss combines Reconstruction Error and Latent Regularization.';
            metricsContainer.innerHTML = `
                <div class="theater-metric-row"><span>Epochs Run</span><strong>${epochs}</strong></div>
                <div class="theater-metric-row"><span>Final Total Loss</span><strong style="color:#4f46e5">${finalTotal.toFixed(3)}</strong></div>
                <div class="theater-metric-row"><span>Reconstruction Loss</span><strong style="color:#059669">${finalRecon.toFixed(3)}</strong></div>
                <div class="theater-metric-row"><span>KL Divergence</span><strong style="color:#d97706">${finalKL.toFixed(3)}</strong></div>
            `;
        } else {
            infoText.textContent = 'Expanded interactive view. Use the toolbar at the top right to zoom, pan, or download high-resolution PNG.';
        }
    }

    // -------------------------------------------------------------
    // Tab Navigation
    // -------------------------------------------------------------
    const navLinks = document.querySelectorAll('.nav-links li');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const pageTitle = document.getElementById('page-title');
    const pageDesc = document.getElementById('page-desc');

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            const targetTab = link.getAttribute('data-tab');
            switchTab(targetTab);
        });
    });

    function switchTab(targetTab) {
        if (!targetTab) return;
        localStorage.setItem('sankalan_active_tab', targetTab);

        navLinks.forEach(l => l.classList.remove('active'));
        tabPanes.forEach(p => p.classList.remove('active'));

        const activeLink = document.querySelector(`[data-tab="${targetTab}"]`);
        if (activeLink) activeLink.classList.add('active');

        const pane = document.getElementById(targetTab);
        if (pane) pane.classList.add('active');

        if (tabMeta[targetTab]) {
            pageTitle.textContent = tabMeta[targetTab].title;
            pageDesc.textContent = tabMeta[targetTab].desc;
        }

        // Trigger chart rendering & resizing on tab switch
        if (targetTab === 'tab-eda' && !AppState.edaData) {
            loadEDAData();
        } else if (targetTab === 'tab-compare' && !AppState.comparisonData) {
            loadComparisonData();
        }

        // Force Plotly resize to fit active tab
        setTimeout(() => {
            document.querySelectorAll('.chart-plot').forEach(el => {
                if (el.data) Plotly.Plots.resize(el);
            });
        }, 150);
    }

    // -------------------------------------------------------------
    // Initial System Status Check & State Restoration on Refresh
    // -------------------------------------------------------------
    checkSystemStatus();

    function checkSystemStatus() {
        fetch('/api/status')
            .then(res => res.json())
            .then(status => {
                const sysStatus = document.getElementById('system-status');

                // 1. Restore Dataset & Profiling State
                if (status.dataset_loaded) {
                    sysStatus.textContent = `Dataset Ready: ${status.raw_filename}`;
                    const banner = document.getElementById('demo-banner');
                    if (banner) banner.style.display = 'flex';

                    if (status.profile) {
                        AppState.currentProfile = status.profile;
                        AppState.currentPreprocessing = status.preprocessing;
                        renderOverviewTable(status.profile, status.preprocessing);
                    }

                    // Preload EDA in background
                    loadEDAData(false);
                }

                // 2. Restore Synthesis, Models & Fidelity Reports
                if (status.synthetic_ready) {
                    document.getElementById('btn-download-csv').disabled = false;
                    document.getElementById('btn-download-report').disabled = false;

                    if (status.latest_quality_score) {
                        document.getElementById('quality-score').textContent = `${status.latest_quality_score}%`;
                    }

                    if (status.training_history) {
                        AppState.trainingHistory = status.training_history;
                        renderTrainingLossCurve(status.training_history);
                    }

                    if (status.evaluation_report) {
                        AppState.qualityReport = status.evaluation_report;
                        renderQualityReport(status.evaluation_report);
                    }

                    // Preload Comparison Data
                    loadComparisonData();
                }

                // 3. Restore Last Active Tab or Intelligent Default
                const savedTab = localStorage.getItem('sankalan_active_tab');
                if (savedTab && document.getElementById(savedTab)) {
                    switchTab(savedTab);
                } else if (status.synthetic_ready) {
                    switchTab('tab-compare');
                } else if (status.dataset_loaded) {
                    switchTab('tab-overview');
                } else {
                    switchTab('tab-upload');
                }
            })
            .catch(err => console.log('Status check error:', err));
    }

    document.getElementById('btn-use-demo').addEventListener('click', () => {
        loadEDAData(true);
        switchTab('tab-overview');
    });

    // -------------------------------------------------------------
    // File Upload Handling
    // -------------------------------------------------------------
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const uploadText = document.getElementById('upload-text');

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--accent-color)';
        dropZone.style.background = '#f5f3ff';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#cbd5e1';
        dropZone.style.background = '#f8fafc';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#cbd5e1';
        dropZone.style.background = '#f8fafc';
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });

    function handleFileUpload(file) {
        if (!file.name.endsWith('.csv') && !file.name.endsWith('.xlsx')) {
            alert('Please upload a valid .csv or .xlsx file.');
            return;
        }

        dropZone.classList.add('hidden');
        uploadStatus.classList.remove('hidden');
        uploadText.textContent = `Uploading and profiling ${file.name}...`;

        const formData = new FormData();
        formData.append('file', file);

        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.profile) {
                AppState.currentProfile = data.profile;
                AppState.currentPreprocessing = data.preprocessing;
                AppState.edaData = null;
                AppState.comparisonData = null;
                AppState.qualityReport = null;
                AppState.trainingHistory = null;

                // Reset synthetic generation states for new dataset
                document.getElementById('btn-download-csv').disabled = true;
                document.getElementById('btn-download-report').disabled = true;
                document.getElementById('quality-score').textContent = '--';

                uploadText.textContent = 'Preprocessing complete! Redirecting to Overview...';
                document.getElementById('system-status').textContent = `Dataset: ${data.filename}`;

                renderOverviewTable(data.profile, data.preprocessing);

                // Fetch fresh EDA data for newly uploaded dataset
                loadEDAData(false);

                setTimeout(() => {
                    uploadStatus.classList.add('hidden');
                    dropZone.classList.remove('hidden');
                    switchTab('tab-overview');
                }, 900);
            } else {
                throw new Error(data.message || 'Upload failed');
            }
        })
        .catch(err => {
            console.error('Upload Error:', err);
            uploadText.textContent = `Error: ${err.message}`;
            uploadText.style.color = '#dc2626';
            setTimeout(() => {
                uploadStatus.classList.add('hidden');
                dropZone.classList.remove('hidden');
                uploadText.style.color = 'inherit';
            }, 3000);
        });
    }

    // -------------------------------------------------------------
    // Tab 2: Overview Schema & Profiling
    // -------------------------------------------------------------
    function renderOverviewTable(profile, preprocessing) {
        if (!profile) return;

        document.getElementById('kpi-rows').textContent = profile.num_rows.toLocaleString();
        document.getElementById('kpi-cols').textContent = profile.num_cols.toLocaleString();
        document.getElementById('kpi-missing').textContent = profile.total_missing_values.toLocaleString();
        document.getElementById('kpi-vae-dim').textContent = preprocessing ? preprocessing.vae_input_dim : '--';

        const tbody = document.getElementById('columns-table-body');
        tbody.innerHTML = '';

        profile.columns.forEach(col => {
            const tr = document.createElement('tr');
            
            let roleBadge = '<span class="badge badge-identifier">Identifier / Date</span>';
            if (col.type === 'Numerical') {
                roleBadge = '<span class="badge badge-numeric"><i class="fa-solid fa-calculator"></i> Generative Numeric</span>';
            } else if (col.type === 'Categorical') {
                roleBadge = '<span class="badge badge-categorical"><i class="fa-solid fa-list-check"></i> Generative Categorical</span>';
            }

            tr.innerHTML = `
                <td style="font-weight: 700;">${col.name}</td>
                <td>${col.type} <span style="color: var(--text-muted); font-size: 0.8rem;">(${col.dtype})</span></td>
                <td>${roleBadge}</td>
                <td style="font-family: 'JetBrains Mono';">${col.unique_count.toLocaleString()}</td>
                <td style="color: ${col.missing_percentage > 0 ? 'var(--danger-color)' : 'var(--text-secondary)'}; font-weight: 600;">
                    ${col.missing_percentage}% <span style="font-size: 0.78rem; color: var(--text-muted);">(${col.missing_count})</span>
                </td>
            `;
            tbody.appendChild(tr);
        });
    }

    const searchInput = document.getElementById('column-search-input');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const term = e.target.value.toLowerCase();
            const rows = document.querySelectorAll('#columns-table-body tr');
            rows.forEach(r => {
                const text = r.textContent.toLowerCase();
                r.style.display = text.includes(term) ? '' : 'none';
            });
        });
    }

    // -------------------------------------------------------------
    // Tab 3: Exploratory Data Analysis (EDA)
    // -------------------------------------------------------------
    function loadEDAData(autoSwitch = false) {
        fetch('/api/eda')
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    AppState.edaData = data;
                    populateEDASelectors(data);
                    renderEDACorrelation(data.correlation);

                    if (!AppState.currentProfile && data.column_distributions) {
                        const numCols = Object.keys(data.column_distributions);
                        document.getElementById('kpi-rows').textContent = '~7,043';
                        document.getElementById('kpi-cols').textContent = (numCols.length + Object.keys(data.categorical_distributions).length).toString();
                        document.getElementById('kpi-missing').textContent = '0 (Imputed)';
                        document.getElementById('kpi-vae-dim').textContent = '46';
                    }

                    if (autoSwitch) switchTab('tab-eda');
                }
            })
            .catch(err => console.log('Error loading EDA:', err));
    }

    function populateEDASelectors(data) {
        const numSelect = document.getElementById('eda-num-select');
        const catSelect = document.getElementById('eda-cat-select');

        numSelect.innerHTML = '';
        catSelect.innerHTML = '';

        if (data.num_columns) {
            data.num_columns.forEach(col => {
                const opt = document.createElement('option');
                opt.value = col;
                opt.textContent = col;
                numSelect.appendChild(opt);
            });
        }

        if (data.cat_columns) {
            data.cat_columns.forEach(col => {
                const opt = document.createElement('option');
                opt.value = col;
                opt.textContent = col;
                catSelect.appendChild(opt);
            });
        }

        numSelect.onchange = () => renderEDANumericDistribution(numSelect.value);
        catSelect.onchange = () => renderEDACategorical(catSelect.value);

        if (data.num_columns && data.num_columns.length) {
            renderEDANumericDistribution(data.num_columns[0]);
        }
        if (data.cat_columns && data.cat_columns.length) {
            renderEDACategorical(data.cat_columns[0]);
        }
    }

    function renderEDANumericDistribution(colName) {
        const dist = AppState.edaData.column_distributions[colName];
        if (!dist) return;

        document.getElementById('eda-dist-title').textContent = `Distribution of ${colName}`;
        document.getElementById('eda-dist-tag').textContent = `Mean: ${dist.mean} • Std: ${dist.std}`;

        const trace = {
            x: dist.values,
            type: 'histogram',
            name: colName,
            marker: {
                color: '#4f46e5',
                line: { color: '#3730a3', width: 1 }
            },
            opacity: 0.8
        };

        const layout = {
            ...plotlyLayoutBase,
            xaxis: { ...plotlyLayoutBase.xaxis, title: colName, gridcolor: '#e2e8f0' },
            yaxis: { ...plotlyLayoutBase.yaxis, title: 'Record Count', gridcolor: '#e2e8f0' },
            bargap: 0.05
        };

        Plotly.newPlot('chart-eda-distribution', [trace], layout, plotlyConfig);
    }

    function renderEDACategorical(colName) {
        const cat = AppState.edaData.categorical_distributions[colName];
        if (!cat) return;

        document.getElementById('eda-cat-title').textContent = `Category Counts: ${colName}`;

        const trace = {
            x: cat.labels,
            y: cat.counts,
            type: 'bar',
            marker: {
                color: '#7c3aed',
                line: { color: '#5b21b6', width: 1 }
            }
        };

        const layout = {
            ...plotlyLayoutBase,
            xaxis: { ...plotlyLayoutBase.xaxis, title: colName, tickangle: -20, gridcolor: '#e2e8f0' },
            yaxis: { ...plotlyLayoutBase.yaxis, title: 'Count', gridcolor: '#e2e8f0' }
        };

        Plotly.newPlot('chart-eda-category', [trace], layout, plotlyConfig);
    }

    function renderEDACorrelation(corr) {
        if (!corr) return;

        const trace = {
            z: corr.z,
            x: corr.x,
            y: corr.y,
            type: 'heatmap',
            colorscale: 'Blues',
            colorbar: { title: 'Pearson r' },
            zmin: -1,
            zmax: 1
        };

        const layout = {
            ...plotlyLayoutBase,
            margin: { t: 40, b: 90, l: 140, r: 40, automargin: true }
        };

        Plotly.newPlot('chart-eda-correlation', [trace], layout, plotlyConfig);
    }

    // -------------------------------------------------------------
    // Tab 4: VAE Training with Live Continuous Progress & Stop Support
    // -------------------------------------------------------------
    const btnGenerate = document.getElementById('btn-generate');
    const btnStopTraining = document.getElementById('btn-stop-training');
    const genProgress = document.getElementById('generation-progress');
    const genProgressBar = document.getElementById('gen-progress-bar');
    const genStatusText = document.getElementById('gen-status-text');
    const genPctText = document.getElementById('gen-pct-text');
    const genLiveMetrics = document.getElementById('gen-live-metrics');

    btnGenerate.addEventListener('click', () => {
        const numRecords = parseInt(document.getElementById('num-records').value) || 5000;
        const epochs = parseInt(document.getElementById('epochs').value) || 40;
        const latentDim = parseInt(document.getElementById('latent-dim').value) || 8;

        // UI state: training active
        btnGenerate.classList.add('hidden');
        btnStopTraining.classList.remove('hidden');
        genProgress.classList.remove('hidden');

        genProgressBar.style.width = '5%';
        genProgressBar.style.background = 'linear-gradient(90deg, var(--accent-color), var(--purple-accent), #38bdf8)';
        genPctText.textContent = '5%';
        genStatusText.textContent = `Initializing training for ${epochs} epochs...`;
        genLiveMetrics.innerHTML = '<span><i class="fa-solid fa-spinner fa-spin"></i> Initializing model tensors...</span>';

        // Start live continuous progress polling from backend
        startProgressPolling();

        fetch(`/api/generate?num_records=${numRecords}&epochs=${epochs}&latent_dim=${latentDim}`, {
            method: 'POST'
        })
        .then(res => res.json())
        .then(data => {
            stopProgressPolling();
            btnGenerate.classList.remove('hidden');
            btnStopTraining.classList.add('hidden');

            if (data.quality_report) {
                genProgressBar.style.width = '100%';
                genPctText.textContent = '100%';
                genStatusText.textContent = 'Training & Synthesis complete!';
                genLiveMetrics.innerHTML = '<span style="color: var(--success-color);"><i class="fa-solid fa-check-circle"></i> VAE successfully generated synthetic dataset!</span>';

                AppState.qualityReport = data.quality_report;
                AppState.trainingHistory = data.training_history;

                if (data.training_history) {
                    renderTrainingLossCurve(data.training_history);
                }

                renderQualityReport(data.quality_report);

                document.getElementById('btn-download-csv').disabled = false;
                document.getElementById('btn-download-report').disabled = false;

                setTimeout(() => {
                    switchTab('tab-compare');
                }, 1200);
            } else {
                throw new Error(data.message || 'Generation failed');
            }
        })
        .catch(err => {
            stopProgressPolling();
            btnGenerate.classList.remove('hidden');
            btnStopTraining.classList.add('hidden');
            genStatusText.textContent = `Error: ${err.message}`;
            genStatusText.style.color = '#dc2626';
            genProgressBar.style.background = '#dc2626';
            genLiveMetrics.innerHTML = `<span style="color: #dc2626;"><i class="fa-solid fa-circle-exclamation"></i> Training stopped due to error.</span>`;
            console.error('Generation Error:', err);
        });
    });

    btnStopTraining.addEventListener('click', () => {
        btnStopTraining.disabled = true;
        genStatusText.textContent = 'Halting training... Finishing current epoch...';
        fetch('/api/train/stop', { method: 'POST' })
            .then(res => res.json())
            .then(res => {
                console.log('Stop signal sent:', res);
            })
            .catch(err => console.error('Stop error:', err));
    });

    function startProgressPolling() {
        if (AppState.progressPollTimer) clearInterval(AppState.progressPollTimer);
        AppState.progressPollTimer = setInterval(() => {
            fetch('/api/train/status')
                .then(res => res.json())
                .then(state => {
                    if (state.is_training) {
                        const pct = Math.max(5, state.percent);
                        genProgressBar.style.width = `${pct}%`;
                        genPctText.textContent = `${pct}%`;
                        genStatusText.textContent = state.message;
                        genLiveMetrics.innerHTML = `
                            <span><strong>Epoch:</strong> ${state.current_epoch}/${state.total_epochs}</span>
                            <span><strong>Loss:</strong> ${state.loss}</span>
                            <span><strong>Recon:</strong> ${state.recon_loss}</span>
                            <span><strong>KL:</strong> ${state.kl_loss}</span>
                        `;
                    }
                })
                .catch(() => {});
        }, 400);
    }

    function stopProgressPolling() {
        if (AppState.progressPollTimer) {
            clearInterval(AppState.progressPollTimer);
            AppState.progressPollTimer = null;
        }
    }

    function renderTrainingLossCurve(history) {
        const epochs = Array.from({ length: history.total_loss.length }, (_, i) => i + 1);

        const traceTotal = {
            x: epochs,
            y: history.total_loss,
            mode: 'lines+markers',
            name: 'Total Loss',
            line: { color: '#4f46e5', width: 2.5 }
        };

        const traceRecon = {
            x: epochs,
            y: history.recon_loss,
            mode: 'lines',
            name: 'Reconstruction Loss',
            line: { color: '#059669', width: 2 }
        };

        const traceKL = {
            x: epochs,
            y: history.kl_loss,
            mode: 'lines',
            name: 'KL Divergence (Regularization)',
            line: { color: '#d97706', width: 2, dash: 'dot' }
        };

        const layout = {
            ...plotlyLayoutBase,
            xaxis: { ...plotlyLayoutBase.xaxis, title: 'Training Epochs', gridcolor: '#e2e8f0' },
            yaxis: { ...plotlyLayoutBase.yaxis, title: 'Loss Objective', gridcolor: '#e2e8f0' },
            legend: { orientation: 'h', y: 1.15 }
        };

        Plotly.newPlot('chart-training-loss', [traceTotal, traceRecon, traceKL], layout, plotlyConfig);
    }

    // -------------------------------------------------------------
    // Tab 5: Real vs Synthetic Comparison
    // -------------------------------------------------------------
    function loadComparisonData(numCol = null, catCol = null) {
        let url = '/api/comparison';
        const params = [];
        if (numCol) params.push(`num_col=${encodeURIComponent(numCol)}`);
        if (catCol) params.push(`cat_col=${encodeURIComponent(catCol)}`);
        if (params.length) url += `?${params.join('&')}`;

        fetch(url)
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    AppState.comparisonData = data;
                    setupComparisonSelectors(data);
                    renderComparisonNumeric(data.num_distribution);
                    renderComparisonCategorical(data.cat_distribution);
                    renderComparisonCorrelations(data.correlation_comparison);
                }
            })
            .catch(err => console.log('Comparison fetch error:', err));
    }

    function setupComparisonSelectors(data) {
        const numSelect = document.getElementById('compare-num-select');
        const catSelect = document.getElementById('compare-cat-select');

        if (numSelect && data.available_num_columns && data.available_num_columns.length) {
            numSelect.innerHTML = '';
            data.available_num_columns.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c;
                opt.textContent = c;
                numSelect.appendChild(opt);
            });
            if (data.num_distribution && data.num_distribution.column) {
                numSelect.value = data.num_distribution.column;
            }
            numSelect.onchange = () => loadComparisonData(numSelect.value, catSelect ? catSelect.value : null);
        }

        if (catSelect && data.available_cat_columns && data.available_cat_columns.length) {
            catSelect.innerHTML = '';
            data.available_cat_columns.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c;
                opt.textContent = c;
                catSelect.appendChild(opt);
            });
            if (data.cat_distribution && data.cat_distribution.column) {
                catSelect.value = data.cat_distribution.column;
            }
            catSelect.onchange = () => loadComparisonData(numSelect ? numSelect.value : null, catSelect.value);
        }
    }

    function renderComparisonNumeric(dist) {
        if (!dist) return;

        document.getElementById('compare-num-title').textContent = `Distribution Alignment: ${dist.column}`;

        const traceReal = {
            x: dist.real_values,
            type: 'histogram',
            name: 'Real Data (Blue)',
            opacity: 0.7,
            marker: { 
                color: '#2563eb',
                line: { color: '#1d4ed8', width: 1.5 }
            }
        };

        const traceSynth = {
            x: dist.synth_values,
            type: 'histogram',
            name: 'Synthetic VAE Data (Orange)',
            opacity: 0.7,
            marker: { 
                color: '#f97316',
                line: { color: '#ea580c', width: 1.5 }
            }
        };

        const layout = {
            ...plotlyLayoutBase,
            barmode: 'overlay',
            xaxis: { ...plotlyLayoutBase.xaxis, title: dist.column, gridcolor: '#e2e8f0' },
            yaxis: { ...plotlyLayoutBase.yaxis, title: 'Sample Count', gridcolor: '#e2e8f0' },
            legend: { 
                orientation: 'h', 
                y: 1.15,
                font: { size: 12, weight: 600 }
            }
        };

        Plotly.newPlot('chart-compare-numeric', [traceReal, traceSynth], layout, plotlyConfig);

        const r = dist.real_stats;
        const s = dist.synth_stats;
        const statsBox = document.getElementById('num-stats-comparison-box');
        statsBox.innerHTML = `
            <div class="stats-grid-row">
                <div class="stats-item">
                    <div class="stats-item-title">Metric</div>
                    <div style="font-weight: 800; color: #2563eb; margin-top: 0.2rem;"><i class="fa-solid fa-circle" style="font-size:0.6rem; color:#2563eb; margin-right:4px;"></i>Real</div>
                    <div style="font-weight: 800; color: #ea580c; margin-top: 0.2rem;"><i class="fa-solid fa-circle" style="font-size:0.6rem; color:#ea580c; margin-right:4px;"></i>Synthetic</div>
                </div>
                <div class="stats-item">
                    <div class="stats-item-title">Mean</div>
                    <div class="stats-val-real">${r.mean}</div>
                    <div class="stats-val-synth">${s.mean}</div>
                </div>
                <div class="stats-item">
                    <div class="stats-item-title">Median</div>
                    <div class="stats-val-real">${r.median}</div>
                    <div class="stats-val-synth">${s.median}</div>
                </div>
                <div class="stats-item">
                    <div class="stats-item-title">Std Dev</div>
                    <div class="stats-val-real">${r.std}</div>
                    <div class="stats-val-synth">${s.std}</div>
                </div>
                <div class="stats-item">
                    <div class="stats-item-title">Min</div>
                    <div class="stats-val-real">${r.min}</div>
                    <div class="stats-val-synth">${s.min}</div>
                </div>
                <div class="stats-item">
                    <div class="stats-item-title">Max</div>
                    <div class="stats-val-real">${r.max}</div>
                    <div class="stats-val-synth">${s.max}</div>
                </div>
            </div>
        `;
    }

    function renderComparisonCategorical(cat) {
        if (!cat) return;

        document.getElementById('compare-cat-title').textContent = `Category Proportion: ${cat.column}`;

        const traceReal = {
            x: cat.labels,
            y: cat.real_pct,
            type: 'bar',
            name: 'Real % (Blue)',
            marker: { 
                color: '#2563eb',
                line: { color: '#1d4ed8', width: 1 }
            }
        };

        const traceSynth = {
            x: cat.labels,
            y: cat.synth_pct,
            type: 'bar',
            name: 'Synthetic % (Orange)',
            marker: { 
                color: '#f97316',
                line: { color: '#ea580c', width: 1 }
            }
        };

        const layout = {
            ...plotlyLayoutBase,
            barmode: 'group',
            xaxis: { ...plotlyLayoutBase.xaxis, title: cat.column, tickangle: -20, gridcolor: '#e2e8f0' },
            yaxis: { ...plotlyLayoutBase.yaxis, title: 'Share (%)', gridcolor: '#e2e8f0' },
            legend: { 
                orientation: 'h', 
                y: 1.15,
                font: { size: 12, weight: 600 }
            }
        };

        Plotly.newPlot('chart-compare-categorical', [traceReal, traceSynth], layout, plotlyConfig);
    }

    function renderComparisonCorrelations(corr) {
        if (!corr) return;

        const traceReal = {
            z: corr.real,
            x: corr.columns,
            y: corr.columns,
            type: 'heatmap',
            colorscale: 'Blues',
            zmin: -1,
            zmax: 1
        };

        const traceSynth = {
            z: corr.synth,
            x: corr.columns,
            y: corr.columns,
            type: 'heatmap',
            colorscale: 'Oranges',
            zmin: -1,
            zmax: 1
        };

        const corrLayout = {
            ...plotlyLayoutBase,
            margin: { t: 40, b: 80, l: 140, r: 40, automargin: true }
        };

        Plotly.newPlot('chart-compare-corr-real', [traceReal], corrLayout, plotlyConfig);
        Plotly.newPlot('chart-compare-corr-synth', [traceSynth], corrLayout, plotlyConfig);
    }

    // -------------------------------------------------------------
    // Tab 6: Quality Report & Downloads
    // -------------------------------------------------------------
    function renderQualityReport(report) {
        if (!report) return;

        const scoreElem = document.getElementById('quality-score');
        scoreElem.textContent = `${report.overall_score}%`;
        
        const badge = document.getElementById('score-status-badge');
        if (report.overall_score >= 80) {
            scoreElem.style.borderColor = 'var(--success-color)';
            badge.className = 'score-badge-status';
            badge.innerHTML = '<i class="fa-solid fa-circle-check"></i> High Fidelity Benchmark Achieved';
        } else if (report.overall_score >= 60) {
            scoreElem.style.borderColor = 'var(--warning-color)';
            badge.className = 'score-badge-status';
            badge.style.background = 'var(--warning-light)';
            badge.style.color = 'var(--warning-color)';
            badge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Moderate Statistical Fidelity';
        } else {
            scoreElem.style.borderColor = 'var(--danger-color)';
            badge.className = 'score-badge-status';
            badge.style.background = 'var(--danger-light)';
            badge.style.color = 'var(--danger-color)';
            badge.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Low Convergence Score';
        }

        document.getElementById('score-summary-text').textContent = 
            `Synthetic data captures ${report.statistical_similarity}% statistical similarity and ${report.distribution_similarity}% Wasserstein distribution alignment with an exact duplicate memorization rate of ${report.exact_duplicate_rate}%.`;

        document.getElementById('stat-score-val').textContent = `${report.statistical_similarity}%`;
        document.getElementById('dist-score-val').textContent = `${report.distribution_similarity}%`;
        document.getElementById('corr-score-val').textContent = `${report.correlation_similarity}%`;
        document.getElementById('cat-score-val').textContent = `${report.categorical_similarity}%`;
        document.getElementById('privacy-score-val').textContent = `${report.privacy_score}%`;

        const utilBadge = document.getElementById('utility-badge');
        const utilBody = document.getElementById('utility-body');
        if (report.downstream_utility) {
            const u = report.downstream_utility;
            utilBadge.textContent = `${u.utility_retention_pct}% Utility Preserved`;
            utilBody.innerHTML = `
                <div class="utility-kpi-row">
                    <div class="utility-kpi">
                        <div class="utility-kpi-label">Target Feature</div>
                        <div class="utility-kpi-val" style="font-size: 1.25rem;">${u.target_column}</div>
                    </div>
                    <div class="utility-kpi">
                        <div class="utility-kpi-label">Train-on-Real Accuracy (TRTR)</div>
                        <div class="utility-kpi-val">${u.trtr_accuracy}%</div>
                    </div>
                    <div class="utility-kpi">
                        <div class="utility-kpi-label">Train-on-Synthetic Accuracy (TSTR)</div>
                        <div class="utility-kpi-val" style="color: var(--accent-color);">${u.tstr_accuracy}%</div>
                    </div>
                </div>
            `;
        }

        const tbody = document.getElementById('report-columns-table-body');
        tbody.innerHTML = '';
        if (report.numeric_metrics) {
            report.numeric_metrics.forEach(m => {
                const tr = document.createElement('tr');
                const badgeClass = m.similarity_score >= 80 ? 'badge-score-high' : (m.similarity_score >= 60 ? 'badge-score-med' : 'badge-score-low');
                tr.innerHTML = `
                    <td style="font-weight: 700;">${m.column}</td>
                    <td>${m.real_mean}</td>
                    <td style="color: var(--synth-color); font-weight: 600;">${m.synth_mean}</td>
                    <td>${m.real_std}</td>
                    <td style="color: var(--synth-color); font-weight: 600;">${m.synth_std}</td>
                    <td>${m.wasserstein_distance}</td>
                    <td><span class="badge ${badgeClass}">${m.similarity_score}%</span></td>
                `;
                tbody.appendChild(tr);
            });
        }
    }

    document.getElementById('btn-download-csv').addEventListener('click', () => {
        window.location.href = AppState.synthDownloadUrl;
    });

    document.getElementById('btn-download-report').addEventListener('click', () => {
        window.location.href = AppState.reportDownloadUrl;
    });
});
