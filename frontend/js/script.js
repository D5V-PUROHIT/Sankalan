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
        reportDownloadUrl: '/api/download/report'
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
            desc: 'Tune latent hyperparameters, train the encoder-decoder network, and generate synthetic rows'
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

    // Plotly Default Theme
    const plotlyLayoutBase = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#0f172a', family: 'Plus Jakarta Sans, sans-serif' },
        margin: { t: 40, b: 40, l: 50, r: 30 },
        autosize: true
    };

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

        // Lazy load charts when relevant tab is selected
        if (targetTab === 'tab-eda' && !AppState.edaData) {
            loadEDAData();
        } else if (targetTab === 'tab-compare') {
            loadComparisonData();
        }
    }

    // -------------------------------------------------------------
    // Initial System Status Check
    // -------------------------------------------------------------
    checkSystemStatus();

    function checkSystemStatus() {
        fetch('/api/status')
            .then(res => res.json())
            .then(status => {
                const sysStatus = document.getElementById('system-status');
                if (status.dataset_loaded) {
                    sysStatus.textContent = `Dataset Ready: ${status.raw_filename}`;
                    document.getElementById('demo-banner').style.display = 'flex';
                    // Auto-fetch EDA if data is ready
                    loadEDAData(false);
                }
                if (status.synthetic_ready) {
                    document.getElementById('btn-download-csv').disabled = false;
                    document.getElementById('btn-download-report').disabled = false;
                    if (status.latest_quality_score) {
                        document.getElementById('quality-score').textContent = `${status.latest_quality_score}%`;
                    }
                }
            })
            .catch(err => console.log('Status check error:', err));
    }

    // Demo dataset button handler
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
                AppState.edaData = null; // force reload EDA

                uploadText.textContent = 'Preprocessing complete! Redirecting to Overview...';
                document.getElementById('system-status').textContent = `Dataset: ${data.filename}`;

                renderOverviewTable(data.profile, data.preprocessing);

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
                <td style="font-weight: 600;">${col.name}</td>
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

    // Search filter for columns table
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

                    // Build overview if not yet loaded
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
        document.getElementById('eda-dist-tag').textContent = `Mean: ${dist.mean} | Std: ${dist.std}`;

        const trace = {
            x: dist.values,
            type: 'histogram',
            name: colName,
            marker: {
                color: '#4f46e5',
                line: { color: '#3730a3', width: 1 }
            },
            opacity: 0.75
        };

        const layout = {
            ...plotlyLayoutBase,
            xaxis: { title: colName, gridcolor: '#e2e8f0' },
            yaxis: { title: 'Frequency', gridcolor: '#e2e8f0' },
            bargap: 0.05
        };

        Plotly.newPlot('chart-eda-distribution', [trace], layout, { responsive: true, displayModeBar: false });
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
            xaxis: { title: 'Classes', tickangle: -25 },
            yaxis: { title: 'Count', gridcolor: '#e2e8f0' }
        };

        Plotly.newPlot('chart-eda-category', [trace], layout, { responsive: true, displayModeBar: false });
    }

    function renderEDACorrelation(corr) {
        if (!corr) return;

        const trace = {
            z: corr.z,
            x: corr.x,
            y: corr.y,
            type: 'heatmap',
            colorscale: 'Viridis',
            colorbar: { title: 'Pearson r' }
        };

        const layout = {
            ...plotlyLayoutBase,
            margin: { t: 40, b: 70, l: 80, r: 40 }
        };

        Plotly.newPlot('chart-eda-correlation', [trace], layout, { responsive: true, displayModeBar: false });
    }

    // -------------------------------------------------------------
    // Tab 4: VAE Training & Synthesis
    // -------------------------------------------------------------
    const btnGenerate = document.getElementById('btn-generate');
    const genProgress = document.getElementById('generation-progress');
    const genProgressBar = document.getElementById('gen-progress-bar');
    const genStatusText = document.getElementById('gen-status-text');
    const genPctText = document.getElementById('gen-pct-text');

    btnGenerate.addEventListener('click', () => {
        const numRecords = parseInt(document.getElementById('num-records').value) || 5000;
        const epochs = parseInt(document.getElementById('epochs').value) || 40;
        const latentDim = parseInt(document.getElementById('latent-dim').value) || 8;

        btnGenerate.disabled = true;
        genProgress.classList.remove('hidden');
        genProgressBar.style.width = '15%';
        genPctText.textContent = '15%';
        genStatusText.textContent = `Training VAE (${epochs} epochs, latent dim=${latentDim})...`;

        // Simulate step progress while waiting
        let progress = 15;
        const progressTimer = setInterval(() => {
            if (progress < 85) {
                progress += Math.floor(Math.random() * 8) + 2;
                genProgressBar.style.width = `${progress}%`;
                genPctText.textContent = `${progress}%`;
            }
        }, 800);

        fetch(`/api/generate?num_records=${numRecords}&epochs=${epochs}&latent_dim=${latentDim}`, {
            method: 'POST'
        })
        .then(res => res.json())
        .then(data => {
            clearInterval(progressTimer);
            if (data.quality_report) {
                genProgressBar.style.width = '100%';
                genPctText.textContent = '100%';
                genStatusText.textContent = 'Training & Synthesis complete!';

                AppState.qualityReport = data.quality_report;
                AppState.trainingHistory = data.training_history;

                // Render loss curve
                if (data.training_history) {
                    renderTrainingLossCurve(data.training_history);
                }

                // Populate Quality Report
                renderQualityReport(data.quality_report);

                // Enable download buttons
                document.getElementById('btn-download-csv').disabled = false;
                document.getElementById('btn-download-report').disabled = false;

                // Auto switch to comparison tab
                setTimeout(() => {
                    btnGenerate.disabled = false;
                    switchTab('tab-compare');
                }, 1200);
            } else {
                throw new Error(data.message || 'Generation failed');
            }
        })
        .catch(err => {
            clearInterval(progressTimer);
            btnGenerate.disabled = false;
            genStatusText.textContent = `Error: ${err.message}`;
            genStatusText.style.color = '#dc2626';
            console.error('Generation Error:', err);
        });
    });

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
            line: { color: '#059669', width: 2, dash: 'solid' }
        };

        const traceKL = {
            x: epochs,
            y: history.kl_loss,
            mode: 'lines',
            name: 'KL Divergence (Latent Regularization)',
            line: { color: '#d97706', width: 2, dash: 'dot' }
        };

        const layout = {
            ...plotlyLayoutBase,
            xaxis: { title: 'Epochs', gridcolor: '#e2e8f0' },
            yaxis: { title: 'Objective Loss', gridcolor: '#e2e8f0' },
            legend: { orientation: 'h', y: 1.15 }
        };

        Plotly.newPlot('chart-training-loss', [traceTotal, traceRecon, traceKL], layout, { responsive: true });
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

        if (numSelect.children.length === 0 && data.available_num_columns) {
            data.available_num_columns.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c;
                opt.textContent = c;
                numSelect.appendChild(opt);
            });
            numSelect.onchange = () => loadComparisonData(numSelect.value, catSelect.value);
        }

        if (catSelect.children.length === 0 && data.available_cat_columns) {
            data.available_cat_columns.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c;
                opt.textContent = c;
                catSelect.appendChild(opt);
            });
            catSelect.onchange = () => loadComparisonData(numSelect.value, catSelect.value);
        }
    }

    function renderComparisonNumeric(dist) {
        if (!dist) return;

        document.getElementById('compare-num-title').textContent = `Distribution Alignment: ${dist.column}`;

        const traceReal = {
            x: dist.real_values,
            type: 'histogram',
            name: 'Real Data',
            opacity: 0.65,
            marker: { color: '#4f46e5' }
        };

        const traceSynth = {
            x: dist.synth_values,
            type: 'histogram',
            name: 'Synthetic VAE Data',
            opacity: 0.65,
            marker: { color: '#7c3aed' }
        };

        const layout = {
            ...plotlyLayoutBase,
            barmode: 'overlay',
            xaxis: { title: dist.column, gridcolor: '#e2e8f0' },
            yaxis: { title: 'Density / Count', gridcolor: '#e2e8f0' },
            legend: { orientation: 'h', y: 1.15 }
        };

        Plotly.newPlot('chart-compare-numeric', [traceReal, traceSynth], layout, { responsive: true, displayModeBar: false });

        // Update side-by-side stats comparison box
        const r = dist.real_stats;
        const s = dist.synth_stats;
        const statsBox = document.getElementById('num-stats-comparison-box');
        statsBox.innerHTML = `
            <div class="stats-grid-row">
                <div class="stats-item">
                    <div class="stats-item-title">Metric</div>
                    <div style="font-weight: 700; color: #4f46e5; margin-top: 0.2rem;">Real</div>
                    <div style="font-weight: 700; color: #7c3aed;">Synthetic</div>
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

        document.getElementById('compare-cat-title').textContent = `Category Proportion Alignment: ${cat.column}`;

        const traceReal = {
            x: cat.labels,
            y: cat.real_pct,
            type: 'bar',
            name: 'Real %',
            marker: { color: '#4f46e5' }
        };

        const traceSynth = {
            x: cat.labels,
            y: cat.synth_pct,
            type: 'bar',
            name: 'Synthetic %',
            marker: { color: '#7c3aed' }
        };

        const layout = {
            ...plotlyLayoutBase,
            barmode: 'group',
            xaxis: { title: 'Category', tickangle: -20 },
            yaxis: { title: 'Proportion (%)', gridcolor: '#e2e8f0' },
            legend: { orientation: 'h', y: 1.15 }
        };

        Plotly.newPlot('chart-compare-categorical', [traceReal, traceSynth], layout, { responsive: true, displayModeBar: false });
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
            colorscale: 'Purples',
            zmin: -1,
            zmax: 1
        };

        Plotly.newPlot('chart-compare-corr-real', [traceReal], plotlyLayoutBase, { responsive: true, displayModeBar: false });
        Plotly.newPlot('chart-compare-corr-synth', [traceSynth], plotlyLayoutBase, { responsive: true, displayModeBar: false });
    }

    // -------------------------------------------------------------
    // Tab 6: Quality Report & Downloads
    // -------------------------------------------------------------
    function renderQualityReport(report) {
        if (!report) return;

        // Overall Score
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

        // Metric Breakdown Cards
        document.getElementById('stat-score-val').textContent = `${report.statistical_similarity}%`;
        document.getElementById('dist-score-val').textContent = `${report.distribution_similarity}%`;
        document.getElementById('corr-score-val').textContent = `${report.correlation_similarity}%`;
        document.getElementById('cat-score-val').textContent = `${report.categorical_similarity}%`;
        document.getElementById('privacy-score-val').textContent = `${report.privacy_score}%`;

        // Downstream Utility
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

        // Column Benchmarking Table
        const tbody = document.getElementById('report-columns-table-body');
        tbody.innerHTML = '';
        if (report.numeric_metrics) {
            report.numeric_metrics.forEach(m => {
                const tr = document.createElement('tr');
                const badgeClass = m.similarity_score >= 80 ? 'badge-score-high' : (m.similarity_score >= 60 ? 'badge-score-med' : 'badge-score-low');
                tr.innerHTML = `
                    <td style="font-weight: 700;">${m.column}</td>
                    <td>${m.real_mean}</td>
                    <td style="color: var(--accent-color); font-weight: 600;">${m.synth_mean}</td>
                    <td>${m.real_std}</td>
                    <td style="color: var(--accent-color); font-weight: 600;">${m.synth_std}</td>
                    <td>${m.wasserstein_distance}</td>
                    <td><span class="badge ${badgeClass}">${m.similarity_score}%</span></td>
                `;
                tbody.appendChild(tr);
            });
        }
    }

    // Download Button Actions
    document.getElementById('btn-download-csv').addEventListener('click', () => {
        window.location.href = AppState.synthDownloadUrl;
    });

    document.getElementById('btn-download-report').addEventListener('click', () => {
        window.location.href = AppState.reportDownloadUrl;
    });
});
