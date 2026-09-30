document.addEventListener('DOMContentLoaded', () => {
    // Tab Navigation Logic
    const navLinks = document.querySelectorAll('.nav-links li');
    const tabPanes = document.querySelectorAll('.tab-pane');

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            // Remove active from all
            navLinks.forEach(l => l.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            // Add active to clicked
            link.classList.add('active');
            const targetTab = link.getAttribute('data-tab');
            document.getElementById(targetTab).classList.add('active');
            
            // Load EDA charts dynamically when tab is clicked
            if (targetTab === 'tab-eda' && !window.edaLoaded) {
                loadEDACharts();
            }
        });
    });
    
    function loadEDACharts() {
        fetch('/api/eda')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    window.edaLoaded = true;
                    
                    // Render Distribution Chart
                    if (data.distribution) {
                        const distData = [{
                            x: data.distribution.values,
                            type: 'histogram',
                            marker: { color: 'var(--accent-color)' },
                            opacity: 0.7
                        }];
                        const distLayout = {
                            title: `Distribution of ${data.distribution.column_name}`,
                            paper_bgcolor: 'transparent',
                            plot_bgcolor: 'transparent',
                            font: { color: 'var(--text-primary)', family: 'Inter' },
                            margin: { t: 40, b: 40, l: 40, r: 40 }
                        };
                        Plotly.newPlot('chart-distribution', distData, distLayout, {responsive: true});
                    }
                    
                    // Render Correlation Heatmap
                    if (data.correlation) {
                        const corrData = [{
                            z: data.correlation.z,
                            x: data.correlation.x,
                            y: data.correlation.y,
                            type: 'heatmap',
                            colorscale: 'Blues'
                        }];
                        const corrLayout = {
                            title: 'Numerical Correlation Matrix',
                            paper_bgcolor: 'transparent',
                            plot_bgcolor: 'transparent',
                            font: { color: 'var(--text-primary)', family: 'Inter' },
                            margin: { t: 40, b: 60, l: 60, r: 40 }
                        };
                        Plotly.newPlot('chart-correlation', corrData, corrLayout, {responsive: true});
                    }
                }
            })
            .catch(err => console.error("Error loading EDA:", err));
    }

    // File Upload Logic
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const uploadText = document.getElementById('upload-text');

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--accent-color)';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border-color)';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border-color)';
        if(e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if(e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });

    function handleFileUpload(file) {
        if(!file.name.endsWith('.csv') && !file.name.endsWith('.xlsx')) {
            alert('Please upload a valid CSV or Excel file.');
            return;
        }

        // UI Feedback
        dropZone.classList.add('hidden');
        uploadStatus.classList.remove('hidden');
        uploadText.textContent = "Uploading and profiling dataset...";
        
        // Create FormData to send the file
        const formData = new FormData();
        formData.append('file', file);

        // Send file to FastAPI backend
        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.profile) {
                uploadText.textContent = "Profiling complete. Ready for EDA.";
                document.getElementById('system-status').textContent = "Dataset Loaded";
                
                // Auto switch to overview tab after a short delay
                setTimeout(() => {
                    document.querySelector('[data-tab="tab-overview"]').click();
                    uploadStatus.classList.add('hidden');
                    dropZone.classList.remove('hidden');
                    
                    const p = data.profile;
                    const pre = data.preprocessing;
                    
                    // Dynamically inject the actual profile data returned from Pandas
                    document.getElementById('overview-table').innerHTML = `
                        <div style="background: var(--bg-color); border: 1px solid var(--border-color); padding: 1.5rem; border-radius: 8px; text-align: left; margin-bottom: 1.5rem;">
                            <p style="color: var(--success-color); font-weight: 600; margin-bottom: 0.5rem;"><i class="fa-solid fa-circle-check"></i> Successfully Profiled & Cleaned ${data.filename}</p>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                                <div>
                                    <p style="color: var(--text-primary); margin-bottom: 0.25rem;"><strong>Original Rows:</strong> ${p.num_rows.toLocaleString()}</p>
                                    <p style="color: var(--text-secondary); margin-bottom: 0.25rem;"><strong>Missing Values Detected:</strong> ${p.total_missing_values.toLocaleString()}</p>
                                    <p style="color: var(--text-secondary);"><strong>Duplicate Rows Found:</strong> ${p.duplicate_count.toLocaleString()}</p>
                                </div>
                                <div style="border-left: 2px solid var(--border-color); padding-left: 1rem;">
                                    <p style="color: var(--text-primary); margin-bottom: 0.25rem;"><strong>Column Role Classification:</strong></p>
                                    <p style="color: var(--text-secondary); margin-bottom: 0.25rem;">- Generative (Numeric): ${pre.roles_assigned.numeric}</p>
                                    <p style="color: var(--text-secondary); margin-bottom: 0.25rem;">- Generative (Categorical): ${pre.roles_assigned.categorical}</p>
                                    <p style="color: var(--text-secondary); margin-bottom: 0.25rem;">- Ignored (Dates/IDs): ${pre.roles_assigned.dates + pre.roles_assigned.identifiers}</p>
                                    <p style="color: var(--text-secondary); margin-bottom: 0.25rem;">- VAE Input Dimension: ${pre.vae_input_dim}</p>
                                </div>
                            </div>
                        </div>
                        <div style="margin-top: 1rem; text-align: left;">
                            <h3 style="margin-bottom: 1rem;">Column Details</h3>
                            <table style="width: 100%; border-collapse: collapse; text-align: left;">
                                <thead style="border-bottom: 2px solid var(--border-color);">
                                    <tr>
                                        <th style="padding: 0.5rem;">Column Name</th>
                                        <th style="padding: 0.5rem;">Type</th>
                                        <th style="padding: 0.5rem;">Unique Values</th>
                                        <th style="padding: 0.5rem;">Missing %</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${p.columns.map(c => `
                                        <tr style="border-bottom: 1px solid var(--border-color);">
                                            <td style="padding: 0.5rem; font-weight: 500;">${c.name}</td>
                                            <td style="padding: 0.5rem;">${c.type} (${c.dtype})</td>
                                            <td style="padding: 0.5rem;">${c.unique_count.toLocaleString()}</td>
                                            <td style="padding: 0.5rem; color: ${c.missing_percentage > 0 ? '#ef4444' : 'var(--text-secondary)'};">${c.missing_percentage}%</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    `;
                }, 1000);
            } else {
                throw new Error(data.message || "Unknown error occurred");
            }
        })
        .catch(error => {
            console.error('Upload Error:', error);
            uploadText.textContent = "Error: " + error.message;
            uploadText.style.color = "#ef4444";
            document.querySelector('.loader').style.display = 'none';
            
            // Allow retry
            setTimeout(() => {
                uploadStatus.classList.add('hidden');
                dropZone.classList.remove('hidden');
                uploadText.style.color = "inherit";
                document.querySelector('.loader').style.display = 'block';
                uploadText.textContent = "Uploading and profiling dataset...";
            }, 3000);
        });
    }

    // Generation Logic
    const btnGenerate = document.getElementById('btn-generate');
    const genProgress = document.getElementById('generation-progress');
    const genProgressBar = document.getElementById('gen-progress-bar');
    const genStatusText = document.getElementById('gen-status-text');

    btnGenerate.addEventListener('click', () => {
        const numRecords = document.getElementById('num-records').value || 5000;
        const epochs = document.getElementById('epochs').value || 50;

        genProgress.classList.remove('hidden');
        btnGenerate.disabled = true;
        btnGenerate.style.opacity = '0.5';
        
        genProgressBar.style.width = '10%';
        genStatusText.textContent = "Training VAE model and generating records. This may take a moment...";

        // Send generation request to FastAPI
        fetch(`/api/generate?num_records=${numRecords}&epochs=${epochs}`, {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.quality_report) {
                genProgressBar.style.width = '100%';
                genProgressBar.style.background = 'var(--success-color)';
                genStatusText.textContent = "Generation Complete!";
                
                setTimeout(() => {
                    // Go to Quality Report tab
                    document.querySelector('[data-tab="tab-report"]').click();
                    
                    const r = data.quality_report;
                    document.getElementById('quality-score').textContent = `${r.overall_score}`;
                    
                    // Add details to the report tab
                    const scoreCard = document.querySelector('.score-card');
                    if (!document.getElementById('report-details')) {
                        const details = document.createElement('div');
                        details.id = 'report-details';
                        details.style.marginTop = '2rem';
                        details.innerHTML = `
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; text-align: left;">
                                <div style="background: var(--bg-color); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border-color);">
                                    <h4 style="color: var(--text-secondary); margin-bottom: 0.5rem; text-transform: uppercase; font-size: 0.8rem;">Statistical Similarity</h4>
                                    <p style="font-size: 1.5rem; font-weight: bold; color: var(--text-primary);">${r.statistical_similarity}/100</p>
                                </div>
                                <div style="background: var(--bg-color); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border-color);">
                                    <h4 style="color: var(--text-secondary); margin-bottom: 0.5rem; text-transform: uppercase; font-size: 0.8rem;">Correlation Preservation</h4>
                                    <p style="font-size: 1.5rem; font-weight: bold; color: var(--text-primary);">${r.correlation_similarity}/100</p>
                                </div>
                                <div style="background: var(--bg-color); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border-color);">
                                    <h4 style="color: var(--text-secondary); margin-bottom: 0.5rem; text-transform: uppercase; font-size: 0.8rem;">Category Preservation</h4>
                                    <p style="font-size: 1.5rem; font-weight: bold; color: var(--text-primary);">${r.categorical_similarity}/100</p>
                                </div>
                            </div>
                        `;
                        scoreCard.appendChild(details);
                    }
                    
                    // Setup download button
                    document.getElementById('btn-download-csv').onclick = () => {
                        window.location.href = data.download_url;
                    };
                    
                }, 1000);
            } else {
                throw new Error(data.message || "Generation failed.");
            }
        })
        .catch(error => {
            console.error('Generation Error:', error);
            genStatusText.textContent = "Error: " + error.message;
            genStatusText.style.color = "#ef4444";
            genProgressBar.style.background = '#ef4444';
            
            setTimeout(() => {
                btnGenerate.disabled = false;
                btnGenerate.style.opacity = '1';
                genStatusText.style.color = "inherit";
                genProgressBar.style.background = 'var(--accent-color)';
                genProgress.classList.add('hidden');
            }, 5000);
        });
    });
});
