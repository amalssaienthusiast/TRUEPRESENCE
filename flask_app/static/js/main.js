document.addEventListener('DOMContentLoaded', function() {
    // Get button elements
    const getFacesBtn = document.getElementById('get-faces-btn');
    const extractFeaturesBtn = document.getElementById('extract-features-btn');
    const attendanceBtn = document.getElementById('attendance-btn');
    const stopBtn = document.getElementById('stop-btn');
    
    // Get status elements
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.getElementById('status-text');
    const currentProcess = document.getElementById('current-process');
    const outputConsole = document.getElementById('output-console');
    
    // Set up status polling
    let statusInterval;
    
    // Button click handlers
    getFacesBtn.addEventListener('click', function() {
        runScript('get_faces');
    });
    
    extractFeaturesBtn.addEventListener('click', function() {
        runScript('extract_features');
    });
    
    attendanceBtn.addEventListener('click', function() {
        runScript('attendance');
    });
    
    stopBtn.addEventListener('click', function() {
        stopProcess();
    });
    
    // Function to run a script
    function runScript(scriptName) {
        // Disable all buttons during request
        setButtonsEnabled(false);
        
        fetch('/run_script', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'script=' + scriptName
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Start polling for status updates
                startStatusPolling();
                showAlert(`Started ${scriptName} process`, 'success');
            } else {
                showAlert(data.message, 'error');
                setButtonsEnabled(true);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert('Failed to start process: ' + error, 'error');
            setButtonsEnabled(true);
        });
    }
    
    // Function to stop the current process
    function stopProcess() {
        fetch('/stop_script', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showAlert('Process stopped', 'success');
                // Will be updated in next status poll
            } else {
                showAlert(data.message, 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert('Failed to stop process: ' + error, 'error');
        });
    }
    
    // Start polling for status updates
    function startStatusPolling() {
        // Clear any existing interval
        if (statusInterval) {
            clearInterval(statusInterval);
        }
        
        // Poll immediately
        checkStatus();
        
        // Set up polling interval (every 1 second)
        statusInterval = setInterval(checkStatus, 1000);
    }
    
    // Check the current process status
    function checkStatus() {
        fetch('/status')
        .then(response => response.json())
        .then(data => {
            updateStatusUI(data);
        })
        .catch(error => {
            console.error('Status check error:', error);
        });
    }
    
    // Update the UI based on status data
    function updateStatusUI(data) {
        if (data.running) {
            // Process is running
            statusDot.className = 'status-dot active';
            statusText.textContent = 'Process running';
            currentProcess.textContent = data.script || 'Unknown process';
            stopBtn.style.display = 'inline-flex';
            setButtonsEnabled(false);
        } else {
            // No process running
            statusDot.className = 'status-dot inactive';
            statusText.textContent = 'No process running';
            if (data.script) {
                currentProcess.textContent = `${data.script} (completed)`;
            } else {
                currentProcess.textContent = 'None';
            }
            stopBtn.style.display = 'none';
            setButtonsEnabled(true);
            
            // If not running, clear the interval
            if (statusInterval) {
                clearInterval(statusInterval);
                statusInterval = null;
            }
        }
        
        // Update console output
        if (data.output) {
            outputConsole.textContent = data.output;
            outputConsole.scrollTop = outputConsole.scrollHeight; // Auto-scroll to bottom
        }
    }
    
    // Enable or disable all action buttons
    function setButtonsEnabled(enabled) {
        getFacesBtn.disabled = !enabled;
        extractFeaturesBtn.disabled = !enabled;
        attendanceBtn.disabled = !enabled;
        
        // Add visual feedback for disabled state
        const buttons = [getFacesBtn, extractFeaturesBtn, attendanceBtn];
        buttons.forEach(btn => {
            if (enabled) {
                btn.style.opacity = '1';
                btn.style.cursor = 'pointer';
            } else {
                btn.style.opacity = '0.6';
                btn.style.cursor = 'not-allowed';
            }
        });
    }
    
    // Show alert message
    function showAlert(message, type) {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert ${type}`;
        alertDiv.textContent = message;
        
        // Add styles
        alertDiv.style.position = 'fixed';
        alertDiv.style.top = '20px';
        alertDiv.style.left = '50%';
        alertDiv.style.transform = 'translateX(-50%)';
        alertDiv.style.padding = '10px 20px';
        alertDiv.style.borderRadius = '4px';
        alertDiv.style.zIndex = '1000';
        
        if (type === 'success') {
            alertDiv.style.backgroundColor = '#d4edda';
            alertDiv.style.color = '#155724';
            alertDiv.style.border = '1px solid #c3e6cb';
        } else {
            alertDiv.style.backgroundColor = '#f8d7da';
            alertDiv.style.color = '#721c24';
            alertDiv.style.border = '1px solid #f5c6cb';
        }
        
        // Add to document
        document.body.appendChild(alertDiv);
        
        // Remove after 3 seconds
        setTimeout(() => {
            alertDiv.style.opacity = '0';
            alertDiv.style.transition = 'opacity 0.5s';
            setTimeout(() => {
                document.body.removeChild(alertDiv);
            }, 500);
        }, 3000);
    }
    
    // Check status on initial load
    checkStatus();
}); 