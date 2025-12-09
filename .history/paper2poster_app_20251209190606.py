#!/usr/bin/env python3
"""
Paper2Poster Web Application
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import subprocess
import tempfile
import shutil
import uuid
import threading
import time
from datetime import datetime
from werkzeug.utils import secure_filename

# Create Flask app instance but do not start it here
app = Flask(__name__)

JOBS = {}
RESULTS = {}

def _get_client_ip(req) -> str:
    """Best-effort client IP extraction behind proxies."""
    try:
        xff = (req.headers.get('X-Forwarded-For') or req.headers.get('X-Forwarded-for') or '').strip()
        if xff:
            first = xff.split(',')[0].strip()
            if first:
                return first
        xri = (req.headers.get('X-Real-IP') or '').strip()
        if xri:
            return xri
    except Exception:
        pass
    return req.remote_addr

def run_paper2poster_job(job_id: str, payload: dict):
    """Run a paper2poster job"""
    try:
        JOBS[job_id] = {"percent": 0, "stage": 1, "message": "Stage 1/3: Preparing environment and uploading PDF...", "done": False}
        
        openai_api_key = payload.get('openai_api_key', '')
        pdf_path = payload.get('pdf_path')
        pdf_filename = payload.get('pdf_filename')
        
        if not openai_api_key or not pdf_path or not pdf_filename:
            JOBS[job_id].update({"done": True, "error": "Missing OpenAI API key or PDF file", "message": "Error."})
            return
        
        # Use a fixed output directory
        output_dir = '/home/nuochen/paper2poster/output'
        os.makedirs(output_dir, exist_ok=True)
        
        JOBS[job_id].update({"percent": 10, "stage": 1, "message": "Stage 1/3: PDF file ready..."})
        
        # Get the filename without its extension
        pdf_name = os.path.splitext(pdf_filename)[0]
        
        JOBS[job_id].update({"percent": 20, "stage": 2, "message": "Stage 2/3: Starting Docker container... (Please wait about 5 minutes)"})
        
        # Build Docker command - keep working directory at ~/paper2poster
        pdfs_dir = os.path.dirname(pdf_path)  # Get directory containing the PDF
        docker_cmd = [
            'docker', 'run', '--rm',
            '-e', f'OPENAI_API_KEY={openai_api_key}',
            '-v', f'{pdfs_dir}:/data',
            '-v', f'{output_dir}:/output',
            '-w', '/app',
            'paper2poster:ubuntu24.04',
            'sh', '-c',
            f'python -m PosterAgent.new_pipeline '
            f'--poster_path="/data/{pdf_filename}" '
            f'--model_name_t="4o" '
            f'--model_name_v="4o" '
            f'--poster_width_inches=48 '
            f'--poster_height_inches=36 && '
            f'cp -r "<4o_4o>_generated_posters"/* /output/'
        ]
        
        JOBS[job_id].update({"percent": 30, "stage": 2, "message": "Stage 2/3: Running Docker container... (Processing in progress)"})
        
        # Run the Docker command
        print(f"Debug: Running Docker command: {' '.join(docker_cmd)}")
        process = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='/home/nuochen/paper2poster',  # Set working directory
            bufsize=1,  # Line buffering
            universal_newlines=True
        )
        
        # Monitor progress
        start_time = time.time()
        while process.poll() is None:
            elapsed = time.time() - start_time
            # Simple time-based progress estimate
            if elapsed < 60:  # First minute
                progress = 30 + int(40 * elapsed / 60)
            elif elapsed < 180:  # 1-3 minutes
                progress = 70 + int(20 * (elapsed - 60) / 120)
            else:  # Beyond 3 minutes
                progress = 90 + int(8 * min(elapsed - 180, 120) / 120)
            
            JOBS[job_id].update({
                "percent": min(98, progress),
                "stage": 2,
                "message": f"Stage 2/3: Processing... (Elapsed: {int(elapsed)}s, Please wait about 5 minutes)"
            })
            time.sleep(2)
        
        # Check command result
        try:
            # Use timeout to avoid blocking
            stdout, stderr = process.communicate(timeout=600)  # 10-minute timeout
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            JOBS[job_id].update({
                "done": True,
                "error": "Docker execution timeout (10 minutes)",
                "message": "Error."
            })
            return
        except Exception as comm_error:
            JOBS[job_id].update({
                "done": True,
                "error": f"Process communication error: {str(comm_error)}",
                "message": "Error."
            })
            return
        
        if process.returncode != 0:
            JOBS[job_id].update({
                "done": True,
                "error": f"Docker execution failed (code {process.returncode}): {stderr}",
                "message": "Error."
            })
            return
        
        JOBS[job_id].update({"percent": 95, "stage": 3, "message": "Stage 3/3: Waiting for files to be ready..."})
        
        # Wait for file copy to finish - give the Docker container time to complete file operations
        time.sleep(5)
        
        # Locate the generated PowerPoint file
        expected_output_path = os.path.join(output_dir, 'data', f'{pdf_name}.pdf', 'data.pptx')
        
        if not os.path.exists(expected_output_path):
            # Try to find any other possible output file
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.pptx'):
                        expected_output_path = os.path.join(root, file)
                        break
                if expected_output_path and os.path.exists(expected_output_path):
                    break
        
        if not os.path.exists(expected_output_path):
            JOBS[job_id].update({
                "done": True,
                "error": "Generated PowerPoint file not found",
                "message": "Error."
            })
            return
        
        # Use the located file directly as the final output (no copy needed)
        final_path = expected_output_path  # Directly use the located file path
        actual_filename = os.path.basename(final_path)  # Actual filename (e.g., data.pptx)
        download_filename = f'{pdf_name}.pptx'  # Filename shown to the user for download

        # Validate file exists
        if not os.path.exists(final_path):
            JOBS[job_id].update({
                "done": True,
                "error": "Generated file not found",
                "message": "Error."
            })
            return

        # Clean up the uploaded PDF file
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
        except:
            pass
        
        RESULTS[job_id] = {
            "success": True,
            "filename": download_filename,  # Filename presented to the user
            "actual_filename": actual_filename,  # Actual filename on disk
            "path": final_path,
            "pdf_name": pdf_name
        }
        
        JOBS[job_id].update({"percent": 100, "done": True, "message": "Completed successfully!"})
        
    except Exception as e:
        JOBS[job_id].update({"done": True, "error": str(e), "message": "Error."})
        # Clean up PDF file
        try:
            if 'pdf_path' in locals() and os.path.exists(pdf_path):
                os.remove(pdf_path)
        except:
            pass

@app.route('/paper2poster')
def paper2poster_index():
    """Paper2Poster homepage"""
    return render_template('paper2poster.html')

@app.route('/api/paper2poster/upload', methods=['POST'])
def paper2poster_upload():
    """Handle PDF upload and OpenAI API key"""
    try:
        # Check that a file was uploaded
        if 'pdf_file' not in request.files:
            return jsonify({"error": "No PDF file uploaded"}), 400
        
        pdf_file = request.files['pdf_file']
        if pdf_file.filename == '':
            return jsonify({"error": "No PDF file selected"}), 400
        
        # Validate file type
        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400
        
        # Retrieve OpenAI API key
        openai_api_key = request.form.get('openai_api_key', '').strip()
        if not openai_api_key:
            return jsonify({"error": "OpenAI API Key is required"}), 400
        
        # Create job
        job_id = uuid.uuid4().hex
        JOBS[job_id] = {"percent": 0, "stage": 0, "message": "Queued...", "done": False}
        
        # Prepare job data
        payload = {
            'openai_api_key': openai_api_key,
            'pdf_file': pdf_file
        }
        
        # Start background task
        thread = threading.Thread(target=run_paper2poster_job, args=(job_id, payload), daemon=True)
        thread.start()
        
        return jsonify({"job_id": job_id})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/paper2poster/progress')
def paper2poster_progress():
    """Get job progress"""
    job_id = request.args.get('job_id', '')
    if not job_id or job_id not in JOBS:
        return jsonify({"error": "job_id not found"}), 404
    return jsonify(JOBS[job_id])

@app.route('/api/paper2poster/result')
def paper2poster_result():
    """Get job result"""
    job_id = request.args.get('job_id', '')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    if job_id not in RESULTS:
        return jsonify({"error": "result not ready"}), 404
    return jsonify(RESULTS[job_id])

@app.route('/api/paper2poster/download/<filename>')
def paper2poster_download(filename):
    """Download the generated PowerPoint file"""
    try:
        # Look for the file in the output directory
        output_dir = '/home/nuochen/paper2poster/output'
        file_path = None
        
        # First try the data subdirectory
        data_path = os.path.join(output_dir, 'data', filename)
        if os.path.exists(data_path):
            file_path = data_path
        else:
            # Search the entire output directory
            for root, dirs, files in os.walk(output_dir):
                if filename in files:
                    file_path = os.path.join(root, filename)
                    break
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def create_paper2poster_template():
    """Create the Paper2Poster HTML template"""
    os.makedirs('templates', exist_ok=True)
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📄 Paper to Poster Generator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        input[type="text"], input[type="file"] {
            width: 100%;
            padding: 15px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus, input[type="file"]:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .btn-primary:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .btn-secondary {
            background: #f8f9fa;
            color: #333;
            border: 2px solid #e1e5e9;
        }
        
        .btn-secondary:hover {
            background: #e9ecef;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .progress-wrapper {
            margin: 20px 0;
            background: #f8f9fa;
            border: 1px solid #e1e5e9;
            border-radius: 8px;
            padding: 15px;
            display: none;
        }
        
        .progress-bar {
            width: 100%;
            height: 16px;
            background: #e9ecef;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .progress-bar-inner {
            height: 100%;
            width: 0%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }
        
        .progress-status {
            margin-top: 8px;
            font-size: 0.95em;
            color: #555;
        }
        
        .error {
            background: #ffe6e6;
            border: 1px solid #ffcccc;
            color: #cc0000;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        
        .success {
            background: #e6f4ea;
            border: 1px solid #34a853;
            color: #1e7e34;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        
        .back-link {
            text-align: center;
            margin-top: 20px;
        }
        
        .back-link a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }
        
        .back-link a:hover {
            text-decoration: underline;
        }
        
        .file-info {
            margin-top: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            font-size: 0.9em;
            color: #666;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📄 Paper to Poster Generator</h1>
            <p>Convert your research paper to a professional poster using AI</p>
        </div>
        
        <div class="content">
            <div class="error" id="errorMessage"></div>
            <div class="success" id="successMessage"></div>
            
            <form id="posterForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="openaiKey">🔑 OpenAI API Key</label>
                    <input type="text" id="openaiKey" name="openai_key" placeholder="Enter your OpenAI API key (will be deleted after use)" required>
                </div>
                
                <div class="form-group">
                    <label for="pdfFile">📁 PDF File</label>
                    <input type="file" id="pdfFile" name="pdf_file" accept=".pdf" required>
                    <div class="file-info" id="fileInfo"></div>
                </div>
                
                <div class="form-group">
                    <button type="submit" class="btn btn-primary" id="submitBtn">
                        🚀 Generate Poster
                    </button>
                    <button type="button" class="btn btn-secondary" onclick="clearForm()">
                        🗑️ Clear
                    </button>
                </div>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Please wait approximately 5 minutes for poster generation...</p>
            </div>
            
            <div class="progress-wrapper" id="progressWrapper">
                <div class="progress-bar">
                    <div id="progressInner" class="progress-bar-inner"></div>
                </div>
                <div id="progressStatus" class="progress-status">Preparing...</div>
            </div>
            
            <div class="back-link">
                <a href="/">← Back to Paper Recommendation</a>
            </div>
        </div>
    </div>

    <script>
        let currentJobId = null;
        let progressTimer = null;
        
        document.getElementById('pdfFile').addEventListener('change', function(e) {
            const file = e.target.files[0];
            const fileInfo = document.getElementById('fileInfo');
            
            if (file) {
                fileInfo.innerHTML = `
                    <strong>Selected File:</strong> ${file.name}<br>
                    <strong>Size:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB<br>
                    <strong>Type:</strong> ${file.type}
                `;
                fileInfo.style.display = 'block';
            } else {
                fileInfo.style.display = 'none';
            }
        });
        
        document.getElementById('posterForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const openaiKey = document.getElementById('openaiKey').value.trim();
            const pdfFile = document.getElementById('pdfFile').files[0];
            
            if (!openaiKey) {
                showError('Please enter your OpenAI API key');
                return;
            }
            
            if (!pdfFile) {
                showError('Please select a PDF file');
                return;
            }
            
            if (pdfFile.size > 50 * 1024 * 1024) {
                showError('File size must be less than 50MB');
                return;
            }
            
            formData.append('openai_api_key', openaiKey);
            formData.append('pdf_file', pdfFile);
            
            try {
                hideMessages();
                showLoading();
                
                const response = await fetch('/api/paper2poster/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.error || 'Failed to start poster generation');
                }
                
                currentJobId = result.job_id;
                showProgress();
                startProgressPolling(currentJobId);
                
            } catch (error) {
                hideLoading();
                showError('Network error: ' + error.message);
            }
        });
        
        async function startProgressPolling(jobId) {
            if (progressTimer) clearInterval(progressTimer);
            
            progressTimer = setInterval(async () => {
                try {
                    const response = await fetch('/api/paper2poster/progress?job_id=' + encodeURIComponent(jobId));
                    const progress = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(progress.error || 'Progress check failed');
                    }
                    
                    updateProgress(progress.percent || 0, progress.message || 'Processing...');
                    
                    if (progress.done) {
                        clearInterval(progressTimer);
                        hideLoading();
                        hideProgress();
                        
                        if (progress.error) {
                            showError(progress.error);
                        } else {
                            showSuccess('Poster generated successfully! Download will start automatically.');
                            // Auto-download the poster
                            downloadPoster(jobId);
                        }
                    }
                } catch (error) {
                    clearInterval(progressTimer);
                    hideLoading();
                    hideProgress();
                    showError('Progress check failed: ' + error.message);
                }
            }, 2000);
        }
        
        function updateProgress(percent, message) {
            const progressInner = document.getElementById('progressInner');
            const progressStatus = document.getElementById('progressStatus');
            
            progressInner.style.width = Math.max(0, Math.min(100, percent)) + '%';
            progressStatus.textContent = message;
        }
        
        async function downloadPoster(jobId) {
            try {
                // First retrieve result metadata
                const resultResponse = await fetch('/api/paper2poster/result?job_id=' + encodeURIComponent(jobId));
                
                if (!resultResponse.ok) {
                    throw new Error('Failed to get result');
                }
                
                const result = await resultResponse.json();
                
                if (!result.success) {
                    throw new Error('Generation failed');
                }
                
                // Use the correct filename and pass the job_id for download
                const response = await fetch('/api/paper2poster/download/' + encodeURIComponent(result.filename) + '?job_id=' + encodeURIComponent(jobId));
                
                if (!response.ok) {
                    throw new Error('Download failed');
                }
                
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = result.filename;  // Use the actual filename
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
            } catch (error) {
                showError('Download failed: ' + error.message);
            }
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('errorMessage');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }
        
        function showSuccess(message) {
            const successDiv = document.getElementById('successMessage');
            successDiv.textContent = message;
            successDiv.style.display = 'block';
        }
        
        function hideMessages() {
            document.getElementById('errorMessage').style.display = 'none';
            document.getElementById('successMessage').style.display = 'none';
        }
        
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('submitBtn').disabled = true;
        }
        
        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('submitBtn').disabled = false;
        }
        
        function showProgress() {
            document.getElementById('progressWrapper').style.display = 'block';
        }
        
        function hideProgress() {
            document.getElementById('progressWrapper').style.display = 'none';
        }
        
        function clearForm() {
            document.getElementById('posterForm').reset();
            document.getElementById('fileInfo').style.display = 'none';
            hideMessages();
            hideLoading();
            hideProgress();
            if (progressTimer) {
                clearInterval(progressTimer);
                progressTimer = null;
            }
        }
    </script>
</body>
</html>
    """
    
    with open('templates/paper2poster.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

# Initialization helper for the main application
def init_paper2poster():
    """Initialize the paper2poster module"""
    # Ensure required directories exist
    os.makedirs('/home/nuochen/paper2poster/output/data', exist_ok=True)
    os.makedirs('/home/nuochen/paper2poster/pdfs', exist_ok=True)
    
    # Create the template file
    create_paper2poster_template()
    
    print("✓ Paper2Poster module initialized")

# The original __main__ startup code has been removed