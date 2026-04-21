# Wait for the currently running gee_pipeline.py process (ID 11512) to finish
Write-Host "Waiting for data extraction to complete..."
Wait-Process -Id 11512 -ErrorAction SilentlyContinue

Write-Host "Data extraction finished. Starting HydroFormer training..."
c:\Users\ankit\OneDrive\Desktop\InlandRoute\backend\venv\Scripts\python.exe train.py
Write-Host "Training complete! Check ml/models/saved for the PyTorch artifacts."
