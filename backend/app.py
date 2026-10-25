"""
================================================================================
SANKALAN AI PLATFORM - APPLICATION LAUNCHER
File: backend/app.py
Notice: The main application & FastAPI routing logic is organized in:
        backend/server_app.py
================================================================================
This file serves as a convenient launcher and re-exports app for compatibility.
"""

from server_app import app, uvicorn

if __name__ == "__main__":
    print("===================================================")
    print("       Sankalan AI Platform - FastAPI Server       ")
    print("       Web Dashboard: http://127.0.0.1:8000        ")
    print("===================================================")
    uvicorn.run("server_app:app", host="127.0.0.1", port=8000, reload=True)
