"""
09_start_frontend_service.py
Pipeline step to start the Streamlit frontend after training completes.
Can be enabled/disabled via environment variable.
Requires the API service to be running (08_start_api_service.py).
"""
import os
import sys
import subprocess
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_api_available():
    """Check if API service is available."""
    try:
        import requests
        api_url = os.getenv('API_URL', 'http://localhost:8000')
        response = requests.get(f"{api_url}/", timeout=5)
        if response.status_code == 200:
            logger.info(f"API service is available at {api_url}")
            return True
    except Exception as e:
        logger.warning(f"API service not available: {e}")
    return False


def start_frontend_service():
    """Start the Streamlit frontend service."""
    logger.info("=" * 60)
    logger.info("Starting Frontend Service (Step 09)")
    logger.info("=" * 60)
    
    # Check if we should start the service
    start_service = os.getenv('START_FRONTEND_SERVICE', '0').strip()
    if start_service not in ('1', 'true', 'True', 'TRUE', 'yes', 'Yes', 'YES'):
        logger.info("Frontend service start is disabled (START_FRONTEND_SERVICE not set to 1/true/yes)")
        logger.info("To enable, set environment variable: START_FRONTEND_SERVICE=1")
        logger.info("Skipping frontend service startup.")
        return 0
    
    # Check if API is available (optional warning)
    if not check_api_available():
        logger.warning("API service is not available!")
        logger.warning("Frontend will start but may not work correctly without the API.")
        logger.warning("Consider setting START_API_SERVICE=1 to start the API first.")
        time.sleep(2)  # Give user time to read warning
    
    # Determine host and port
    frontend_host = os.getenv('FRONTEND_HOST', '0.0.0.0')
    frontend_port = int(os.getenv('FRONTEND_PORT', '8501'))
    api_url = os.getenv('API_URL', 'http://localhost:8000')
    
    logger.info(f"Starting Streamlit frontend at {frontend_host}:{frontend_port}")
    logger.info(f"Frontend will connect to API at: {api_url}")
    logger.info("Frontend will be accessible at:")
    logger.info(f"  - http://localhost:{frontend_port} (from host)")
    logger.info(f"  - http://{frontend_host}:{frontend_port} (from network)")
    logger.info("")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 60)
    
    # Get current script directory
    script_dir = Path(__file__).parent
    frontend_app_path = script_dir / 'frontend' / 'app.py'
    
    if not frontend_app_path.exists():
        logger.error(f"Frontend app not found at: {frontend_app_path}")
        return 1
    
    # Start Streamlit server
    try:
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(frontend_app_path),
            f"--server.port={frontend_port}",
            f"--server.address={frontend_host}",
            "--server.headless=true"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=script_dir)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start frontend service: {e}")
        return 1
    except FileNotFoundError:
        logger.error("streamlit not installed! Install it with: pip install streamlit")
        return 1
    except KeyboardInterrupt:
        logger.info("\nFrontend service stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Error starting frontend service: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(start_frontend_service())
