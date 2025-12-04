"""
08_start_api_service.py
Pipeline step to start the API service after training completes.
Can be enabled/disabled via environment variable.
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_models_exist():
    """Check if required models are available."""
    output_dir = Path(os.getenv('OUTPUT_DIR', '/app/output'))
    models_dir = output_dir / 'models'
    
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return False
    
    transformer_model = models_dir / 'best_transformer_model'
    baseline_model = models_dir / 'baseline_model.pkl'
    
    if not transformer_model.exists():
        logger.warning("Transformer model not found!")
        
    if not baseline_model.exists():
        logger.warning("Baseline model not found!")
    
    # At least one model should exist
    if transformer_model.exists() or baseline_model.exists():
        logger.info("Models found, ready to start API service")
        return True
    else:
        logger.error("No models found! Cannot start API service.")
        return False


def start_api_service():
    """Start the FastAPI service."""
    logger.info("=" * 60)
    logger.info("Starting API Service (Step 08)")
    logger.info("=" * 60)
    
    # Check if we should start the service
    start_service = os.getenv('START_API_SERVICE', '0').strip()
    if start_service not in ('1', 'true', 'True', 'TRUE', 'yes', 'Yes', 'YES'):
        logger.info("API service start is disabled (START_API_SERVICE not set to 1/true/yes)")
        logger.info("To enable, set environment variable: START_API_SERVICE=1")
        logger.info("Skipping API service startup.")
        return 0
    
    # Check models
    if not check_models_exist():
        logger.error("Cannot start API service without models!")
        return 1
    
    # Determine host and port
    api_host = os.getenv('API_HOST', '0.0.0.0')
    api_port = int(os.getenv('API_PORT', '8000'))
    
    logger.info(f"Starting FastAPI server at {api_host}:{api_port}")
    logger.info("API will be accessible at:")
    logger.info(f"  - http://localhost:{api_port} (from host)")
    logger.info(f"  - http://{api_host}:{api_port} (from network)")
    logger.info("")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 60)
    
    # Get current script directory
    script_dir = Path(__file__).parent
    api_app_path = script_dir / 'api' / 'app.py'
    
    if not api_app_path.exists():
        logger.error(f"API app not found at: {api_app_path}")
        return 1
    
    # Start uvicorn server
    try:
        # Import here to avoid dependency issues if not needed
        import uvicorn
        
        # Run the API server (blocking call)
        uvicorn.run(
            "api.app:app",
            host=api_host,
            port=api_port,
            reload=False,
            log_level="info"
        )
        
    except ImportError:
        logger.warning("uvicorn not found, trying subprocess method...")
        try:
            cmd = [
                sys.executable, 
                "-m", 
                "uvicorn", 
                "api.app:app",
                "--host", api_host,
                "--port", str(api_port)
            ]
            subprocess.run(cmd, check=True, cwd=script_dir)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start API service: {e}")
            return 1
        except FileNotFoundError:
            logger.error("uvicorn not installed! Install it with: pip install uvicorn")
            return 1
    except KeyboardInterrupt:
        logger.info("\nAPI service stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Error starting API service: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(start_api_service())
