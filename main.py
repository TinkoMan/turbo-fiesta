"""
main.py — Entry point for Bhajan Video Pipeline REST API.
Matches the docker-rest-user-service template structure (main.go).
"""

import os
import logging
from app import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("bhajan-pipeline")

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info("Bhajan Video Pipeline API starting on port %d", port)
    app.run(host="0.0.0.0", port=port)
