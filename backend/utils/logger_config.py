import logging
import os
from datetime import datetime

def setup_logger():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler for all logs
            logging.FileHandler(
                f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'
            ),
            # Stream handler for console output
            logging.StreamHandler()
        ]
    )
    
    # Suppress watchfiles logging
    logging.getLogger('watchfiles').setLevel(logging.WARNING)

    # Create logger instance
    logger = logging.getLogger('RAGAgent')
    
    # Set logging levels for different types of messages
    logger.setLevel(logging.INFO)
    
    return logger

# Create a global logger instance
# logger = setup_logger()
