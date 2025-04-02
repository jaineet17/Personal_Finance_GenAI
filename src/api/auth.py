from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
import os
import time
import logging
from typing import Dict, Optional
import secrets
import hashlib
from datetime import datetime, timedelta
import json
import pathlib

# Configure logging
logger = logging.getLogger("finance-llm-auth")

# API key header definition
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Fixed development API key - also hardcoded in frontend
DEV_API_KEY = 'a35014c8d44eef77bc8784eb7e27a8a8'

# Path to store API keys
API_KEYS_FILE = os.path.join('data', 'api_keys.json')

# Simple in-memory token store (would be replaced with proper database in production)
class TokenStore:
    def __init__(self):
        self.tokens = {}
        self.api_keys = {}
        self.request_counts = {}
        self.rate_limits = {}

        # Load API keys from file or create a default one for development
        self._load_api_keys()

    def _load_api_keys(self):
        # Ensure directory exists
        os.makedirs(os.path.dirname(API_KEYS_FILE), exist_ok=True)

        # Try to load existing keys first
        if os.path.exists(API_KEYS_FILE):
            try:
                with open(API_KEYS_FILE, 'r') as f:
                    stored_keys = json.load(f)
                    self.api_keys = stored_keys.get('keys', {})
                    logger.info(f"Loaded {len(self.api_keys)} API keys from file")
            except Exception as e:
                logger.error(f"Error loading API keys from file: {e}")
                self.api_keys = {}

        # Always ensure the development key exists
        dev_key_hash = hashlib.sha256(DEV_API_KEY.encode()).hexdigest()
        if dev_key_hash not in self.api_keys:
            self.api_keys[dev_key_hash] = {
                "user_id": "default",
                "created_at": time.time(),
                "name": "Default API Key"
            }

        # If no keys exist, create a default one
        if len(self.api_keys) <= 1:
            # For additional keys beyond the dev key
            default_api_key = os.environ.get("DEFAULT_API_KEY", secrets.token_hex(16))
            key_hash = hashlib.sha256(default_api_key.encode()).hexdigest()
            self.api_keys[key_hash] = {
                "user_id": "default",
                "created_at": time.time(),
                "name": "Generated API Key"
            }

        # Save keys to file
        self._save_api_keys()

        # Set up a default rate limit
        self.rate_limits["default"] = {
            "requests_per_minute": int(os.environ.get("DEFAULT_RATE_LIMIT", "60")),
            "burst": int(os.environ.get("DEFAULT_BURST_LIMIT", "10"))
        }

        logger.info(f"API key store initialized with {len(self.api_keys)} keys")

        # For development, print available API keys
        if os.environ.get("ENVIRONMENT", "development") == "development":
            # Log the development API key for convenience
            logger.info(f"Default API key for development: {DEV_API_KEY}")
            # Log other available keys
            for key_hash in self.api_keys:
                if key_hash != dev_key_hash:
                    logger.info(f"Additional API key info: {self.api_keys[key_hash]}")

    def _save_api_keys(self):
        """Save API keys to persistent storage"""
        try:
            with open(API_KEYS_FILE, 'w') as f:
                json.dump({"keys": self.api_keys}, f, indent=2)
            logger.info(f"Saved {len(self.api_keys)} API keys to file")
        except Exception as e:
            logger.error(f"Error saving API keys to file: {e}")

    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key"""
        if not api_key:
            return False

        # Always allow the development key
        if api_key == DEV_API_KEY:
            return True

        # Hash the key for comparison
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Check if key exists
        return key_hash in self.api_keys

    def get_user_for_api_key(self, api_key: str) -> Optional[str]:
        """Get the user ID associated with an API key"""
        if not api_key:
            return None

        # Special handling for development key
        if api_key == DEV_API_KEY:
            return "default"

        # Hash the key for comparison
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Check if key exists and return user_id
        if key_hash in self.api_keys:
            return self.api_keys[key_hash]["user_id"]

        return None

    def check_rate_limit(self, api_key: str) -> Dict:
        """Check if request exceeds rate limit, and update counts"""
        user_id = self.get_user_for_api_key(api_key)

        if not user_id:
            return {"allowed": False, "reason": "Invalid API key"}

        # Get current time
        current_time = time.time()

        # Initialize request count for this user if not exists
        if user_id not in self.request_counts:
            self.request_counts[user_id] = []

        # Clean up old requests (older than 1 minute)
        self.request_counts[user_id] = [
            req_time for req_time in self.request_counts[user_id] 
            if current_time - req_time < 60
        ]

        # Get rate limit for this user (or default)
        rate_limit = self.rate_limits.get(
            user_id, 
            self.rate_limits.get("default", {"requests_per_minute": 60, "burst": 10})
        )

        # Check if exceeding rate limit
        if len(self.request_counts[user_id]) >= rate_limit["requests_per_minute"]:
            return {
                "allowed": False, 
                "reason": "Rate limit exceeded",
                "limit": rate_limit["requests_per_minute"],
                "reset_in": 60 - (current_time - self.request_counts[user_id][0])
            }

        # Add current request to the count
        self.request_counts[user_id].append(current_time)

        return {
            "allowed": True,
            "limit": rate_limit["requests_per_minute"],
            "remaining": rate_limit["requests_per_minute"] - len(self.request_counts[user_id]),
            "reset_in": 60 - (current_time - self.request_counts[user_id][0]) if self.request_counts[user_id] else 60
        }

# Create a singleton instance of the token store
token_store = TokenStore()

async def get_api_key(api_key_header: str = Depends(API_KEY_HEADER)) -> str:
    """Dependency to get and validate API key"""
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not token_store.validate_api_key(api_key_header):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Check rate limit
    rate_limit_check = token_store.check_rate_limit(api_key_header)
    if not rate_limit_check["allowed"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=rate_limit_check["reason"],
            headers={
                "X-Rate-Limit-Limit": str(rate_limit_check.get("limit", 0)),
                "X-Rate-Limit-Reset": str(int(rate_limit_check.get("reset_in", 60))),
            },
        )

    return api_key_header

async def get_current_user(api_key: str = Depends(get_api_key)) -> str:
    """Dependency to get the current user from an API key"""
    user_id = token_store.get_user_for_api_key(api_key)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return user_id 