#!/usr/bin/env python3
"""
Health Check Script for ML4T Inference System.

Simple health check script for Docker container monitoring.
Validates that the inference API is responding and healthy.
"""

import sys
import time
import requests
from typing import Dict, Any


def check_health(endpoint: str = "http://localhost:8090/health", timeout: int = 10) -> Dict[str, Any]:
    """
    Check health status of the ML4T inference system.
    
    Args:
        endpoint: Health check endpoint URL
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with health check results
    """
    try:
        response = requests.get(endpoint, timeout=timeout)
        
        if response.status_code == 200:
            health_data = response.json()
            
            # Check critical health indicators
            is_healthy = (
                health_data.get('status') == 'healthy' and
                health_data.get('inference_ready', False) and
                health_data.get('model_loaded', False) and
                health_data.get('health_score', 0) >= 50.0
            )
            
            return {
                'success': True,
                'healthy': is_healthy,
                'status': health_data.get('status'),
                'health_score': health_data.get('health_score'),
                'uptime_seconds': health_data.get('uptime_seconds'),
                'response_time_ms': response.elapsed.total_seconds() * 1000
            }
        else:
            return {
                'success': False,
                'healthy': False,
                'error': f"HTTP {response.status_code}: {response.text}",
                'response_time_ms': response.elapsed.total_seconds() * 1000
            }
            
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'healthy': False,
            'error': f"Connection failed to {endpoint}"
        }
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'healthy': False,
            'error': f"Request timeout after {timeout} seconds"
        }
    except Exception as e:
        return {
            'success': False,
            'healthy': False,
            'error': f"Health check failed: {str(e)}"
        }


def main():
    """Main health check execution."""
    # Try the standard health endpoint first
    health_result = check_health("http://localhost:8090/health", timeout=5)
    
    if not health_result['success']:
        # Fallback to main application port
        health_result = check_health("http://localhost:8080/health", timeout=5)
    
    # Print results for logging
    print(f"Health check result: {health_result}")
    
    # Return appropriate exit code
    if health_result['healthy']:
        print("✓ System is healthy")
        sys.exit(0)
    else:
        print(f"✗ System unhealthy: {health_result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()