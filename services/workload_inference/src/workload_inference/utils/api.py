import logging

import requests

from workload_inference.data.data_structures import ExperimentStatus

API_TIMEOUT = 0.1  # seconds

# Suppress anoying http request deub messages
logging.getLogger("urllib3").setLevel(logging.WARNING)


class ExperimentAPIError(Exception):
    """Custom exception for errors related to the ExperimentAPI."""

    pass


class ExperimentAPI:
    def __init__(self, endpoint: str = "http://localhost", port: int = 8080):
        self.endpoint = endpoint
        self.port = port

    def get_experiment_state(self) -> ExperimentStatus:
        """Fetches the current state of the experiment from the API.

        Returns:
            ExperimentStatus: The current status (including the states) of the experiment.
        Raises:
            ExperimentAPIError: If there was an error making the HTTP request.
        """
        try:
            response = requests.get(
                f"{self.endpoint}:{self.port}/api/state", timeout=API_TIMEOUT
            )
            response.raise_for_status()
            return ExperimentStatus.from_dict(response.json())
        except Exception as e:
            raise ExperimentAPIError(f"Error fetching experiment state: {e}") from e

    def trigger_next_state(self) -> None:
        """Sends a request to the API to move to the next state in the experiment if possible.

        Raises:
            ExperimentAPIError: If there was an error making the HTTP request.
        """
        try:
            response = requests.get(
                f"{self.endpoint}:{self.port}/api/operatorclicked", timeout=API_TIMEOUT
            )
            response.raise_for_status()
        except Exception as e:
            raise ExperimentAPIError(f"Error triggering next state: {e}") from e
