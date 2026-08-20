"""Client for the NESO Carbon Intensity API.

The API is documented at https://carbon-intensity.github.io/api-definitions/ and
requires no authentication. All datetimes it accepts and returns are UTC.

Data licensed CC BY 4.0 by the National Energy System Operator.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field

import requests

BASE_URL = "https://api.carbonintensity.org.uk"

#: The /intensity/{from}/{to} endpoint refuses ranges longer than this.
MAX_RANGE = dt.timedelta(days=14)

#: Statuses worth retrying: transient server faults and rate limiting.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class CarbonIntensityError(RuntimeError):
    """Raised when the API returns an error the client will not retry."""


def format_timestamp(moment: dt.datetime) -> str:
    """Render a datetime in the ``YYYY-MM-DDThh:mmZ`` form the API expects.

    Naive datetimes are assumed to be UTC; aware ones are converted.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.UTC)
    return moment.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%MZ")


def parse_timestamp(text: str) -> dt.datetime:
    """Parse an API timestamp into an aware UTC datetime."""
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%MZ").replace(tzinfo=dt.UTC)


def window_range(
    start: dt.datetime, end: dt.datetime, step: dt.timedelta = MAX_RANGE
) -> list[tuple[dt.datetime, dt.datetime]]:
    """Split ``[start, end]`` into consecutive windows no longer than ``step``.

    Used for backfill, because the range endpoint caps a single request at
    fourteen days.
    """
    if end <= start:
        raise ValueError("end must be after start")
    windows: list[tuple[dt.datetime, dt.datetime]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + step, end)
        windows.append((cursor, stop))
        cursor = stop
    return windows


@dataclass
class CarbonIntensityClient:
    """Thin wrapper over the API with retries and a shared session."""

    base_url: str = BASE_URL
    timeout: float = 20.0
    max_attempts: int = 4
    backoff: float = 1.5
    session: requests.Session = field(default_factory=requests.Session)

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        last_error: str | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.session.get(
                    url, headers={"Accept": "application/json"}, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    payload = response.json()
                    if "data" not in payload:
                        raise CarbonIntensityError(f"{url}: response contained no data field")
                    return payload
                if response.status_code not in RETRY_STATUSES:
                    raise CarbonIntensityError(f"{url}: HTTP {response.status_code}")
                last_error = f"HTTP {response.status_code}"

            if attempt < self.max_attempts:
                time.sleep(self.backoff ** (attempt - 1))

        raise CarbonIntensityError(
            f"{url}: giving up after {self.max_attempts} attempts ({last_error})"
        )

    # -- national intensity -------------------------------------------------

    def current_intensity(self) -> dict:
        """Intensity for the settlement period in progress."""
        return self._get("/intensity")

    def forecast_48h(self, start: dt.datetime) -> dict:
        """The forecast issued for the 48 hours following ``start``.

        This is the only way to record a forecast at its issue time: the API
        overwrites forecasts as they are revised and serves no history of them.
        """
        return self._get(f"/intensity/{format_timestamp(start)}/fw48h")

    def intensity_range(self, start: dt.datetime, end: dt.datetime) -> dict:
        """Intensity between two datetimes, at most fourteen days apart."""
        if end - start > MAX_RANGE:
            raise ValueError(f"range exceeds the API limit of {MAX_RANGE.days} days")
        return self._get(f"/intensity/{format_timestamp(start)}/{format_timestamp(end)}")

    # -- generation mix -----------------------------------------------------

    def current_generation(self) -> dict:
        """Generation mix for the settlement period in progress."""
        return self._get("/generation")

    def generation_range(self, start: dt.datetime, end: dt.datetime) -> dict:
        """Generation mix between two datetimes."""
        return self._get(f"/generation/{format_timestamp(start)}/{format_timestamp(end)}")
