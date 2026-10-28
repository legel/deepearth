"""Bounded retries for the soil services, standard library only (the sourcing client's regime).

Soil Data Access and the POLARIS tile server are public services with no SLA. A request is
retried on a 5xx, a 429, a timeout or a dropped connection, with exponential backoff and
jitter, about five attempts over about five minutes; any other 4xx is our request's fault and
fails at once. What finally fails names the service (`OutageError`), in the shape
`sims.fetch.ExternalServiceError` records, so a sims job can tell an outage from a bug.
"""
from __future__ import annotations

import random
import socket
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

RETRY_STATUS = (429, 500, 502, 503, 504)
ATTEMPTS, BASE_S, CAP_S = 5, 15.0, 120.0


class OutageError(RuntimeError):
    """A soil service kept failing through every retry."""

    is_outage = True

    def __init__(self, service: str, url: str, detail: str, attempts: int):
        self.service, self.url, self.detail, self.attempts = service, url, detail, attempts
        super().__init__(f'external service outage: {service} ({url}) failed {attempts} attempts; last: {detail}')

    def record(self) -> dict:
        return {'service': self.service, 'url': self.url, 'detail': self.detail, 'attempts': self.attempts}


def backoff(k: int, base_s: float = BASE_S, cap_s: float = CAP_S) -> float:
    """Seconds to wait after failed attempt `k` (0-based), with jitter."""
    return min(cap_s, base_s * 2 ** k) * (0.5 + 0.5 * random.random())


def urlopen(req: urllib.request.Request, service: str, timeout: float = 120, attempts: int = ATTEMPTS,
            sleep=time.sleep) -> bytes:
    """The body of one request, with bounded retries.

    Args:
        req: The request.
        service: The service's name for the final error.
        timeout: Per attempt [s].
        attempts: Requests at most.
        sleep: Clock, injectable for tests.
    Returns:
        The response body.
    Raises:
        OutageError: every attempt failed transiently.
        urllib.error.HTTPError: a 4xx other than 429.
    """
    last = ''
    for k in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS:
                raise
            last = 'HTTP %d' % e.code
        except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as e:
            last = '%s: %s' % (type(e).__name__, str(e)[:200])
        if k + 1 < attempts:
            sleep(backoff(k))
    p = urlsplit(req.full_url)
    raise OutageError(service, '%s://%s%s' % (p.scheme, p.netloc, p.path), last, attempts)
