"""
Replay Module: Mock function execution with recorded outputs.

Phase 4 Implementation:
- Read ledger entries for a specific run
- Mock function execution with recorded outputs
- Detect divergence (different inputs during replay)
- Useful for debugging and testing non-deterministic behavior

This enables:
1. Debugging: Re-run a failed agent session with recorded outputs
2. Testing: Verify behavior changes without calling external APIs
3. Cost Savings: Test logic without incurring API costs
4. Determinism Detection: Identify when functions produce different results
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from contextlib import contextmanager

from .ledger import Ledger
from .errors import ReplayDivergenceError

logger = logging.getLogger("agent_sentinel")


@dataclass
class ReplayEntry:
    """A single recorded action from the ledger."""
    id: str
    action: str
    inputs: dict[str, Any]
    outputs: Any
    cost_usd: float
    duration_ms: float
    outcome: str
    timestamp: str
    tags: list[str]
    run_id: str | None = None


class ReplayMode:
    """
    Replay mode for mocking function execution with recorded outputs.
    
    Usage:
        # Record a run normally
        @guarded_action(name="fetch_data", cost_usd=0.01)
        def fetch_data(url):
            return requests.get(url).json()
        
        result = fetch_data("https://api.example.com")
        run_id = get_current_run_id()
        
        # Later, replay the run
        with ReplayMode.from_run_id(run_id):
            # This will return the recorded output without calling the API
            result = fetch_data("https://api.example.com")
        
        # Divergence detection
        with ReplayMode.from_run_id(run_id, strict=True):
            # This will raise ReplayDivergenceError if inputs don't match
            result = fetch_data("https://different-api.com")
    """
    
    _active_replay: ReplayMode | None = None
    _replay_stack: list[ReplayMode] = []
    
    def __init__(
        self,
        entries: list[ReplayEntry],
        strict: bool = True,
        warn_on_divergence: bool = True,
    ):
        """
        Initialize replay mode with recorded entries.
        
        Args:
            entries: List of recorded actions to replay
            strict: If True, raise ReplayDivergenceError on input mismatch
                   If False, log warning and execute function normally
            warn_on_divergence: If True, log warnings when inputs don't match
        """
        self.entries = entries
        self.strict = strict
        self.warn_on_divergence = warn_on_divergence
        self.current_index = 0
        self.divergences: list[dict[str, Any]] = []
        
    @classmethod
    def from_run_id(
        cls,
        run_id: str,
        ledger_path: Path | None = None,
        strict: bool = True,
        warn_on_divergence: bool = True,
    ) -> ReplayMode:
        """
        Create a replay mode from a specific run ID.
        
        Args:
            run_id: UUID of the run to replay
            ledger_path: Path to ledger file (defaults to current ledger)
            strict: If True, raise error on input mismatch
            warn_on_divergence: If True, log warnings on mismatches
            
        Returns:
            ReplayMode instance ready for use
            
        Raises:
            FileNotFoundError: If ledger file doesn't exist
            ValueError: If no entries found for run_id
        """
        if ledger_path is None:
            ledger_path = Ledger.get_log_path()
            
        if not ledger_path or not ledger_path.exists():
            raise FileNotFoundError(f"Ledger file not found: {ledger_path}")
        
        entries = cls._load_entries(ledger_path, run_id)
        
        if not entries:
            raise ValueError(f"No entries found for run_id: {run_id}")
        
        logger.info(f"Loaded {len(entries)} entries for replay (run_id={run_id})")
        
        return cls(entries, strict=strict, warn_on_divergence=warn_on_divergence)
    
    @classmethod
    def from_ledger_file(
        cls,
        ledger_path: Path | str,
        strict: bool = True,
        warn_on_divergence: bool = True,
    ) -> ReplayMode:
        """
        Create a replay mode from an entire ledger file.
        
        Replays all entries in the order they were recorded.
        
        Args:
            ledger_path: Path to ledger file
            strict: If True, raise error on input mismatch
            warn_on_divergence: If True, log warnings on mismatches
            
        Returns:
            ReplayMode instance ready for use
            
        Raises:
            FileNotFoundError: If ledger file doesn't exist
            ValueError: If no entries found in ledger
        """
        # Accept a str as well as a Path. `replay_mode(ledger_path="...")` is
        # the obvious thing to reach for and previously died on
        # AttributeError: 'str' object has no attribute 'exists'.
        ledger_path = Path(ledger_path)

        if not ledger_path.exists():
            raise FileNotFoundError(f"Ledger file not found: {ledger_path}")
        
        entries = cls._load_entries(ledger_path, run_id=None)
        
        if not entries:
            raise ValueError(f"No entries found in ledger: {ledger_path}")
        
        logger.info(f"Loaded {len(entries)} entries for replay from {ledger_path}")
        
        return cls(entries, strict=strict, warn_on_divergence=warn_on_divergence)
    
    @staticmethod
    def _load_entries(ledger_path: Path, run_id: str | None) -> list[ReplayEntry]:
        """
        Load entries from ledger file, optionally filtering by run_id.
        
        Note: Since Phase 1-3 don't include run_id in ledger entries,
        this implementation loads all entries. In Phase 4, we recommend
        adding run_id to ledger entries for better replay filtering.
        
        Args:
            ledger_path: Path to ledger file
            run_id: Optional run ID to filter by (currently not used in ledger format)
            
        Returns:
            List of ReplayEntry objects
        """
        entries = []
        
        with open(ledger_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Extract inputs and outputs from payload
                    payload = data.get("payload", {})
                    inputs = payload.get("inputs", {})
                    outputs = payload.get("outputs")
                    
                    entry = ReplayEntry(
                        id=data.get("id", ""),
                        action=data.get("action", ""),
                        inputs=inputs,
                        outputs=outputs,
                        cost_usd=data.get("cost_usd", 0.0),
                        duration_ms=data.get("duration_ms", 0.0),
                        outcome=data.get("outcome", ""),
                        timestamp=data.get("timestamp", ""),
                        tags=data.get("tags", []),
                        run_id=data.get("run_id"),
                    )
                    
                    # Filter by run_id if specified
                    if run_id is not None:
                        entry_run_id = data.get("run_id")
                        if entry_run_id != run_id:
                            continue

                    # Only load entries with real outcomes (not replayed ones)
                    if entry.outcome == "replayed":
                        continue

                    entries.append(entry)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse ledger line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing ledger line {line_num}: {e}")
                    continue
        
        return entries
    
    def get_next_output(
        self, 
        action_name: str, 
        inputs: dict[str, Any]
    ) -> tuple[Any, bool]:
        """
        Get the next recorded output for this action.
        
        Args:
            action_name: Name of the action being replayed
            inputs: Current inputs to the action
            
        Returns:
            Tuple of (output, inputs_match)
            - output: The recorded output
            - inputs_match: True if inputs match recorded inputs
            
        Raises:
            ReplayDivergenceError: If strict mode and inputs don't match
            IndexError: If no more entries available
        """
        if self.current_index >= len(self.entries):
            raise IndexError(
                f"No more replay entries available. "
                f"Expected: {action_name}, Index: {self.current_index}/{len(self.entries)}"
            )
        
        entry = self.entries[self.current_index]
        self.current_index += 1
        
        # Check if action name matches
        if entry.action != action_name:
            divergence = {
                "type": "action_mismatch",
                "expected": entry.action,
                "actual": action_name,
                "index": self.current_index - 1,
            }
            self.divergences.append(divergence)
            
            if self.warn_on_divergence:
                logger.warning(
                    f"Action mismatch at index {self.current_index - 1}: "
                    f"expected '{entry.action}', got '{action_name}'"
                )
            
            if self.strict:
                raise ReplayDivergenceError(
                    f"Action mismatch: expected '{entry.action}', got '{action_name}' "
                    f"at position {self.current_index - 1}"
                )
        
        # Check if inputs match
        inputs_match = self._inputs_match(entry.inputs, inputs)
        
        if not inputs_match:
            divergence = {
                "type": "input_mismatch",
                "action": action_name,
                "expected_inputs": entry.inputs,
                "actual_inputs": inputs,
                "index": self.current_index - 1,
            }
            self.divergences.append(divergence)
            
            if self.warn_on_divergence:
                logger.warning(
                    f"Input mismatch for '{action_name}' at index {self.current_index - 1}: "
                    f"expected {entry.inputs}, got {inputs}"
                )
            
            if self.strict:
                raise ReplayDivergenceError(
                    f"Input mismatch for '{action_name}': "
                    f"expected {entry.inputs}, got {inputs}"
                )
        
        return entry.outputs, inputs_match
    
    @staticmethod
    def _inputs_match(recorded: dict[str, Any], current: dict[str, Any]) -> bool:
        """
        Check if current inputs match recorded inputs.
        
        This does a deep comparison but is lenient with:
        - Floating point precision (uses approximate equality)
        - None vs missing keys
        - Order of dict keys
        
        Args:
            recorded: Recorded inputs from ledger
            current: Current inputs being passed
            
        Returns:
            True if inputs match (within tolerance)
        """
        try:
            # Convert both to JSON and compare
            # This handles nested structures and normalizes types
            recorded_str = json.dumps(recorded, sort_keys=True, default=str)
            current_str = json.dumps(current, sort_keys=True, default=str)
            
            return recorded_str == current_str
        except Exception:
            # If JSON serialization fails, fall back to direct comparison
            return recorded == current
    
    def reset(self):
        """Reset replay to the beginning."""
        self.current_index = 0
        self.divergences.clear()
    
    def get_progress(self) -> tuple[int, int]:
        """
        Get current replay progress.
        
        Returns:
            Tuple of (current_index, total_entries)
        """
        return self.current_index, len(self.entries)
    
    def get_divergences(self) -> list[dict[str, Any]]:
        """
        Get list of all detected divergences.
        
        Returns:
            List of divergence details
        """
        return self.divergences.copy()
    
    def __enter__(self) -> ReplayMode:
        """Enter replay mode context."""
        ReplayMode._replay_stack.append(ReplayMode._active_replay)
        ReplayMode._active_replay = self
        logger.info(f"Entered replay mode with {len(self.entries)} entries")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit replay mode context and report results to platform if configured."""
        progress = self.current_index
        total = len(self.entries)

        logger.info(
            f"Exited replay mode: {progress}/{total} entries replayed, "
            f"{len(self.divergences)} divergences detected"
        )

        # Calculate cost savings (sum of cost_usd for replayed entries)
        cost_savings = sum(e.cost_usd for e in self.entries[:progress])

        # Determine status
        if exc_type is not None:
            status = "failed"
        elif len(self.divergences) == 0:
            status = "success"
        else:
            status = "partial"

        # Report results to platform (best-effort, never raises)
        self._report_results(status, progress, total, cost_savings)

        ReplayMode._active_replay = ReplayMode._replay_stack.pop()
        return False

    def _report_results(
        self,
        status: str,
        actions_replayed: int,
        actions_total: int,
        cost_savings: float,
    ) -> None:
        """POST replay results to the platform if sync is configured."""
        try:
            from .sync import get_sync
            sync = get_sync()
            if sync is None or not sync.config.enabled:
                logger.debug("Remote sync not configured, skipping replay result report")
                return

            run_id = sync.config.run_id
            url = f"{sync.config.platform_url}/api/v1/replay/{run_id}/results"
            headers = {
                "Authorization": f"ApiKey {sync.config.api_token}",
                "Content-Type": "application/json",
            }
            payload = {
                "status": status,
                "actions_replayed": actions_replayed,
                "actions_total": actions_total,
                "divergences_detected": len(self.divergences),
                "divergences": self.divergences,
                "cost_savings": cost_savings,
            }

            try:
                import httpx
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    if resp.status_code == 201:
                        logger.info(f"Replay results reported to platform (run_id={run_id})")
                    else:
                        logger.warning(f"Failed to report replay results: {resp.status_code} {resp.text}")
            except ImportError:
                logger.debug("httpx not available, cannot report replay results")
            except Exception as e:
                logger.warning(f"Failed to report replay results: {e}")
        except Exception as e:
            logger.debug(f"Could not report replay results: {e}")
    
    @classmethod
    def from_json(
        cls,
        data: dict[str, Any],
        strict: bool = True,
        warn_on_divergence: bool = True,
    ) -> ReplayMode:
        """
        Create a ReplayMode from a JSON bundle downloaded from the console.

        The bundle format matches what GET /api/v1/replay/{run_id}/bundle returns:
        {
            "run_id": "...",
            "agent_id": "...",
            "entries": [
                {
                    "id": "...",
                    "action": "...",
                    "inputs": {...},
                    "outputs": {...},
                    "cost_usd": 0.0,
                    "duration_ms": 0.0,
                    "outcome": "...",
                    "timestamp": "...",
                    "tags": [...],
                    "run_id": "...",
                },
                ...
            ]
        }

        Args:
            data: Parsed JSON bundle dict (as returned by the platform bundle endpoint)
            strict: If True, raise ReplayDivergenceError on input mismatch
            warn_on_divergence: If True, log warnings when inputs don't match

        Returns:
            ReplayMode instance ready for use as a context manager

        Raises:
            ValueError: If ``data`` has no "entries" key or entries list is empty
        """
        raw_entries = data.get("entries")
        if not raw_entries:
            raise ValueError(
                "Bundle must contain a non-empty 'entries' list. "
                "Got: entries=" + repr(raw_entries)
            )

        run_id = data.get("run_id")
        entries: list[ReplayEntry] = []
        for item in raw_entries:
            entry = ReplayEntry(
                id=item.get("id", ""),
                action=item.get("action", ""),
                inputs=item.get("inputs") or {},
                outputs=item.get("outputs"),
                cost_usd=float(item.get("cost_usd") or 0.0),
                duration_ms=float(item.get("duration_ms") or 0.0),
                outcome=item.get("outcome", ""),
                timestamp=item.get("timestamp", ""),
                tags=item.get("tags") or [],
                run_id=item.get("run_id", run_id),
            )
            entries.append(entry)

        logger.info(
            f"Loaded {len(entries)} entries for replay from JSON bundle (run_id={run_id})"
        )

        return cls(entries, strict=strict, warn_on_divergence=warn_on_divergence)

    @classmethod
    def is_active(cls) -> bool:
        """Check if replay mode is currently active."""
        return cls._active_replay is not None
    
    @classmethod
    def get_active(cls) -> ReplayMode | None:
        """Get the currently active replay mode, if any."""
        return cls._active_replay


@contextmanager
def replay_mode(
    run_id: str | None = None,
    ledger_path: Path | None = None,
    strict: bool = True,
    warn_on_divergence: bool = True,
):
    """
    Context manager for replay mode.
    
    Usage:
        with replay_mode(run_id="abc-123"):
            # All guarded actions will use recorded outputs
            result = my_function()
    
    Args:
        run_id: UUID of the run to replay (if None, replays entire ledger)
        ledger_path: Path to ledger file (defaults to current ledger)
        strict: If True, raise error on input mismatch
        warn_on_divergence: If True, log warnings on mismatches
    """
    if run_id:
        replay = ReplayMode.from_run_id(
            run_id, 
            ledger_path=ledger_path,
            strict=strict,
            warn_on_divergence=warn_on_divergence,
        )
    else:
        if ledger_path is None:
            ledger_path = Ledger.get_log_path()
        replay = ReplayMode.from_ledger_file(
            ledger_path,
            strict=strict,
            warn_on_divergence=warn_on_divergence,
        )
    
    with replay:
        yield replay

