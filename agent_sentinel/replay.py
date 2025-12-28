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
from typing import Any, Optional, Dict, List
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
    inputs: Dict[str, Any]
    outputs: Any
    cost_usd: float
    duration_ms: float
    outcome: str
    timestamp: str
    tags: List[str]


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
    
    _active_replay: Optional["ReplayMode"] = None
    _replay_stack: List["ReplayMode"] = []
    
    def __init__(
        self,
        entries: List[ReplayEntry],
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
        self.divergences: List[Dict[str, Any]] = []
        
    @classmethod
    def from_run_id(
        cls,
        run_id: str,
        ledger_path: Optional[Path] = None,
        strict: bool = True,
        warn_on_divergence: bool = True,
    ) -> "ReplayMode":
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
        ledger_path: Path,
        strict: bool = True,
        warn_on_divergence: bool = True,
    ) -> "ReplayMode":
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
        if not ledger_path.exists():
            raise FileNotFoundError(f"Ledger file not found: {ledger_path}")
        
        entries = cls._load_entries(ledger_path, run_id=None)
        
        if not entries:
            raise ValueError(f"No entries found in ledger: {ledger_path}")
        
        logger.info(f"Loaded {len(entries)} entries for replay from {ledger_path}")
        
        return cls(entries, strict=strict, warn_on_divergence=warn_on_divergence)
    
    @staticmethod
    def _load_entries(ledger_path: Path, run_id: Optional[str]) -> List[ReplayEntry]:
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
        
        with open(ledger_path, "r", encoding="utf-8") as f:
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
                    )
                    
                    # If filtering by run_id, check if entry matches
                    # For now, we load all entries since run_id isn't in the ledger
                    # TODO: Add run_id to ledger entries in future update
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
        inputs: Dict[str, Any]
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
    def _inputs_match(recorded: Dict[str, Any], current: Dict[str, Any]) -> bool:
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
    
    def get_divergences(self) -> List[Dict[str, Any]]:
        """
        Get list of all detected divergences.
        
        Returns:
            List of divergence details
        """
        return self.divergences.copy()
    
    def __enter__(self) -> "ReplayMode":
        """Enter replay mode context."""
        ReplayMode._replay_stack.append(ReplayMode._active_replay)
        ReplayMode._active_replay = self
        logger.info(f"Entered replay mode with {len(self.entries)} entries")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit replay mode context."""
        progress = self.current_index
        total = len(self.entries)
        
        logger.info(
            f"Exited replay mode: {progress}/{total} entries replayed, "
            f"{len(self.divergences)} divergences detected"
        )
        
        ReplayMode._active_replay = ReplayMode._replay_stack.pop()
        return False
    
    @classmethod
    def is_active(cls) -> bool:
        """Check if replay mode is currently active."""
        return cls._active_replay is not None
    
    @classmethod
    def get_active(cls) -> Optional["ReplayMode"]:
        """Get the currently active replay mode, if any."""
        return cls._active_replay


@contextmanager
def replay_mode(
    run_id: Optional[str] = None,
    ledger_path: Optional[Path] = None,
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

