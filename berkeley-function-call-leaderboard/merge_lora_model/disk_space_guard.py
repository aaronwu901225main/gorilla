import os
import shutil
from typing import Optional, Tuple, Union


EXTRA_REQUIRED_BYTES = 10 * 1024 ** 3


def format_bytes(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    value = float(num_bytes)

    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{num_bytes:,} bytes"
            return f"{value:.2f} {unit} ({num_bytes:,} bytes)"
        value /= 1024.0

    return f"{num_bytes:,} bytes"


def get_directory_size_bytes(directory: Union[str, os.PathLike]) -> int:
    if not os.path.isdir(directory):
        raise ValueError(
            "Base model path must be a local directory for exact disk-space "
            f"checks: {directory}"
        )

    total_size = 0
    for dirpath, _, filenames in os.walk(directory, followlinks=False):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(file_path)
            except OSError as exc:
                raise OSError(f"Unable to read file size: {file_path}") from exc

    return total_size


def _existing_disk_usage_path(path: Union[str, os.PathLike]) -> str:
    candidate = os.path.abspath(os.fspath(path))
    if os.path.exists(candidate):
        return candidate

    parent = os.path.dirname(candidate)
    while parent and not os.path.exists(parent):
        next_parent = os.path.dirname(parent)
        if next_parent == parent:
            break
        parent = next_parent

    return parent or "."


def require_merge_output_space(
    output_path: Union[str, os.PathLike],
    base_model_size_bytes: int,
    checkpoint_name: Optional[str] = None,
    extra_required_bytes: int = EXTRA_REQUIRED_BYTES,
) -> Tuple[int, int]:
    required_bytes = base_model_size_bytes + extra_required_bytes
    disk_usage_path = _existing_disk_usage_path(output_path)
    available_bytes = shutil.disk_usage(disk_usage_path).free

    if available_bytes < required_bytes:
        missing_bytes = required_bytes - available_bytes
        lines = ["Insufficient disk space before merge."]
        if checkpoint_name:
            lines.append(f"  Checkpoint: {checkpoint_name}")
        lines.extend(
            [
                f"  Output path: {output_path}",
                f"  Base model size: {format_bytes(base_model_size_bytes)}",
                f"  Required free space: {format_bytes(required_bytes)}",
                f"  Available free space: {format_bytes(available_bytes)}",
                f"  Missing space: {format_bytes(missing_bytes)}",
            ]
        )
        raise RuntimeError("\n".join(lines))

    return available_bytes, required_bytes
