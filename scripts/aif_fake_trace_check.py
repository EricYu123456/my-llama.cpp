#!/usr/bin/env python3
import argparse
import math
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

POST_RE = re.compile(
    r"^POST rows=(\d+) cols=(\d+) nbytes=(\d+) start_lba=(\d+) nblocks=(\d+) host_addr=(\S+)"
)
GEMV_RE = re.compile(
    r"^GEMV input_dim=(\d+) output_dim=(\d+) matrix_start_lba=(\d+) matrix_nblocks=(\d+) input_addr=(\S+) output_addr=(\S+)"
)


@dataclass(frozen=True)
class PostRecord:
    line_no: int
    rows: int
    cols: int
    nbytes: int
    start_lba: int
    nblocks: int
    host_addr: str

    @property
    def end_lba(self) -> int:
        return self.start_lba + self.nblocks


@dataclass(frozen=True)
class GemvRecord:
    line_no: int
    input_dim: int
    output_dim: int
    start_lba: int
    nblocks: int
    input_addr: str
    output_addr: str


def is_null_ptr(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in {"0", "0x0", "(nil)", "null", "nullptr"}


def parse_trace(path: str) -> tuple[List[PostRecord], List[GemvRecord], List[str]]:
    posts: List[PostRecord] = []
    gemvs: List[GemvRecord] = []
    warnings: List[str] = []

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for idx, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            post_match = POST_RE.match(line)
            if post_match:
                rows, cols, nbytes, start_lba, nblocks, host_addr = post_match.groups()
                posts.append(
                    PostRecord(
                        line_no=idx,
                        rows=int(rows),
                        cols=int(cols),
                        nbytes=int(nbytes),
                        start_lba=int(start_lba),
                        nblocks=int(nblocks),
                        host_addr=host_addr,
                    )
                )
                continue

            gemv_match = GEMV_RE.match(line)
            if gemv_match:
                input_dim, output_dim, start_lba, nblocks, input_addr, output_addr = gemv_match.groups()
                gemvs.append(
                    GemvRecord(
                        line_no=idx,
                        input_dim=int(input_dim),
                        output_dim=int(output_dim),
                        start_lba=int(start_lba),
                        nblocks=int(nblocks),
                        input_addr=input_addr,
                        output_addr=output_addr,
                    )
                )
                continue

            warnings.append(f"Line {idx}: unrecognized trace entry: {line}")

    return posts, gemvs, warnings


def find_post_for_lba(posts: List[PostRecord], start_lba: int) -> Optional[PostRecord]:
    for post in posts:
        if post.start_lba <= start_lba < post.end_lba:
            return post
    return None


def validate_trace(
    posts: List[PostRecord],
    gemvs: List[GemvRecord],
    block_size: int,
    allow_gaps: bool,
) -> tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not posts:
        warnings.append("No POST entries found.")
    if not gemvs:
        warnings.append("No GEMV entries found.")

    posts_by_start: Dict[int, PostRecord] = {}
    last_end: Optional[int] = None

    for post in posts:
        if post.rows <= 0 or post.cols <= 0:
            errors.append(f"Line {post.line_no}: POST rows/cols must be positive.")
        if post.nbytes <= 0:
            errors.append(f"Line {post.line_no}: POST nbytes must be positive.")
        if post.nblocks <= 0:
            errors.append(f"Line {post.line_no}: POST nblocks must be positive.")
        if is_null_ptr(post.host_addr):
            errors.append(f"Line {post.line_no}: POST host_addr is null.")

        expected_blocks = (post.nbytes + block_size - 1) // block_size
        if expected_blocks != post.nblocks:
            errors.append(
                f"Line {post.line_no}: POST nblocks mismatch (expected {expected_blocks}, got {post.nblocks})."
            )

        if post.start_lba in posts_by_start:
            errors.append(
                f"Line {post.line_no}: POST start_lba {post.start_lba} duplicates line {posts_by_start[post.start_lba].line_no}."
            )
        posts_by_start[post.start_lba] = post

        if last_end is not None:
            if post.start_lba < last_end:
                errors.append(
                    f"Line {post.line_no}: POST start_lba {post.start_lba} overlaps previous range ending at {last_end}."
                )
            elif post.start_lba > last_end and not allow_gaps:
                errors.append(
                    f"Line {post.line_no}: POST start_lba {post.start_lba} is not contiguous with previous end {last_end}."
                )
            elif post.start_lba > last_end:
                warnings.append(
                    f"Line {post.line_no}: POST gap detected (prev end {last_end}, next start {post.start_lba})."
                )
        last_end = post.end_lba

    posts_sorted = sorted(posts, key=lambda p: p.start_lba)
    gemv_hits: Dict[int, int] = {}

    for gemv in gemvs:
        if gemv.input_dim <= 0 or gemv.output_dim <= 0:
            errors.append(f"Line {gemv.line_no}: GEMV input/output dims must be positive.")
        if gemv.nblocks <= 0:
            errors.append(f"Line {gemv.line_no}: GEMV matrix_nblocks must be positive.")
        if is_null_ptr(gemv.input_addr):
            errors.append(f"Line {gemv.line_no}: GEMV input_addr is null.")
        if is_null_ptr(gemv.output_addr):
            errors.append(f"Line {gemv.line_no}: GEMV output_addr is null.")

        post = find_post_for_lba(posts_sorted, gemv.start_lba)
        if not post:
            errors.append(
                f"Line {gemv.line_no}: GEMV matrix_start_lba {gemv.start_lba} does not match any POST range."
            )
            continue

        gemv_hits[post.start_lba] = gemv_hits.get(post.start_lba, 0) + 1

        if gemv.start_lba + gemv.nblocks > post.end_lba:
            errors.append(
                f"Line {gemv.line_no}: GEMV range exceeds POST range (POST end {post.end_lba})."
            )

        if gemv.input_dim != post.cols:
            errors.append(
                f"Line {gemv.line_no}: GEMV input_dim {gemv.input_dim} does not match POST cols {post.cols}."
            )

        if gemv.output_dim > post.rows:
            errors.append(
                f"Line {gemv.line_no}: GEMV output_dim {gemv.output_dim} exceeds POST rows {post.rows}."
            )
        elif gemv.output_dim != post.rows:
            warnings.append(
                f"Line {gemv.line_no}: GEMV output_dim {gemv.output_dim} differs from POST rows {post.rows} (likely submatrix view)."
            )

        if gemv.start_lba == post.start_lba and gemv.output_dim == post.rows:
            if gemv.nblocks != post.nblocks:
                errors.append(
                    f"Line {gemv.line_no}: GEMV matrix_nblocks {gemv.nblocks} does not match POST nblocks {post.nblocks}."
                )

    for post in posts_sorted:
        if post.start_lba not in gemv_hits:
            warnings.append(
                f"Line {post.line_no}: POST range starting at {post.start_lba} is never referenced by GEMV."
            )

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate GGML AIF fake ioctl trace against AiFSSD expectations."
    )
    parser.add_argument(
        "--trace",
        default="/tmp/aif_fake.trace",
        help="Path to trace file (default: /tmp/aif_fake.trace)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=4096,
        help="AIF block size in bytes (default: 4096)",
    )
    parser.add_argument(
        "--allow-gaps",
        action="store_true",
        help="Allow gaps between consecutive POST LBA ranges",
    )

    args = parser.parse_args()

    try:
        posts, gemvs, parse_warnings = parse_trace(args.trace)
    except FileNotFoundError:
        print(f"Trace file not found: {args.trace}")
        return 2

    errors, warnings = validate_trace(posts, gemvs, args.block_size, args.allow_gaps)
    warnings = parse_warnings + warnings

    print(f"Trace summary: {len(posts)} POST, {len(gemvs)} GEMV")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nOK: trace passes logical checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
