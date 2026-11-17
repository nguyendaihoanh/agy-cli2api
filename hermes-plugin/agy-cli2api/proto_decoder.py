"""Minimal zero-dependency protobuf wire-format decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List


@dataclass(frozen=True)
class ProtoField:
    number: int
    wire_type: int
    value: Any


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, offset
        shift += 7
        if shift > 70:
            raise ValueError("varint exceeded 10 bytes")
    raise ValueError("truncated varint")


def decode_message(data: bytes, max_depth: int = 16) -> List[ProtoField]:
    """Decode one protobuf payload without requiring .proto schemas."""
    if max_depth <= 0:
        return []

    fields: List[ProtoField] = []
    offset = 0
    size = len(data)

    while offset < size:
        try:
            tag, offset = read_varint(data, offset)
        except ValueError:
            return fields

        field_number = tag >> 3
        wire_type = tag & 0x07
        if field_number == 0:
            break

        if wire_type == 0:
            try:
                value, offset = read_varint(data, offset)
            except ValueError:
                break
            fields.append(ProtoField(field_number, wire_type, value))
        elif wire_type == 1:
            if offset + 8 > size:
                break
            fields.append(ProtoField(field_number, wire_type, data[offset : offset + 8]))
            offset += 8
        elif wire_type == 2:
            try:
                length, offset = read_varint(data, offset)
            except ValueError:
                break
            if length < 0 or offset + length > size:
                break
            raw = data[offset : offset + length]
            offset += length
            nested = _try_decode_nested(raw, max_depth - 1)
            fields.append(ProtoField(field_number, wire_type, nested if nested is not None else raw))
        elif wire_type == 5:
            if offset + 4 > size:
                break
            fields.append(ProtoField(field_number, wire_type, data[offset : offset + 4]))
            offset += 4
        else:
            break

    return fields


def _try_decode_nested(raw: bytes, max_depth: int) -> Any:
    if not raw or max_depth <= 0 or _looks_like_text(raw):
        return None
    fields = decode_message(raw, max_depth=max_depth)
    if not fields:
        return None
    return fields


def _looks_like_text(raw: bytes) -> bool:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if not text:
        return True
    bad = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t")
    return bad / len(text) < 0.05


def find_fields(fields: List[ProtoField], number: int) -> List[ProtoField]:
    return [field for field in fields if field.number == number]


def raw_text(fields: List[ProtoField], number: int) -> str:
    parts: List[str] = []
    for field in find_fields(fields, number):
        if isinstance(field.value, bytes):
            try:
                text = field.value.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if text:
                parts.append(text)
    return "\n".join(parts)


def walk_text(fields: List[ProtoField]) -> Iterator[str]:
    for field in fields:
        if isinstance(field.value, list):
            yield from walk_text(field.value)
        elif isinstance(field.value, bytes):
            try:
                text = field.value.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if text.strip():
                yield text
