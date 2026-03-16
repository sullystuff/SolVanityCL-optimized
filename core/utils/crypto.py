import json
import logging
from pathlib import Path
from typing import Set

from base58 import b58encode
from nacl.signing import SigningKey

# In-memory set to track saved keys within this run (faster than file existence check)
_seen_pubkeys: Set[str] = set()


def get_public_key_from_private_bytes(pv_bytes: bytes) -> str:
    """
    Private key -> Public key (base58 encode)
    """
    pv = SigningKey(pv_bytes)
    pb_bytes = bytes(pv.verify_key)
    return b58encode(pb_bytes).decode()


def save_keypair(pv_bytes: bytes, output_dir: str, quiet: bool = False) -> str:
    """
    Save private key to JSON file, return public key.
    Skips saving if already seen (deduplication via in-memory set + file check).
    Returns pubkey regardless.
    """
    # Validate private key length
    if len(pv_bytes) != 32:
        logging.warning(f"Invalid private key length: {len(pv_bytes)} (expected 32)")
        return ""
    
    pv = SigningKey(pv_bytes)
    pb_bytes = bytes(pv.verify_key)
    pubkey = b58encode(pb_bytes).decode()
    
    # Fast in-memory deduplication
    if pubkey in _seen_pubkeys:
        return pubkey
    _seen_pubkeys.add(pubkey)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_path = Path(output_dir) / f"{pubkey}.json"
    
    # Skip if file already exists (from previous run)
    if file_path.exists():
        return pubkey
    
    file_path.write_text(json.dumps(list(pv_bytes + pb_bytes)))
    if not quiet:
        logging.info(f"Found: {pubkey}")
    return pubkey
