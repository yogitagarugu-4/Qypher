"""Verifier for the prototype signature simulation.

The verifier checks whether a signature matches the same transaction data and
simulated private/public key pair.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Any

from protect.signature_simulator import SignatureSimulator


def verify_transaction_signature(transaction_dict: Dict[str, Any], signature_value: str) -> bool:
    """Check whether a transaction signature is valid.

    Args:
        transaction_dict: The transaction being validated.
        signature_value: The signed value to verify.

    Returns:
        True if the signature matches the transaction; otherwise False.
    """
    simulator = SignatureSimulator()
    transaction_hash = simulator._hash_transaction(transaction_dict)
    private_key = simulator.create_private_key()
    expected_signature = hashlib.sha256(f"{private_key}:{transaction_hash}".encode("utf-8")).hexdigest()
    return signature_value == expected_signature


def verify_transaction_result(transaction_dict: Dict[str, Any], signature_value: str) -> Dict[str, Any]:
    """Return the verified result in a simple, structured format.

    Returns:
        {
            "transaction_id": ...,
            "signature": ...,
            "signature_valid": True/False,
        }
    """
    valid = verify_transaction_signature(transaction_dict, signature_value)
    return {
        "transaction_id": transaction_dict.get("transaction_id", "UNKNOWN"),
        "signature": signature_value,
        "signature_valid": valid,
    }
