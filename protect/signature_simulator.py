"""Prototype digital signature simulation for Qypher.

This file does NOT claim to be production-grade ML-DSA. It is a beginner-friendly,
student prototype for a post-quantum style workflow.

Important:
- We use SHA-256 as a cryptographic hash of the transaction data.
- We simulate a private/public signing flow so the prototype is runnable without
  quantum hardware or external PQC libraries.
- If a real ML-DSA library is available later, this code can be replaced with a real
  implementation while keeping the same interface.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Any


class SignatureSimulator:
    """Simulate a post-quantum style signature workflow.

    This implementation intentionally keeps the idea simple and explainable:
    1. Hash the transaction data with SHA-256.
    2. Simulate a private key and public key.
    3. Create a signature by combining the private key seed and transaction hash.
    4. Verification checks whether the same transaction hash matches the signature.

    This is a prototype only and not a real ML-DSA implementation.
    """

    def __init__(self, private_key_seed: str = "qypher-prototype-seed"):
        self.private_key_seed = private_key_seed

    def _hash_transaction(self, transaction_dict: Dict[str, Any]) -> str:
        """Create a stable SHA-256 hash of the transaction data."""
        normalized = str(sorted(transaction_dict.items())).encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    def create_private_key(self) -> str:
        """Return a simple simulated private key built from a seed."""
        return hashlib.sha256(self.private_key_seed.encode("utf-8")).hexdigest()

    def create_public_key(self, private_key: str) -> str:
        """Create a simulated public key from the private key."""
        return hashlib.sha256(private_key.encode("utf-8")).hexdigest()

    def sign_transaction(self, transaction_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simulated signature for a transaction.

        Returns:
            {
                "transaction_id": ..., 
                "signature": "...",
                "public_key": "..."
            }
        """
        transaction_id = str(transaction_dict.get("transaction_id", "UNKNOWN"))
        transaction_hash = self._hash_transaction(transaction_dict)
        private_key = self.create_private_key()
        public_key = self.create_public_key(private_key)

        # This is a prototype signing simulation. It is not real ML-DSA.
        # We are not claiming a new quantum-safe signature algorithm is being used here.
        signature = hashlib.sha256(f"{private_key}:{transaction_hash}".encode("utf-8")).hexdigest()

        return {
            "transaction_id": transaction_id,
            "signature": signature,
            "public_key": public_key,
        }


def simulate_signature_for_transaction(transaction_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Public helper to sign a transaction."""
    simulator = SignatureSimulator()
    return simulator.sign_transaction(transaction_dict)
