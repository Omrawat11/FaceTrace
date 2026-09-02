"""Blockchain module for EVM contract interactions and evidence registry anchoring."""

from src.blockchain.abi import EVIDENCE_REGISTRY_ABI
from src.blockchain.base import (
    AnchorResult,
    BlockchainProvider,
    OnChainRecord,
    TransactionReceipt,
)
from src.blockchain.mock import MockBlockchainProvider
from src.blockchain.service import (
    BlockchainError,
    EthereumBlockchainService,
    anchor_evidence,
)
from src.core.config import Settings, get_settings


def get_blockchain_provider(settings: Settings | None = None) -> BlockchainProvider:
    """Factory creating the configured blockchain provider instance."""
    cfg = settings or get_settings()
    provider_name = cfg.BLOCKCHAIN_PROVIDER.lower()

    if provider_name == "local":
        service = EthereumBlockchainService(
            network_name="local",
            contract_address=cfg.EVIDENCE_REGISTRY_CONTRACT_ADDRESS,
            gas_limit=cfg.BLOCKCHAIN_GAS_LIMIT,
        )
        if not service.contract_address:
            # Auto-deploy on local EVM for seamless development
            service.deploy_contract()
        return service

    elif provider_name == "sepolia":
        return EthereumBlockchainService(
            network_name="sepolia",
            contract_address=cfg.EVIDENCE_REGISTRY_CONTRACT_ADDRESS,
            private_key=cfg.ETH_PRIVATE_KEY,
            gas_limit=cfg.BLOCKCHAIN_GAS_LIMIT,
        )

    elif provider_name == "mock":
        return MockBlockchainProvider()

    else:
        raise ValueError(
            f"Unsupported blockchain provider: '{cfg.BLOCKCHAIN_PROVIDER}'. "
            "Must be 'local', 'sepolia', or 'mock'."
        )


__all__ = [
    "AnchorResult",
    "BlockchainProvider",
    "OnChainRecord",
    "TransactionReceipt",
    "BlockchainError",
    "EthereumBlockchainService",
    "MockBlockchainProvider",
    "EVIDENCE_REGISTRY_ABI",
    "anchor_evidence",
    "get_blockchain_provider",
]
