"""Real Ethereum blockchain provider interacting with EvidenceRegistry on local EVM or Sepolia."""

import json
import logging
from pathlib import Path
from typing import Any
from web3 import Web3
from web3.exceptions import ContractLogicError

from src.blockchain.abi import EVIDENCE_REGISTRY_ABI
from src.blockchain.base import AnchorResult, BlockchainProvider, OnChainRecord, TransactionReceipt
from src.core.config import Settings, get_settings
from src.evidence.canonical import to_bytes32_hex

logger = logging.getLogger(__name__)

CONTRACT_ARTIFACT_PATH = Path("contracts/EvidenceRegistry.json")


class BlockchainError(Exception):
    """Exception raised for Ethereum / smart contract interaction errors."""

    pass


class EthereumBlockchainService(BlockchainProvider):
    """Production Ethereum blockchain provider interfacing with EvidenceRegistry.sol."""

    def __init__(
        self,
        w3: Web3 | None = None,
        contract_address: str | None = None,
        private_key: str | None = None,
        network_name: str = "local",
        gas_limit: int = 250000,
    ) -> None:
        settings = get_settings()
        self._network_name = network_name
        self.gas_limit = gas_limit or settings.BLOCKCHAIN_GAS_LIMIT
        self.private_key = private_key or settings.ETH_PRIVATE_KEY
        self.contract_address = contract_address or settings.EVIDENCE_REGISTRY_CONTRACT_ADDRESS

        # If contract_address is still unset on local network, attempt to read from local deployment artifact
        if not self.contract_address and network_name == "local":
            dep_file = Path("contracts/deployment_local.json")
            if dep_file.exists():
                try:
                    dep_data = json.loads(dep_file.read_text())
                    self.contract_address = dep_data.get("contractAddress")
                except Exception:
                    pass

        # 1. Initialize Web3 connection
        if w3 is not None:
            self.w3 = w3
        elif network_name == "local":
            self.w3 = self._init_local_web3()
        elif network_name == "sepolia":
            rpc_url = settings.SEPOLIA_RPC_URL
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            if not self.w3.is_connected():
                raise BlockchainError(f"Failed to connect to Sepolia RPC endpoint: {rpc_url}")
        else:
            raise ValueError(f"Unknown network_name: {network_name}")

        # 2. Setup Default Account
        if self.private_key:
            account = self.w3.eth.account.from_key(self.private_key)
            self.account_address = account.address
        elif self.w3.eth.accounts:
            self.account_address = self.w3.eth.accounts[0]
        else:
            self.account_address = None

        # 3. Contract setup
        self._contract = None
        if self.contract_address:
            try:
                code = self.w3.eth.get_code(Web3.to_checksum_address(self.contract_address))
                if len(code) > 0:
                    self._init_contract(self.contract_address)
                elif network_name == "local":
                    self.deploy_contract()
            except Exception:
                if network_name == "local":
                    self.deploy_contract()
        elif network_name == "local":
            self.deploy_contract()

    @property
    def network_name(self) -> str:
        return self._network_name

    @property
    def contract(self) -> Any:
        if self._contract is None:
            raise BlockchainError(
                "EvidenceRegistry contract is not deployed or configured. "
                "Call deploy_contract() or set EVIDENCE_REGISTRY_CONTRACT_ADDRESS in .env"
            )
        return self._contract

    def _init_local_web3(self) -> Web3:
        """Connect to running local node (e.g. Anvil/Hardhat) or initialize in-process EthereumTester."""
        # Try local RPC port 8545 first
        try:
            http_w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
            if http_w3.is_connected():
                logger.info("Connected to local Ethereum RPC node on http://127.0.0.1:8545")
                return http_w3
        except Exception:
            pass

        # Fallback to in-process EthereumTesterProvider (full EVM)
        logger.info("Initializing in-process EthereumTesterProvider EVM...")
        from web3 import EthereumTesterProvider

        tester_w3 = Web3(EthereumTesterProvider())
        return tester_w3

    def _init_contract(self, address: str) -> None:
        """Instantiate Web3 contract object for deployed address."""
        checksum_address = Web3.to_checksum_address(address)
        self.contract_address = checksum_address
        self._contract = self.w3.eth.contract(address=checksum_address, abi=EVIDENCE_REGISTRY_ABI)

    def deploy_contract(self) -> str:
        """Compile/load EvidenceRegistry bytecode and deploy to the connected network."""
        if not CONTRACT_ARTIFACT_PATH.exists():
            raise BlockchainError(f"Compiled artifact not found at {CONTRACT_ARTIFACT_PATH}")

        artifact = json.loads(CONTRACT_ARTIFACT_PATH.read_text())
        bytecode = artifact.get("bytecode")
        abi = artifact.get("abi", EVIDENCE_REGISTRY_ABI)

        if not bytecode:
            raise BlockchainError("Contract bytecode is empty in artifact.")

        contract_cls = self.w3.eth.contract(abi=abi, bytecode=bytecode)

        if not self.account_address:
            raise BlockchainError("No sender account available to deploy contract.")

        logger.info("Deploying EvidenceRegistry from %s...", self.account_address)

        if self.private_key:
            # Sign and send deployment transaction
            nonce = self.w3.eth.get_transaction_count(self.account_address)
            tx = contract_cls.constructor().build_transaction({
                "from": self.account_address,
                "nonce": nonce,
                "gas": self.gas_limit,
                "gasPrice": self.w3.eth.gas_price,
            })
            signed = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        else:
            # Direct transact via active provider account (e.g. local tester/node)
            tx_hash = contract_cls.constructor().transact({"from": self.account_address})

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status != 1:
            raise BlockchainError(f"Contract deployment failed (receipt status {receipt.status})")

        deployed_addr = receipt.contractAddress
        logger.info("EvidenceRegistry successfully deployed at: %s (tx: %s)", deployed_addr, receipt.transactionHash.hex())
        self._init_contract(deployed_addr)
        return deployed_addr

    def record_fingerprint(
        self,
        fingerprint: str,
        source_url: str,
    ) -> TransactionReceipt:
        """Submit an on-chain transaction anchoring the 32-byte evidence fingerprint.

        Args:
            fingerprint: 64-hex-char or 0x-prefixed 32-byte SHA-256 fingerprint.
            source_url: Legitimate public URL where matching evidence was discovered.

        Returns:
            TransactionReceipt with confirmation status and mined block details.
        """
        # Format as 0x-prefixed bytes32
        b32_hex = to_bytes32_hex(fingerprint)
        b32_bytes = Web3.to_bytes(hexstr=b32_hex)

        if not self.account_address:
            raise BlockchainError("No signing account available for recording transaction.")

        logger.info("Recording evidence fingerprint %s on-chain...", b32_hex)

        try:
            if self.private_key:
                nonce = self.w3.eth.get_transaction_count(self.account_address)
                tx_data = self.contract.functions.recordEvidence(b32_bytes, source_url).build_transaction({
                    "from": self.account_address,
                    "nonce": nonce,
                    "gas": self.gas_limit,
                    "gasPrice": self.w3.eth.gas_price,
                })
                signed_tx = self.w3.eth.account.sign_transaction(tx_data, private_key=self.private_key)
                raw_tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            else:
                raw_tx_hash = self.contract.functions.recordEvidence(b32_bytes, source_url).transact({
                    "from": self.account_address,
                    "gas": self.gas_limit,
                })

            w3_receipt = self.w3.eth.wait_for_transaction_receipt(raw_tx_hash)
        except ContractLogicError as exc:
            raise BlockchainError(f"Smart contract reverted: {exc}") from exc
        except Exception as exc:
            raise BlockchainError(f"Transaction failed: {exc}") from exc

        if w3_receipt.status != 1:
            raise BlockchainError(
                f"Transaction reverted on-chain (status {w3_receipt.status}, tx {w3_receipt.transactionHash.hex()})"
            )

        tx_receipt = TransactionReceipt(
            tx_hash=w3_receipt.transactionHash.hex(),
            block_number=w3_receipt.blockNumber,
            block_hash=w3_receipt.blockHash.hex(),
            gas_used=w3_receipt.gasUsed,
            status=w3_receipt.status,
        )

        logger.info(
            "Evidence fingerprint recorded in tx %s (Block #%d, status=%d)",
            tx_receipt.tx_hash,
            tx_receipt.block_number,
            tx_receipt.status,
        )
        return tx_receipt

    def get_evidence(self, fingerprint: str) -> OnChainRecord:
        """Query the smart contract for an existing fingerprint record (view call, zero gas)."""
        b32_hex = to_bytes32_hex(fingerprint)
        b32_bytes = Web3.to_bytes(hexstr=b32_hex)

        try:
            exists, recorder, timestamp, block_number, source_url = self.contract.functions.getEvidence(b32_bytes).call()
            return OnChainRecord(
                exists=exists,
                fingerprint=b32_hex,
                recorder=recorder,
                timestamp=int(timestamp),
                block_number=int(block_number),
                source_url=source_url,
            )
        except Exception as exc:
            logger.error("Failed to query getEvidence from contract: %s", exc)
            raise BlockchainError(f"Failed to query on-chain evidence: {exc}") from exc

    def has_evidence(self, fingerprint: str) -> bool:
        """Check if the fingerprint exists in the smart contract registry."""
        b32_hex = to_bytes32_hex(fingerprint)
        b32_bytes = Web3.to_bytes(hexstr=b32_hex)

        try:
            return bool(self.contract.functions.hasEvidence(b32_bytes).call())
        except Exception as exc:
            logger.error("Failed to query hasEvidence from contract: %s", exc)
            raise BlockchainError(f"Failed to query on-chain evidence existence: {exc}") from exc

    def anchor_evidence(
        self,
        evidence_hash: str,
        source_url: str = "",
    ) -> AnchorResult:
        """Full transaction lifecycle: submit, await confirmation receipt, verify on-chain storage.

        Args:
            evidence_hash: 64-character hex or 0x-prefixed 32-byte evidence SHA-256 fingerprint.
            source_url: Discovery URL where matching candidate was located.

        Returns:
            AnchorResult containing success flag, transaction hash, mined block,
            contract address, confirmed on-chain evidence hash, and timestamp.
        """
        contract_addr = self.contract_address or "Unknown"
        b32_hex = to_bytes32_hex(evidence_hash)

        try:
            # 1. Submit transaction & wait for mined receipt
            receipt = self.record_fingerprint(b32_hex, source_url)

            # 2. Confirm transaction status
            if not receipt.is_success:
                return AnchorResult(
                    success=False,
                    tx_hash=receipt.tx_hash,
                    block_number=receipt.block_number,
                    contract_address=contract_addr,
                    stored_evidence_hash="",
                    timestamp=None,
                    status=receipt.status,
                    error=f"Transaction reverted on-chain (status {receipt.status})",
                )

            # 3. Retrieve and confirm stored on-chain record
            on_chain = self.get_evidence(b32_hex)
            if not on_chain.exists:
                return AnchorResult(
                    success=False,
                    tx_hash=receipt.tx_hash,
                    block_number=receipt.block_number,
                    contract_address=contract_addr,
                    stored_evidence_hash="",
                    timestamp=None,
                    status=receipt.status,
                    error="Transaction succeeded but evidence was not retrievable on-chain",
                )

            return AnchorResult(
                success=True,
                tx_hash=receipt.tx_hash,
                block_number=receipt.block_number,
                contract_address=contract_addr,
                stored_evidence_hash=on_chain.fingerprint,
                timestamp=on_chain.timestamp,
                status=receipt.status,
                error=None,
            )
        except BlockchainError as exc:
            logger.error("Evidence anchoring failed with BlockchainError: %s", exc)
            return AnchorResult(
                success=False,
                tx_hash="",
                block_number=0,
                contract_address=contract_addr,
                stored_evidence_hash="",
                timestamp=None,
                status=0,
                error=str(exc),
            )
        except Exception as exc:
            logger.error("Unexpected error anchoring evidence: %s", exc)
            return AnchorResult(
                success=False,
                tx_hash="",
                block_number=0,
                contract_address=contract_addr,
                stored_evidence_hash="",
                timestamp=None,
                status=0,
                error=str(exc),
            )


def anchor_evidence(
    evidence_hash: str,
    source_url: str = "",
    provider: BlockchainProvider | None = None,
) -> AnchorResult:
    """Convenience function to anchor an evidence fingerprint using the active blockchain provider."""
    from src.blockchain import get_blockchain_provider

    active_provider = provider or get_blockchain_provider()
    return active_provider.anchor_evidence(evidence_hash, source_url)

