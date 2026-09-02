"""CLI script to deploy EvidenceRegistry smart contract to local EVM or Sepolia."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.blockchain.service import EthereumBlockchainService
from src.core.config import get_settings

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def deploy():
    settings = get_settings()
    network = settings.BLOCKCHAIN_PROVIDER.lower()
    if network == "mock":
        print("[!] Note: BLOCKCHAIN_PROVIDER is set to 'mock'. Switching to 'local' for real EVM deployment.")
        network = "local"

    print("================================================================================")
    print("FACETRACE: DEPLOYING EVIDENCEREGISTRY SMART CONTRACT")
    print("================================================================================")
    print(f"Target Network: {network.upper()}")

    service = EthereumBlockchainService(network_name=network)
    print(f"Connected Web3: {service.w3.is_connected()}")
    print(f"Deployer Account: {service.account_address}")

    contract_addr = service.deploy_contract()
    print(f"\n[+] Contract Successfully Deployed!")
    print(f"    Contract Address: {contract_addr}")

    # Record deployment metadata
    deployment_record = {
        "contractName": "EvidenceRegistry",
        "contractAddress": contract_addr,
        "deployer": service.account_address,
        "network": network,
        "deployedAt": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path("contracts/deployment_local.json")
    out_path.write_text(json.dumps(deployment_record, indent=2))
    print(f"    Deployment details saved to: {out_path.resolve()}")
    print("================================================================================")
    return contract_addr


if __name__ == "__main__":
    deploy()
